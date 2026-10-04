import json
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Attention:
    num_kv_heads = 1
    head_dim = 3

    def __init__(self) -> None:
        self.attn = SimpleNamespace(k_scale=0.5, v_scale=0.25)
        self.input = None

    def kv_proj_only(self, hidden_states: torch.Tensor):
        self.input = hidden_states
        return hidden_states + 1, hidden_states + 2

    def apply_k_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 3

    def apply_k_rope(
        self, _positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states + 4


class DSparkTargetHiddenProjectionTest(CustomTestCase):
    def test_nonfinal_prefill_stage_only_forwards_target_proxies(self) -> None:
        proxy = object()
        target = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=object(), spec_algorithm=None),
            device="cpu",
            forward_batch_generation=lambda batch, *, pp_proxy_tensors, capture_hidden_mode: (
                SimpleNamespace(pp_hidden_states_proxy_tensors=pp_proxy_tensors)
            ),
        )
        with (
            get_parallel().override(pp_group=SimpleNamespace(is_last_rank=False)),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
        ):
            worker = DSparkWorkerV2(None, 0, 0, target)
        worker.alloc_memory_pool()
        worker.init_attention_backends()
        worker.init_cuda_graphs()
        batch = SimpleNamespace(seq_lens=torch.tensor([8]))
        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)
        self.assertIs(result.pp_hidden_states_proxy_tensors, proxy)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIsNone(worker.get_confidence_budget_prepare())
        self.assertIsNone(worker.primary_draft_kv_pool)
        self.assertEqual(worker.preloaded_weights_bytes, 0)
        self.assertEqual(
            worker.spec_v2_attn_backends, (target.model_runner.attn_backend,)
        )

    def test_single_aux_hidden_state_is_returned_without_copy(self) -> None:
        hidden_states = torch.empty(2, 3)

        self.assertIs(hidden_states, pack_aux_hidden_states([hidden_states]))

    def test_preprojected_hidden_is_not_projected_again(self) -> None:
        attention = _Attention()
        draft_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
            project_target_hidden=mock.Mock(
                side_effect=AssertionError("projection must not run twice")
            ),
            _fused_kv_write_bundle=lambda _pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, draft_model
        )
        pool = SimpleNamespace(set_kv_buffer=mock.Mock())
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            device=torch.device("cpu"),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        projected_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        injector.inject_target_hidden(
            target_hidden=projected_hidden,
            cache_loc=torch.arange(2),
            positions=torch.arange(2),
            target_hidden_is_projected=True,
        )

        draft_model.project_target_hidden.assert_not_called()
        self.assertIs(attention.input, projected_hidden)


class _TinyBackbone(nn.Module):
    block_size = 3
    is_nemotron_35_draft = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.layers = nn.ModuleList()

    def load_weights(self, weights) -> None:
        self.backbone_weight_names = [name for name, _ in weights]


class _TinyReducedVocabDraft(DSparkDraftMixin, _TinyBackbone):
    pass


class DSparkReducedVocabTest(CustomTestCase):
    VOCAB, HOT_VOCAB, HIDDEN, RANK, GAMMA = 9, 4, 5, 3, 3
    # Draft row i is target token i + D2T[i].
    D2T = [0, 1, 3, 4]
    HOT_TOKEN_ID = [0, 2, 5, 7]

    def _new_draft(self, *, checkpoint_overrides=None):
        torch.manual_seed(0)
        config = SimpleNamespace(
            vocab_size=self.VOCAB,
            draft_vocab_size=self.HOT_VOCAB,
            hidden_size=self.HIDDEN,
            num_hidden_layers=1,
            markov_rank=self.RANK,
            markov_head_type="vanilla",
            mask_token_id=self.VOCAB - 1,
            block_size=self.GAMMA,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
            sample_from_anchor=True,
        )
        checkpoint = {
            "lm_head.weight": torch.randn(self.HOT_VOCAB, self.HIDDEN),
            "d2t": torch.tensor(self.D2T),
            "t2d": torch.zeros(self.VOCAB, dtype=torch.bool),
            "embed_tokens.weight": torch.randn(self.VOCAB, self.HIDDEN),
            "markov_head.markov_w1.weight": torch.randn(self.VOCAB, self.RANK),
            "markov_head.markov_w2.weight": torch.randn(self.HOT_VOCAB, self.RANK),
            "confidence_head.proj.weight": torch.randn(1, self.HIDDEN + self.RANK),
            "confidence_head.proj.bias": torch.randn(1),
        }
        checkpoint.update(checkpoint_overrides or {})
        draft = _TinyReducedVocabDraft(config)
        draft.load_weights(
            [(name, w) for name, w in checkpoint.items() if w is not None]
        )
        draft.attach_shared_modules(
            embed_tokens=nn.Embedding(self.VOCAB, self.HIDDEN),
            lm_head=SimpleNamespace(org_vocab_size=self.VOCAB),
        )
        return draft, checkpoint

    def test_block_matches_draft_space_reference(self) -> None:
        """A reduced-vocab draft proposes exactly what sampling in its own vocab
        and remapping through d2t proposes, in target token ids."""
        draft, checkpoint = self._new_draft()
        hidden = torch.randn(2, self.GAMMA, self.HIDDEN)
        anchor = torch.tensor([1, 8])  # target ids outside the draft vocab

        base_logits, _ = draft.compute_base_logits(hidden)
        tokens, _ = draft.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=hidden,
            sampler=lambda step_logits, step_idx: step_logits.argmax(dim=-1),
        )

        hot_token_id = torch.tensor(self.HOT_TOKEN_ID)
        w1 = checkpoint["markov_head.markov_w1.weight"]
        w2 = checkpoint["markov_head.markov_w2.weight"]
        hot_logits = hidden @ checkpoint["lm_head.weight"].T
        prev = anchor
        expected = []
        for step in range(self.GAMMA):
            step_logits = hot_logits[:, step] + w1[prev] @ w2.T
            prev = hot_token_id[step_logits.argmax(dim=-1)]
            expected.append(prev)
        self.assertEqual(tokens.tolist(), torch.stack(expected, dim=1).tolist())

        outside = torch.ones(self.VOCAB, dtype=torch.bool)
        outside[hot_token_id] = False
        self.assertTrue(torch.isneginf(base_logits[..., outside]).all())
        self.assertNotIn("lm_head.weight", draft.backbone_weight_names)

    def test_checkpoint_without_d2t_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "d2t"):
            self._new_draft(checkpoint_overrides={"d2t": None})


class SpeculatorsDSparkConfigTest(CustomTestCase):
    def _get_config(self, speculators_config: dict):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps(speculators_config))
            return get_config(tmp, trust_remote_code=False)

    def test_unimplemented_draft_extension_is_rejected(self) -> None:
        """A draft that enables an extension module must fail to load rather than
        serve without it."""
        speculators_config = {
            "speculators_model_type": "dspark",
            "aux_hidden_state_layer_ids": [2],
            "transformer_layer_config": {
                "model_type": "qwen3",
                "dspark_autoregressive_draft": True,
                "dspark_query_residual_context": False,
            },
        }
        with self.assertRaisesRegex(ValueError, r"\['dspark_autoregressive_draft'\]"):
            self._get_config(speculators_config)

    def test_speculators_format_is_flattened(self) -> None:
        """A speculators export has no top-level model_type; its nested layer
        config and aux layer ids must land where the DSpark draft model reads."""
        speculators_config = {
            "architectures": ["DSparkDraftModel"],
            "speculators_model_type": "dspark",
            "aux_hidden_state_layer_ids": [2, 30, 58],
            "block_size": 5,
            "draft_vocab_size": 32,
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "mask_token_id": 63,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 3,
                "vocab_size": 64,
                "sliding_window": 8,
                "layer_types": ["sliding_attention"] * 3,
            },
        }
        config = self._get_config(speculators_config)

        self.assertEqual(config.architectures, ["Qwen3DSparkModel"])
        self.assertEqual(config.num_hidden_layers, 3)
        # speculators ids name the layer whose input is captured.
        self.assertEqual(config.target_layer_ids, [1, 29, 57])
        self.assertFalse(config.sample_from_anchor)
        draft_config = parse_dspark_draft_config(draft_hf_config=config)
        self.assertEqual(draft_config.gamma, 5)
        self.assertEqual(draft_config.mask_token_id, 63)
        self.assertEqual(draft_config.markov_rank, 4)
        self.assertEqual(config.draft_vocab_size, 32)


if __name__ == "__main__":
    unittest.main()
