"""Unit tests for the SKT A.X-K2 (AXK2ForCausalLM) and A.X-K3 (AXK3ForCausalLM) ports.

Covers the pieces that can be verified without a GPU: config/arch
registration, checkpoint weight-name remapping, the numerics of the
modules A.X-K2 adds on top of DeepSeek-V3.2 (gated RMSNorm, the concatenated
q_a_layernorm, the fused query/output-gate projection handshake with o_proj,
and the indexer wq_b slice), and the A.X-K2-NVFP4 (ModelOpt MIXED_PRECISION)
load path: per-layer quant resolution and the per-tensor FP8 weight rewrites.
For A.X-K3: the config routing to the shared top-k pattern, the gated
residual's checkpoint mapping and numerics, and the layer conversion.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import (  # noqa: E402
    dsa_layer_skips_topk,
    is_deepseek_dsa,
)
from sglang.srt.layers.hyperconnection import (  # noqa: E402
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.layer_boundary.factories import layer_stack  # noqa: E402
from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.layers.quantization.modelopt_quant import (  # noqa: E402
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader  # noqa: E402
from sglang.srt.models.axk2 import (  # noqa: E402
    AXK2ForCausalLM,
    AXK2GatedRMSNorm,
    _AXK2ConcatQANorm,
    _AXK2FusedQGateProj,
    _AXK2GatedOProj,
    _AXK2SlicedIndexerWqB,
    _build_axk2_stages,
    _convert_decoder_layer_to_axk2,
    _dequant_modelopt_fp8_indexer_wk,
    _unify_modelopt_fp8_qkv_a_scales,
    rebind_layer_stack,
)
from sglang.srt.models.axk3 import (  # noqa: E402
    AXK3ForCausalLM,
    _remap_gated_residual_weights,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY  # noqa: E402


def _tiny_axk2_config(**overrides):
    fields = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        architectures=["AXK2ForCausalLM"],
        index_topk=16,
        index_n_heads=2,
        index_head_dim=8,
        gated_norm_rank=4,
        topk_method="noaux_tc",
        n_group=2,
        topk_group=2,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    fields.update(overrides)
    return _CONFIG_REGISTRY["axk2"](**fields)


class TestAXK2Registration(CustomTestCase):
    def test_config_registry_has_axk2_alias(self):
        self.assertIn("axk2", _CONFIG_REGISTRY)
        self.assertEqual(_CONFIG_REGISTRY["axk2"].model_type, "axk2")

    def test_axk2_is_deepseek_dsa(self):
        config = SimpleNamespace(architectures=["AXK2ForCausalLM"], index_topk=2048)
        self.assertTrue(is_deepseek_dsa(config))

    def test_axk2_without_index_topk_is_not_dsa(self):
        config = SimpleNamespace(architectures=["AXK2ForCausalLM"])
        self.assertFalse(is_deepseek_dsa(config))


class TestAXK2WeightNameRemap(CustomTestCase):
    def _remap(self, names):
        weights = [(name, torch.zeros(1)) for name in names]
        return [name for name, _ in AXK2ForCausalLM._remap_axk2_names(weights)]

    def test_official_checkpoint_names(self):
        remapped = self._remap(
            [
                "model.layers.3.self_attn.q_b_proj.weight",
                "model.layers.3.input_layernorm.norm.weight",
                "model.layers.3.input_layernorm.W_up.weight",
                "model.layers.3.input_layernorm.W_down.weight",
                "model.layers.3.post_attention_layernorm.norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",
                "model.layers.3.self_attn.indexer.wq_b.weight",
            ]
        )
        self.assertEqual(
            remapped,
            [
                "model.layers.3.self_attn.q_b_proj.weight",
                "model.layers.3.input_layernorm.base_norm.weight",
                "model.layers.3.input_layernorm.W_up.weight",
                "model.layers.3.input_layernorm.W_down.weight",
                "model.layers.3.post_attention_layernorm.base_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",  # the final norm must NOT be touched
                "model.layers.3.self_attn.indexer.wq_b.weight",
            ],
        )

    def test_transformers_resaved_names(self):
        remapped = self._remap(
            [
                "model.layers.3.self_attn.q_gate_proj.weight",
                "model.layers.3.input_layernorm.mlp.fc1.weight",
                "model.layers.3.input_layernorm.mlp.fc2.weight",
                "model.layers.3.post_attention_layernorm.mlp.fc1.weight",
            ]
        )
        self.assertEqual(
            remapped,
            [
                "model.layers.3.self_attn.q_b_proj.weight",
                "model.layers.3.input_layernorm.W_down.weight",
                "model.layers.3.input_layernorm.W_up.weight",
                "model.layers.3.post_attention_layernorm.W_down.weight",
            ],
        )


class TestAXK2GatedRMSNorm(CustomTestCase):
    def _make_norm(self, hidden=32, rank=4, eps=1e-6):
        torch.manual_seed(0)
        norm = AXK2GatedRMSNorm(hidden, gate_rank=rank, eps=eps)
        # Pin the platform dispatch so the reference comparison does not
        # depend on the CI runner (e.g. AMX availability).
        norm.base_norm._forward_method = norm.base_norm.forward_native
        with torch.no_grad():
            norm.base_norm.weight.normal_(mean=1.0, std=0.1)
            norm.W_down.weight.normal_(std=0.5)
            norm.W_up.weight.normal_(std=0.5)
        return norm

    def _reference_gate(self, norm, y):
        gate = F.linear(F.silu(F.linear(y, norm.W_down.weight)), norm.W_up.weight)
        return (y * torch.sigmoid(gate.float())).to(y.dtype)

    def test_forward_matches_reference(self):
        norm = self._make_norm()
        x = torch.randn(5, 32)
        expected = self._reference_gate(norm, norm.base_norm.forward_native(x.clone()))
        torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-5)

    def test_fused_residual_form(self):
        norm = self._make_norm()
        x, residual = torch.randn(5, 32), torch.randn(5, 32)
        out, res_out = norm(x.clone(), residual.clone())
        expected_res = x + residual
        expected = self._reference_gate(
            norm, norm.base_norm.forward_native(expected_res.clone())
        )
        torch.testing.assert_close(res_out, expected_res, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_no_bare_norm_attributes(self):
        # The stage boundaries' fused-norm fast paths key on a bare `weight` /
        # `variance_epsilon`; exposing them would let a caller skip the gate.
        norm = self._make_norm()
        self.assertFalse(hasattr(norm, "weight"))
        self.assertFalse(hasattr(norm, "variance_epsilon"))

    def test_allreduce_fusion_applies_gate(self):
        # The fused AR+residual+RMSNorm result must still go through the gate.
        norm = self._make_norm()
        x, residual = torch.randn(5, 32), torch.randn(5, 32)
        expected_out, expected_res = norm(x.clone(), residual.clone())
        norm.base_norm.forward_with_allreduce_fusion = (
            lambda x, r, p=None, use_attn_tp_group=True: norm.base_norm.forward_native(
                x, r, p
            )
        )
        out, res_out = norm.forward_with_allreduce_fusion(x.clone(), residual.clone())
        torch.testing.assert_close(res_out, expected_res, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)


class TestAXK2ConcatQANorm(CustomTestCase):
    def test_concat_layout_is_normed_then_raw(self):
        torch.manual_seed(0)
        norm = _AXK2ConcatQANorm(8, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(mean=1.0, std=0.1)
        x = torch.randn(3, 8)
        out = norm.forward_native(x)
        self.assertEqual(out.shape, (3, 16))
        torch.testing.assert_close(out[:, :8], RMSNorm.forward_native(norm, x))
        torch.testing.assert_close(out[:, 8:], x)


class TestAXK2FusedQGateProj(CustomTestCase):
    def setUp(self):
        # The linears resolve their TP group at construction, and
        # RowParallelLinear.forward reads it for symmetric-memory allocation;
        # no process group exists in this single-process test.
        enter_scope(self, published_topology())
        enter_scope(self, get_parallel().override(tp_group=None))

    def _make_pair(self):
        torch.manual_seed(0)
        proj = _AXK2FusedQGateProj(
            q_lora_rank=8,
            num_heads=2,
            qk_head_dim=6,
            v_head_dim=4,
            quant_config=None,
            prefix="",
            parallel_group="tp",
        )
        o_proj = _AXK2GatedOProj(
            2 * 4,
            16,
            bias=False,
            quant_config=None,
            reduce_results=False,
            prefix="",
        )
        o_proj._take_gate = proj.take_gate
        with torch.no_grad():
            proj.weight.normal_(std=0.1)
            o_proj.weight.normal_(std=0.1)
        return proj, o_proj

    def test_query_gate_split_and_o_proj_gating(self):
        proj, o_proj = self._make_pair()
        x = torch.randn(3, 16)  # 2 * q_lora_rank
        q, _ = proj(x)
        self.assertEqual(q.shape, (3, 2, 6))

        fused = F.linear(x, proj.weight).view(3, 2, 10)
        torch.testing.assert_close(q, fused[..., :6])

        attn_output = torch.randn(3, 8)  # num_heads * v_head_dim
        out, _ = o_proj(attn_output)
        gate = fused[..., 6:].reshape(3, 8)
        expected_in = attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)
        expected = F.linear(expected_in, o_proj.weight)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        self.assertIsNone(proj._pending_gate)

    def test_o_proj_without_stashed_gate_fails_loudly(self):
        proj, o_proj = self._make_pair()
        with self.assertRaises(AssertionError):
            o_proj(torch.randn(3, 8))

    def test_double_stash_fails_loudly(self):
        proj, _ = self._make_pair()
        x = torch.randn(3, 16)
        proj(x)
        with self.assertRaises(AssertionError):
            proj(x)


class TestAXK2DecoderLayerConversion(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def _make_config(self):
        return _tiny_axk2_config()

    def test_stage_boundaries_bind_the_gated_norms(self):
        # The stage boundaries bind their norms when built; swapping only the
        # layer attributes would leave the layer running the ungated RMSNorm.
        config = self._make_config()
        model = _build_deepseek_layers(config)
        for layer_id in range(config.num_hidden_layers):
            _convert_decoder_layer_to_axk2(
                model.layers[layer_id],
                config=config,
                quant_config=None,
                prefix=f"model.layers.{layer_id}",
            )
        rebind_layer_stack(
            model,
            stage_facts=partial(DeepseekV2DecoderLayer.stage_facts, config),
            build_stages=_build_axk2_stages,
        )
        # Layer 0 is dense (plain post_attention_layernorm), layer 1 is MoE.
        for layer_id, ffn_norm_type in ((0, RMSNorm), (1, AXK2GatedRMSNorm)):
            with self.subTest(layer_id=layer_id):
                layer = model.layers[layer_id]
                self.assertIsInstance(layer.input_layernorm, AXK2GatedRMSNorm)
                self.assertIs(layer.attn_boundary.norm, layer.input_layernorm)
                self.assertIsInstance(layer.post_attention_layernorm, ffn_norm_type)
                self.assertIs(layer.ffn_boundary.norm, layer.post_attention_layernorm)


def _build_deepseek_layers(config, layer_cls=None):
    """The DeepSeek decoder layers of ``config`` built in one layer stack, as
    make_layers builds them, held like DeepseekV2Model holds them."""
    layer_cls = layer_cls or DeepseekV2DecoderLayer
    layers = []
    # No GPU platform on the CPU runner: keep rope off its kernel path.
    with patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True), layer_stack():
        for layer_id in range(config.num_hidden_layers):
            layers.append(
                layer_cls(config, layer_id, prefix=f"model.layers.{layer_id}")
            )
    return SimpleNamespace(
        layers=layers, start_layer=0, end_layer=config.num_hidden_layers
    )


class TestAXK2SlicedIndexerWqB(CustomTestCase):
    def test_consumes_normed_half(self):
        """The indexer reads the normed q-LoRA (the leading half of the concat),
        as transformers' AXK2Attention does; the raw half changes its top-k."""
        torch.manual_seed(0)
        wq_b = _AXK2SlicedIndexerWqB(4, 6, bias=False, quant_config=None, prefix="")
        with torch.no_grad():
            wq_b.weight.normal_(std=0.1)
        x = torch.randn(3, 8)
        out, _ = wq_b(x)
        torch.testing.assert_close(out, F.linear(x[:, :4], wq_b.weight))

    def test_rejects_unconcatenated_input(self):
        wq_b = _AXK2SlicedIndexerWqB(4, 6, bias=False, quant_config=None, prefix="")
        with self.assertRaises(AssertionError):
            wq_b(torch.randn(3, 4))


class TestAXK2Nvfp4QuantResolution(CustomTestCase):
    """A.X-K2-NVFP4 lists the dense / shared-expert MLP by its checkpoint
    shards (gate_proj, up_proj); DeepSeek builds a fused gate_up_proj it has
    no packed mapping for."""

    def setUp(self):
        fp8 = {"quant_algo": "FP8"}
        nvfp4 = {"quant_algo": "NVFP4", "group_size": 16}
        layers = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            layers[f"model.layers.0.mlp.{proj}"] = fp8
            layers[f"model.layers.1.mlp.shared_experts.{proj}"] = fp8
            layers[f"model.layers.1.mlp.experts.0.{proj}"] = nvfp4
        for proj in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wq_b"):
            layers[f"model.layers.1.self_attn.{proj}"] = fp8
        self.config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": ["lm_head", "*.mlp.gate", "*norm*"],
                "quantized_layers": layers,
                "packed_modules_mapping": {
                    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"]
                },
            }
        )

    def test_unmapped_gate_up_proj_resolves_through_shards(self):
        for prefix in (
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.1.mlp.shared_experts.gate_up_proj",
        ):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.config.resolve_quant_algo(prefix), "FP8")

    def test_other_modules_keep_their_scheme(self):
        cases = {
            "model.layers.0.mlp.down_proj": "FP8",
            "model.layers.1.self_attn.fused_qkv_a_proj_with_mqa": "FP8",
            "model.layers.1.self_attn.indexer.wq_b": "FP8",
            "model.layers.1.mlp.experts": "NVFP4",
            "model.layers.1.mlp.gate": None,
        }
        for prefix, algo in cases.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(self.config.resolve_quant_algo(prefix), algo)

    def test_mixed_shards_are_rejected(self):
        self.config.quantized_layers["model.layers.0.mlp.up_proj"] = {
            "quant_algo": "NVFP4"
        }
        with self.assertRaises(ValueError):
            self.config.resolve_quant_algo("model.layers.0.mlp.gate_up_proj")


def _fp8(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn)


class TestAXK2ModelOptFp8QkvAScales(CustomTestCase):
    """The DeepSeek loader concatenates q_a_proj and kv_a_proj_with_mqa and keeps
    only q_a_proj's scalar scales; after the rewrite that must be exact enough
    for both halves."""

    prefix = "model.layers.3"

    def _stream(self, order):
        q_w, kv_w = _fp8([[1.0, -2.0], [0.5, 4.0]]), _fp8([[3.0, -1.5]])
        tensors = {
            ("q_a_proj", "weight"): q_w,
            ("q_a_proj", "weight_scale"): torch.tensor(0.5),
            ("q_a_proj", "input_scale"): torch.tensor(0.25),
            ("kv_a_proj_with_mqa", "weight"): kv_w,
            ("kv_a_proj_with_mqa", "weight_scale"): torch.tensor(2.0),
            ("kv_a_proj_with_mqa", "input_scale"): torch.tensor(0.75),
        }
        keys = sorted(tensors, reverse=order == "reversed")
        stream = [
            (f"{self.prefix}.self_attn.{p}.{s}", tensors[(p, s)]) for p, s in keys
        ]
        other = (f"{self.prefix}.self_attn.q_b_proj.weight", torch.zeros(2, 2))
        return [other] + stream, q_w, kv_w, other

    def test_rewrites_to_one_scale_in_any_order(self):
        for order in ("sorted", "reversed"):
            with self.subTest(order=order):
                stream, q_w, kv_w, other = self._stream(order)
                out = list(_unify_modelopt_fp8_qkv_a_scales(stream))
                self.assertEqual(len(out), len(stream))
                self.assertIs(dict(out)[other[0]], other[1])
                got = {tuple(n.rsplit(".", 2)[-2:]): t for n, t in out if n != other[0]}
                for key in ("weight_scale", "input_scale"):
                    self.assertEqual(
                        got[("q_a_proj", key)], got[("kv_a_proj_with_mqa", key)]
                    )
                self.assertEqual(got[("q_a_proj", "weight_scale")].item(), 2.0)
                self.assertEqual(got[("q_a_proj", "input_scale")].item(), 0.75)
                # What the fused loader builds: cat(weights) * q_a_proj's scale.
                fused = (
                    torch.cat(
                        [
                            got[("q_a_proj", "weight")],
                            got[("kv_a_proj_with_mqa", "weight")],
                        ]
                    ).float()
                    * got[("q_a_proj", "weight_scale")]
                )
                expected = torch.cat([q_w.float() * 0.5, kv_w.float() * 2.0])
                torch.testing.assert_close(fused, expected)

    def test_block_scaled_and_bf16_layers_pass_through(self):
        stream = [
            ("model.layers.3.self_attn.q_a_proj.weight", _fp8([[1.0]])),
            ("model.layers.3.self_attn.q_a_proj.weight_scale_inv", torch.ones(1, 1)),
            ("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", _fp8([[2.0]])),
            (
                "model.layers.3.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
                torch.ones(1, 1),
            ),
            ("model.layers.4.self_attn.q_a_proj.weight", torch.ones(1, 1)),
            ("model.layers.4.self_attn.kv_a_proj_with_mqa.weight", torch.ones(1, 1)),
        ]
        out = dict(_unify_modelopt_fp8_qkv_a_scales(stream))
        self.assertEqual(sorted(out), sorted(name for name, _ in stream))
        for name, tensor in stream:
            self.assertIs(out[name], tensor)


class TestAXK2ModelOptFp8IndexerWk(CustomTestCase):
    prefix = "model.layers.3.self_attn.indexer"

    def test_per_tensor_wk_becomes_bf16_in_any_order(self):
        weight, scale = _fp8([[1.0, -2.0], [0.5, 3.0]]), torch.tensor(0.125)
        for scale_first in (False, True):
            with self.subTest(scale_first=scale_first):
                pair = [
                    (f"{self.prefix}.wk.weight", weight),
                    (f"{self.prefix}.wk.weight_scale", scale),
                ]
                if scale_first:
                    pair.reverse()
                stream = [
                    (f"{self.prefix}.wq_b.weight", torch.zeros(2, 2)),
                    (f"{self.prefix}.wk.input_scale", torch.tensor(1.0)),
                    *pair,
                    (f"{self.prefix}.weights_proj.weight", torch.zeros(4, 2)),
                ]
                out = list(_dequant_modelopt_fp8_indexer_wk(stream))
                self.assertEqual(
                    [name for name, _ in out],
                    [
                        f"{self.prefix}.wq_b.weight",
                        f"{self.prefix}.wk.weight",
                        f"{self.prefix}.weights_proj.weight",
                    ],
                )
                wk = dict(out)[f"{self.prefix}.wk.weight"]
                self.assertEqual(wk.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    wk, (weight.float() * scale).to(torch.bfloat16)
                )

    def test_block_scaled_and_bf16_wk_pass_through(self):
        stream = [
            (f"{self.prefix}.wk.weight", _fp8([[1.0]])),
            (f"{self.prefix}.wk.weight_scale_inv", torch.ones(1, 1)),
            ("model.layers.4.self_attn.indexer.wk.weight", torch.zeros(2, 2)),
        ]
        out = dict(_dequant_modelopt_fp8_indexer_wk(stream))
        self.assertEqual(sorted(out), sorted(name for name, _ in stream))
        for name, tensor in stream:
            self.assertIs(out[name], tensor)


def _tiny_axk3_config(**overrides):
    """A.X-K3 Flash in miniature: one indexer layer then three that share its
    top-k, all MoE, on a gated residual stream."""
    fields = dict(
        num_hidden_layers=4,
        first_k_dense_replace=0,
        layer_types=["full_attention"] + ["sliding_attention"] * 3,
        swa_window_size=256,
        dsa_index_share=True,
        gated_residual=True,
        gated_residual_num_streams=2,
        gated_residual_rank=4,
        gated_residual_final="gated_read",
        attention_output_gate=True,
        gated_norm=True,
    )
    fields.update(overrides)
    return _tiny_axk2_config(**fields)


class TestAXK3Config(CustomTestCase):
    def test_k3_flags_route_to_axk3_with_a_shared_topk_pattern(self):
        config = _tiny_axk3_config()
        self.assertEqual(config.architectures, ["AXK3ForCausalLM"])
        self.assertTrue(is_deepseek_dsa(config))
        self.assertEqual(
            [dsa_layer_skips_topk(config, i) for i in range(4)],
            [False, True, True, True],
        )

    def test_k2_config_keeps_its_architecture(self):
        config = _tiny_axk2_config()
        self.assertEqual(config.architectures, ["AXK2ForCausalLM"])
        self.assertFalse(dsa_layer_skips_topk(config, 1))

    def test_window_shorter_than_the_context_is_rejected(self):
        # Sliding layers run as top-k consumers over the whole context only.
        with self.assertRaisesRegex(ValueError, "swa_window_size"):
            _tiny_axk3_config(swa_window_size=128)


class TestAXK3GatedResidualWeights(CustomTestCase):
    N, H, R = 2, 8, 4

    def _checkpoint(self, prefix):
        torch.manual_seed(0)
        n, h, r = self.N, self.H, self.R
        return {
            f"{prefix}.gamma": 1.0 + 0.25 * torch.randn(n * h),
            f"{prefix}.read_down.weight": torch.randn(r, n * h),
            f"{prefix}.read_up.weight": torch.randn(n * h, r),
            f"{prefix}.write_proj.weight": torch.randn(n, n * h),
            f"{prefix}.write_bias": torch.zeros(n),
        }

    def test_loaded_module_matches_the_checkpoint_formulas(self):
        """The read and write of the checkpoint's AXK2GatedResidual (HF
        modeling_axk2.py): per-branch RMSNorm with gain gamma, then
        x = mean_i(sigmoid(read_up(silu(read_down(U) / n)))_i * U_i) and
        R'_i = R_i + 2 * sigmoid(write_proj(U) / n + write_bias)_i * y."""
        n, h = self.N, self.H
        prefix = "model.layers.0.attn_gated_residual"
        ckpt = self._checkpoint(prefix)
        module = GatedResidual(
            HyperConnectionConfig(
                hc_count=n,
                hidden_size=h,
                params_dtype=torch.float32,
                hc_lowrank=self.R,
                rms_norm_eps=1e-6,
                hc_per_branch_norm=True,
            )
        )
        params = dict(module.named_parameters())
        for name, weight in _remap_gated_residual_weights(ckpt.items()):
            param = params[name.removeprefix(prefix + ".")]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)

        x, y = torch.randn(3, n * h), torch.randn(3, h)
        with torch._dynamo.config.patch(disable=True):
            mixed, residuals = module.mix(x)
            written = module.combine(y, residuals)

        streams = x.view(3, n, h)
        rstd = torch.rsqrt(streams.pow(2).mean(-1, keepdim=True) + 1e-6)
        u = (streams * rstd * ckpt[f"{prefix}.gamma"].view(n, h)).view(3, n * h)
        z = F.silu(u @ ckpt[f"{prefix}.read_down.weight"].T / n)
        gate = torch.sigmoid(z @ ckpt[f"{prefix}.read_up.weight"].T)
        expected_mixed = (gate * u).view(3, n, h).mean(dim=-2)
        write_gate = 2 * torch.sigmoid(
            u @ ckpt[f"{prefix}.write_proj.weight"].T / n + ckpt[f"{prefix}.write_bias"]
        )
        expected_written = (streams + write_gate.unsqueeze(-1) * y.unsqueeze(-2)).view(
            3, n * h
        )
        torch.testing.assert_close(mixed, expected_mixed)
        torch.testing.assert_close(written, expected_written)

    def test_checkpoint_names_and_bf16_gains(self):
        # Every bf16 gain in [0.5, 4.5] (the checkpoint spans 0.50-4.28) must
        # survive the gamma - 1 offset exactly.
        gains = torch.arange(0.5, 4.5, 2**-6).to(torch.bfloat16)
        weights = [
            ("model.final_gated_residual.gamma", gains),
            ("model.final_gated_residual.read_up.weight", torch.zeros(1)),
            ("model.layers.3.mlp_gated_residual.write_proj.weight", torch.zeros(1)),
            ("model.layers.3.mlp_gated_residual.write_bias", torch.zeros(4)),
            ("model.layers.3.self_attn.q_b_proj.weight", torch.zeros(1)),
        ]
        remapped = dict(_remap_gated_residual_weights(weights))
        self.assertEqual(
            list(remapped),
            [
                "model.norm.gated_residual.hc_norm.weight",
                "model.norm.gated_residual.input_mix_weight_up.weight",
                "model.layers.3.mlp_gated_residual.block_inject_weight.weight",
                "model.layers.3.self_attn.q_b_proj.weight",
            ],
        )
        offset = remapped["model.norm.gated_residual.hc_norm.weight"]
        self.assertEqual(offset.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(1.0 + offset.float(), gains.float()))

    def test_nonzero_write_bias_is_rejected(self):
        weights = [("model.layers.0.attn_gated_residual.write_bias", torch.ones(2))]
        with self.assertRaises(NotImplementedError):
            list(_remap_gated_residual_weights(weights))


class TestAXK3DecoderLayerConversion(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_shared_topk_and_gated_residual_layers(self):
        config = _tiny_axk3_config()
        model = _build_deepseek_layers(config)
        for layer_id in range(config.num_hidden_layers):
            AXK3ForCausalLM._convert_decoder_layer(
                model.layers[layer_id],
                config=config,
                quant_config=None,
                prefix=f"model.layers.{layer_id}",
            )
        rebind_layer_stack(
            model,
            stage_facts=partial(AXK3ForCausalLM._layer_stage_facts, config),
            build_stages=AXK3ForCausalLM._build_layer_stages,
        )
        for layer_id, layer in enumerate(model.layers):
            with self.subTest(layer_id=layer_id):
                attn = layer.self_attn
                self.assertIsInstance(attn.q_b_proj, _AXK2FusedQGateProj)
                if layer_id == 0:
                    self.assertIsInstance(attn.indexer.wq_b, _AXK2SlicedIndexerWqB)
                else:
                    self.assertIsNone(attn.indexer)
                # The gated reads normalize the streams; no stage binds a norm.
                self.assertFalse(hasattr(layer, "input_layernorm"))
                self.assertIsNone(layer.attn_boundary.norm)
                self.assertIsNone(layer.ffn_boundary.norm)
                self.assertIsInstance(layer.mlp_gated_residual, GatedResidual)


if __name__ == "__main__":
    unittest.main()
