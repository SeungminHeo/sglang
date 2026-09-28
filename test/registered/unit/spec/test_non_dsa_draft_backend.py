"""A draft that is not a DSA model must not inherit the target's ``dsa``
attention backend (``DeepseekSparseAttnBackend`` asserts on non-DSA configs)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _resolve_non_dsa_draft_attention_backend,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"

# A.X-K2's EAGLE3 head: MLA attention, no DSA indexer.
MLA_EAGLE3_DRAFT = SimpleNamespace(
    architectures=["Eagle3DeepseekV2ForCausalLM"], kv_lora_rank=512
)
LLAMA_EAGLE3_DRAFT = SimpleNamespace(architectures=["LlamaForCausalLMEagle3"])
# A NextN/MTP draft loads the DSA target config itself.
DSA_DRAFT = SimpleNamespace(
    architectures=["AXK2ForCausalLM"], kv_lora_rank=512, index_topk=2048
)

SM100 = dict(is_sm100=True, is_sm90=False, is_xpu=False, is_hip=False)
SM90 = dict(is_sm100=False, is_sm90=True, is_xpu=False, is_hip=False)
HIP = dict(is_sm100=False, is_sm90=False, is_xpu=False, is_hip=True)


class TestNonDsaDraftAttentionBackend(CustomTestCase):
    def _resolve(
        self,
        *,
        draft_config,
        target_backends=("dsa", "dsa"),
        platform=SM100,
        draft_backend=None,
    ):
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path="draft",
            speculative_draft_attention_backend=draft_backend,
        )
        with (
            patch(f"{HOOK_MODULE}.attention_backends_of", return_value=target_backends),
            patch(
                f"{HOOK_MODULE}.get_platform",
                return_value=SimpleNamespace(**platform),
            ),
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_config",
                return_value=draft_config,
            ) as get_config,
        ):
            _resolve_non_dsa_draft_attention_backend(args)
        return resolution_result(
            args, "speculative_draft_attention_backend"
        ), get_config

    def test_non_dsa_draft_on_dsa_target_gets_a_draft_backend(self):
        cases = (
            (MLA_EAGLE3_DRAFT, SM100, "trtllm_mla"),
            (MLA_EAGLE3_DRAFT, SM90, "fa3"),
            (MLA_EAGLE3_DRAFT, HIP, "triton"),
            (LLAMA_EAGLE3_DRAFT, SM100, "flashinfer"),
            (LLAMA_EAGLE3_DRAFT, HIP, "triton"),
        )
        for draft_config, platform, expected in cases:
            with self.subTest(draft=draft_config.architectures[0], platform=platform):
                backend, _ = self._resolve(draft_config=draft_config, platform=platform)
                self.assertEqual(backend, expected)

    def test_dsa_draft_keeps_the_target_backend(self):
        backend, _ = self._resolve(draft_config=DSA_DRAFT)
        self.assertIsNone(backend)

    def test_explicit_draft_backend_wins(self):
        backend, get_config = self._resolve(
            draft_config=MLA_EAGLE3_DRAFT, draft_backend="flashinfer"
        )
        self.assertEqual(backend, "flashinfer")
        get_config.assert_not_called()

    def test_non_dsa_target_is_left_alone(self):
        backend, get_config = self._resolve(
            draft_config=MLA_EAGLE3_DRAFT, target_backends=("fa3", "fa3")
        )
        self.assertIsNone(backend)
        get_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
