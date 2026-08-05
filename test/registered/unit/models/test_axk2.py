"""Unit tests for the SKT A.X-K2 (AXK2ForCausalLM) port.

Covers the pieces that can be verified without a GPU: config/arch
registration, checkpoint weight-name remapping, and the numerics of the
modules A.X-K2 adds on top of DeepSeek-V3.2 (gated RMSNorm, the concatenated
q_a_layernorm, the fused query/output-gate projection handshake with o_proj,
and the indexer wq_b slice).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import is_deepseek_dsa  # noqa: E402
from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.models.axk2 import (  # noqa: E402
    AXK2ForCausalLM,
    AXK2GatedRMSNorm,
    _AXK2ConcatQANorm,
    _AXK2FusedQGateProj,
    _AXK2GatedOProj,
    _AXK2SlicedIndexerWqB,
)
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY  # noqa: E402


class TestAXK2Registration(CustomTestCase):
    def test_config_registry_has_axk2_alias(self):
        self.assertIn("axk2", _CONFIG_REGISTRY)
        self.assertEqual(_CONFIG_REGISTRY["axk2"].model_type, "axk2")

    def test_axk2_is_deepseek_dsa(self):
        config = SimpleNamespace(
            architectures=["AXK2ForCausalLM"], index_topk=2048
        )
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

    def test_no_fused_norm_fast_path_attributes(self):
        # LayerCommunicator falls back to the plain module call only when the
        # norm exposes neither `forward_with_allreduce_fusion` nor a bare
        # `weight`; if either appears the gate would be silently skipped.
        norm = self._make_norm()
        self.assertFalse(hasattr(norm, "forward_with_allreduce_fusion"))
        self.assertFalse(hasattr(norm, "weight"))
        self.assertFalse(hasattr(norm, "variance_epsilon"))


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
    def _make_pair(self):
        torch.manual_seed(0)
        proj = _AXK2FusedQGateProj(
            q_lora_rank=8,
            num_heads=2,
            qk_head_dim=6,
            v_head_dim=4,
            quant_config=None,
            prefix="",
            tp_rank=0,
            tp_size=1,
        )
        o_proj = _AXK2GatedOProj(
            2 * 4,
            16,
            bias=False,
            quant_config=None,
            reduce_results=False,
            prefix="",
            tp_rank=0,
            tp_size=1,
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


class TestAXK2SlicedIndexerWqB(CustomTestCase):
    def test_consumes_raw_half(self):
        torch.manual_seed(0)
        wq_b = _AXK2SlicedIndexerWqB(4, 6, bias=False, quant_config=None, prefix="")
        with torch.no_grad():
            wq_b.weight.normal_(std=0.1)
        x = torch.randn(3, 8)
        out, _ = wq_b(x)
        torch.testing.assert_close(out, F.linear(x[:, 4:], wq_b.weight))

    def test_rejects_unconcatenated_input(self):
        wq_b = _AXK2SlicedIndexerWqB(4, 6, bias=False, quant_config=None, prefix="")
        with self.assertRaises(AssertionError):
            wq_b(torch.randn(3, 4))


if __name__ == "__main__":
    unittest.main()
