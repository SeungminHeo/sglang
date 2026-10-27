"""Unit tests for the Solar Open 2 config and checkpoint-name routing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.hybrid_arch import kimi_linear_config
from sglang.srt.configs.solar_open2 import SolarOpen2Config
from sglang.srt.models.solar_open2 import SolarOpen2ForCausalLM
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

# Layer-pattern fields of upstage/Solar-Open2-250B config.json.
_RELEASED_LAYOUT = dict(
    num_hidden_layers=48,
    gqa_interval=3,
    gqa_layers=list(range(0, 48, 4)),
)


class TestSolarOpen2Config(unittest.TestCase):
    def test_released_layout_puts_softmax_on_every_fourth_layer_from_zero(self):
        """gqa_layers must win over gqa_interval=3, which would pick 2, 5, 8, ..."""
        config = SolarOpen2Config(**_RELEASED_LAYOUT)
        self.assertEqual(config.full_attention_layer_ids, list(range(0, 48, 4)))
        self.assertEqual(len(config.linear_layer_ids), 36)
        self.assertTrue(config.kda_allow_neg_eigval)

    def test_gqa_interval_is_one_based_when_gqa_layers_absent(self):
        config = SolarOpen2Config(num_hidden_layers=8, gqa_interval=4)
        self.assertEqual(config.full_attention_layer_ids, [3, 7])

    def test_layer_types_must_agree_with_gqa_layers(self):
        with self.assertRaises(ValueError):
            SolarOpen2Config(
                num_hidden_layers=4,
                gqa_layers=[0],
                layer_types=["linear_attention"] * 3 + ["full_attention"],
            )

    def test_routes_to_kda_hybrid_stack(self):
        """Transformers has no solar_open2; without the registry entry the
        checkpoint fails to load, and without kimi_linear_config() the KDA
        layers get no linear-attention backend or state pool."""
        self.assertIs(_CONFIG_REGISTRY["solar_open2"], SolarOpen2Config)
        config = SolarOpen2Config(**_RELEASED_LAYOUT)
        model_config = SimpleNamespace(hf_config=config)
        self.assertIs(kimi_linear_config(model_config), config)


class _RecordingParam:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        # Stacked loaders take shard_id positionally; expert loaders pass the
        # param name positionally and shard_id by keyword.
        shard_id = kwargs["shard_id"] if "shard_id" in kwargs else (args or [None])[0]
        self.log.append((self.name, shard_id, kwargs.get("expert_id")))


class TestSolarOpen2WeightRouting(unittest.TestCase):
    """q/k/v projections exist in both layer types under the same checkpoint
    names; they must land in different fused params."""

    _PARAMS = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.g_proj.weight",
        "model.layers.1.self_attn.fused_qkvbfg_a_proj.weight",
        "model.layers.1.self_attn.fused_fg_b_proj.weight",
        "model.layers.1.self_attn.qkv_conv1d.weight",
        "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.1.mlp.experts.w13_weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
    ]

    def _load(self, names):
        log = []
        model = object.__new__(SolarOpen2ForCausalLM)
        nn_init = torch.nn.Module.__init__
        nn_init(model)
        model.config = SolarOpen2Config(
            num_hidden_layers=4, gqa_layers=[0], n_routed_experts=4
        )
        model.pp_group = SimpleNamespace(world_size=1)
        kda_layer = SimpleNamespace(self_attn=SimpleNamespace(do_fuse_qkvbfg=True))
        model.model = SimpleNamespace(layers={1: kda_layer})
        params = {name: _RecordingParam(name, log) for name in self._PARAMS}
        model.named_parameters = lambda: iter(params.items())
        model.load_weights([(name, torch.empty(0)) for name in names])
        return log

    def test_softmax_layer_qkv_stacks_into_qkv_proj(self):
        log = self._load(
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.g_proj.weight",
            ]
        )
        self.assertEqual(
            log,
            [
                ("model.layers.0.self_attn.qkv_proj.weight", "q", None),
                ("model.layers.0.self_attn.qkv_proj.weight", "v", None),
                ("model.layers.0.self_attn.g_proj.weight", None, None),
            ],
        )

    def test_kda_layer_projections_fuse_with_beta_and_gates(self):
        log = self._load(
            [
                "model.layers.1.self_attn.q_proj.weight",
                "model.layers.1.self_attn.b_proj.weight",
                "model.layers.1.self_attn.g_a_proj.weight",
                "model.layers.1.self_attn.g_b_proj.weight",
                "model.layers.1.self_attn.k_conv1d.weight",
            ]
        )
        self.assertEqual(
            log,
            [
                ("model.layers.1.self_attn.fused_qkvbfg_a_proj.weight", 0, None),
                ("model.layers.1.self_attn.fused_qkvbfg_a_proj.weight", 3, None),
                ("model.layers.1.self_attn.fused_qkvbfg_a_proj.weight", 5, None),
                ("model.layers.1.self_attn.fused_fg_b_proj.weight", 1, None),
                ("model.layers.1.self_attn.qkv_conv1d.weight", 1, None),
            ],
        )

    def test_routed_and_shared_experts_use_separate_paths(self):
        """Routed experts share the gate_proj/up_proj names with the shared expert."""
        log = self._load(
            [
                "model.layers.1.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.shared_experts.up_proj.weight",
                "model.layers.1.mlp.gate.e_score_correction_bias",
            ]
        )
        self.assertEqual(
            log,
            [
                ("model.layers.1.mlp.experts.w13_weight", "w3", 2),
                ("model.layers.1.mlp.shared_experts.gate_up_proj.weight", 1, None),
                ("model.layers.1.mlp.gate.e_score_correction_bias", None, None),
            ],
        )


class TestKdaQuantConfigForFusion(unittest.TestCase):
    """KDA projections fuse into one bf16 GEMM only when the checkpoint keeps
    every one of them unquantized; fusing a quantized projection would load
    packed weights into a bf16 parameter."""

    class _Config:
        def __init__(self, quantized_names):
            self.quantized_names = quantized_names

        def get_quant_method(self, layer, prefix):
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

            if prefix.rsplit(".", 1)[-1] in self.quantized_names:
                return object()
            return UnquantizedLinearMethod()

    def test_all_unquantized_drops_quant_config(self):
        from sglang.srt.models.solar_open2 import _kda_quant_config

        self.assertIsNone(
            _kda_quant_config(self._Config(()), "model.layers.1.self_attn")
        )

    def test_any_quantized_projection_keeps_quant_config(self):
        from sglang.srt.models.solar_open2 import _kda_quant_config

        config = self._Config(("g_b_proj",))
        self.assertIs(_kda_quant_config(config, "model.layers.1.self_attn"), config)


class TestKdaAllowNegEigvalGuard(unittest.TestCase):
    """beta = 2 * sigmoid(b) must never reach a KDA kernel that silently drops
    the 2x (in-kernel sigmoid) or is unvalidated for beta > 1 (non-Triton
    prefill); such kernels must raise instead of returning wrong outputs."""

    def _dispatcher(self, extend_kernel):
        from sglang.srt.layers.attention.linear.kda_backend import (
            KDAKernelDispatcher,
        )

        dispatcher = object.__new__(KDAKernelDispatcher)
        dispatcher.extend_kernel = extend_kernel
        dispatcher.triton_kernel = extend_kernel
        return dispatcher

    def _extend(self, dispatcher, allow_neg_eigval):
        x = torch.empty(0)
        return dispatcher.extend(
            x,
            x,
            x,
            x,
            x,
            ssm_states=x,
            cache_indices=x,
            query_start_loc=x,
            allow_neg_eigval=allow_neg_eigval,
        )

    def test_prefill_kernel_without_support_raises(self):
        from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
            LinearAttnKernelBase,
        )

        class _UnsupportedPrefill(LinearAttnKernelBase):
            decode = extend = lambda *args, **kwargs: "wrong output"

        kernel = _UnsupportedPrefill()
        with self.assertRaises(NotImplementedError):
            self._extend(self._dispatcher(kernel), allow_neg_eigval=True)

    def test_triton_prefill_receives_call_without_the_flag(self):
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        calls = []
        kernel = object.__new__(TritonKDAKernel)
        kernel.extend = lambda *args, **kwargs: calls.append(kwargs) or "out"
        self.assertEqual(self._extend(self._dispatcher(kernel), True), "out")
        self.assertNotIn("allow_neg_eigval", calls[0])


if __name__ == "__main__":
    unittest.main()
