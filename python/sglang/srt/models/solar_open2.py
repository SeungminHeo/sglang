# Adapted from: https://github.com/UpstageAI/vllm/blob/v0.22.0-solar-open2/vllm/model_executor/models/solar_open2.py
"""Inference-only Solar Open 2 (upstage/Solar-Open2-250B).

Every fourth layer is softmax GQA (NoPE, sigmoid output gate); the rest are
Kimi Delta Attention with beta = 2 * sigmoid(b). Every layer has an MoE FFN.
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.solar_open2 import SolarOpen2Config
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.deepseek_common.utils import tiny_router_gemm_max_tokens
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, is_cuda, make_pp_layers
from sglang.srt.utils.common import BumpAllocator

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16


_KDA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "o_proj",
)


def _kda_quant_config(
    quant_config: Optional[QuantizationConfig], prefix: str
) -> Optional[QuantizationConfig]:
    """Drop the quant config for a KDA layer whose projections the checkpoint
    leaves unquantized, so KimiDeltaAttention fuses them into one GEMM."""
    if quant_config is None:
        return None
    for name in _KDA_PROJECTIONS:
        # LinearBase resolves the quant method without allocating weights.
        probe = LinearBase(1, 1, quant_config=quant_config, prefix=f"{prefix}.{name}")
        if not isinstance(probe.quant_method, UnquantizedLinearMethod):
            return quant_config
    return None


class SolarOpen2MoE(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        # The reference router upcasts bf16 activations and bf16 checkpoint
        # weights to fp32; a bf16 GEMM with fp32 accumulation and output is the
        # same product up to summation order.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.tiny_router_gemm_max_tokens = tiny_router_gemm_max_tokens(
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            weight_dtype=self.gate.weight.dtype,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            # Biased-sigmoid grouped top-k, as in DeepSeek-V3 (used by in-kernel routing).
            routing_method_type=RoutingMethodType.DeepSeekV3,
            activation=config.hidden_act,
            prefix=add_prefix("experts", prefix),
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )

        if config.n_shared_experts:
            self.shared_experts = LlamaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        shared_output = None
        if self.shared_experts is not None and num_tokens > 0:
            shared_output = self.shared_experts(hidden_states)

        router_logits = self._router_logits(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states.view(num_tokens, hidden_size)

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.gate.weight
        if not _is_cuda or weight.dtype != torch.bfloat16:
            return F.linear(hidden_states.float(), weight.float())
        if hidden_states.shape[0] <= self.tiny_router_gemm_max_tokens:
            return tiny_gemm_bf16(
                hidden_states,
                weight,
                out_dtype=torch.float32,
                max_m=self.tiny_router_gemm_max_tokens,
            )
        return linear_bf16_fp32(hidden_states, weight)


class SolarOpen2Attention(nn.Module):
    """Softmax GQA without positional encoding and with an elementwise sigmoid
    output gate computed from the layer input."""

    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.use_rope or config.use_qk_norm:
            raise NotImplementedError(
                "Solar Open 2 GQA with use_rope or use_qk_norm is not supported; "
                "the released checkpoints use neither."
            )
        tp_size = get_parallel().tp_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        if config.use_gqa_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=config.use_gqa_gate_bias,
                quant_config=quant_config,
                prefix=add_prefix("g_proj", prefix),
            )
        else:
            self.g_proj = None
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, forward_batch)
        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class SolarOpen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.is_kda_layer(layer_id):
            if config.kda_use_full_proj:
                raise NotImplementedError(
                    "Solar Open 2 kda_use_full_proj=True (full-rank f/g "
                    "projections) is not supported yet."
                )
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=_kda_quant_config(
                    quant_config, add_prefix("self_attn", prefix)
                ),
                rms_norm_eps=config.rms_norm_eps,
                prefix=add_prefix("self_attn", prefix),
                reduce_results=False,
                allow_neg_eigval=config.kda_allow_neg_eigval,
            )
        else:
            self.self_attn = SolarOpen2Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )

        attn, ffn = self.stage_facts(config, layer_id)
        if ffn.sparse:
            self.mlp = SolarOpen2MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = LlamaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: SolarOpen2Config, layer_id: int):
        """The stages the layer at ``layer_id`` declares, from the config: the
        model's shared declaration function (see make_layers)."""
        return (
            declare_attn(),
            declare_ffn(
                sparse=layer_id >= config.first_k_dense_replace,
                next_layer_sparse=layer_id + 1 < config.num_hidden_layers
                and layer_id + 1 >= config.first_k_dense_replace,
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class SolarOpen2Model(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.dspark_layers_to_capture: Optional[list[int]] = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarOpen2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: SolarOpen2DecoderLayer.stage_facts(config, idx),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        # KDA layers draw no scratch from it; the shared layer signature requires one.
        zero_allocator = BumpAllocator(
            buffer_size=(self.end_layer - self.start_layer) * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )
            if (
                self.dspark_layers_to_capture is not None
                and i in self.dspark_layers_to_capture
            ):
                # The residual stream after layer i (pre-norm for the last layer).
                aux_hidden_states.append(
                    residual_batch.snapshot(hidden_states, forward_batch)
                )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )
        if self.dspark_layers_to_capture is not None:
            return hidden_states, aux_hidden_states
        return hidden_states


class SolarOpen2ForCausalLM(nn.Module):
    # Lets quantization ignore lists, written with checkpoint names, match the
    # fused params. Quantized KDA layers stay on the unfused qkv_proj path.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: SolarOpen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        if quant_config is not None:
            # Some configs (auto-round) never read the loader's mapping.
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = SolarOpen2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config=config)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        # DSpark target_layer_ids are raw decoder indices, captured after the layer runs.
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def _is_non_local_pp_weight(self, name: str) -> bool:
        if self.pp_group.world_size == 1:
            return False
        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not (self.start_layer <= layer_id < self.end_layer)
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith(("model.norm.", "lm_head.")):
            return not self.pp_group.is_last_rank
        return False

    def _stacked_target(
        self, name: str, param_name: str, weight_name: str
    ) -> Optional[str]:
        """Rename a checkpoint shard to its stacked param, or None if this
        mapping does not apply to the shard's layer.

        q/k/v exist in both layer types: KDA layers fuse them with beta and the
        f/g low-rank inputs, softmax layers stack them into ``qkv_proj``.
        """
        layer_id = get_layer_id(name)
        if param_name in {".fused_qkvbfg_a_proj", ".fused_fg_b_proj"}:
            if layer_id is None or not self.config.is_kda_layer(layer_id):
                return None
            if not self.model.layers[layer_id].self_attn.do_fuse_qkvbfg:
                return None
        return name.replace(weight_name, param_name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # KDA fused path (KimiDeltaAttention.do_fuse_qkvbfg)
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            # Softmax layers, and KDA layers on the unfused (quantized) path
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if self._is_non_local_pp_weight(name):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts carry gate_proj/up_proj too; expert_params_mapping owns them.
                if "mlp.experts." in name:
                    continue
                target = self._stacked_target(name, param_name, weight_name)
                if target is None:
                    continue
                if target.endswith(".bias") and target not in params_dict:
                    continue
                param = params_dict[target]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)


EntryClass = SolarOpen2ForCausalLM
