# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only SKT A.X-K2 (AXK2ForCausalLM).

A.X-K2 is a DeepSeek-V3.2 derivative (MLA + DeepSeek MoE + DSA sparse-attention
indexer, block-FP8 checkpoint) with three deviations:

1. Sparse Gated Attention output gate: the checkpoint's ``q_b_proj`` is a fused
   projection ``Linear(2 * q_lora_rank, num_heads * (qk_head_dim + v_head_dim))``
   consuming ``cat([q_a_layernorm(q_c), q_c])``. Per head it yields the query
   (``qk_head_dim``) and a gate (``v_head_dim``); the attention output is
   multiplied by ``sigmoid(gate)`` right before ``o_proj``.
2. Gated RMSNorm: ``input_layernorm`` on every layer and
   ``post_attention_layernorm`` on MoE layers wrap RMSNorm with a low-rank
   input-dependent gate ``y * sigmoid(W_up(silu(W_down(y))))``.
3. The DSA indexer consumes the *pre-norm* ``q_c`` (DeepSeek-V3.2 feeds it the
   normed q-LoRA).

This file is self-contained: ``AXK2ForCausalLM`` lets the unmodified
DeepSeek-V2 classes build the model, then converts each decoder layer in place
by swapping four leaf modules. The ``q_a_layernorm`` swap makes every q-LoRA
consumer downstream see ``cat([normed, raw])``, the fused ``q_b_proj`` swap
splits that into query + stashed gate, the ``o_proj`` swap consumes the stashed
gate, and the indexer's ``wq_b`` swap slices the raw half back out — so the
shared DeepSeek forward mixins run unchanged. CUDA-only for now: the
ROCm/NPU/CPU fused paths read ``q_a_layernorm.weight`` directly and would
bypass the concat.
"""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.layernorm.lowrank_gated_rmsnorm import (
    LOWRANK_GATED_RMSNORM_MAX_TOKENS,
    lowrank_gated_rmsnorm,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.models.deepseek_common.utils import _is_hip, _is_npu, _is_xpu
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class AXK2GatedRMSNorm(nn.Module):
    """RMSNorm followed by a low-rank input-dependent sigmoid gate.

    y = RMSNorm(x); return y * sigmoid(W_up(silu(W_down(y))))

    Checkpoint layout: ``<prefix>.norm.weight`` (remapped to ``base_norm`` at
    load time so the PP loader's ``.norm.`` final-norm skip does not swallow
    it), ``<prefix>.W_down.weight`` (hidden -> rank) and ``<prefix>.W_up.weight``
    (rank -> hidden), both kept in bf16 (``modules_to_not_convert``).

    Mirrors the RMSNorm call contract used by ``LayerCommunicator``:
    ``forward(x)``, ``forward(x, residual[, post_residual_addition])`` with
    fused residual-add, and ``forward_with_allreduce_fusion`` (all-reduce +
    residual + RMSNorm in one kernel, gate applied afterwards). It deliberately
    does NOT expose ``weight`` / ``variance_epsilon`` so any fused-norm fast
    path that would silently skip the gate fails loudly instead.
    """

    def __init__(
        self,
        hidden_size: int,
        gate_rank: int,
        eps: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.base_norm = RMSNorm(hidden_size, eps=eps)
        self.W_down = ReplicatedLinear(
            hidden_size,
            gate_rank,
            bias=False,
            quant_config=None,
            prefix=add_prefix("W_down", prefix),
        )
        self.W_up = ReplicatedLinear(
            gate_rank,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("W_up", prefix),
        )

    def _apply_gate(self, y: torch.Tensor) -> torch.Tensor:
        if y.numel() == 0:
            return y
        gate, _ = self.W_down(y)
        gate, _ = self.W_up(F.silu(gate))
        if y.is_cuda:
            # y * sigmoid(gate) in fp32, stored in y.dtype: one launch instead
            # of upcast-copy + sigmoid + mul + downcast-copy.
            return fused_sigmoid_mul(y, gate, inplace=True)
        return (y * torch.sigmoid(gate.float())).to(y.dtype)

    def _fused_small_batch(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and x.dim() == 2
            and 0 < x.shape[0] <= LOWRANK_GATED_RMSNORM_MAX_TOKENS
            and x.stride(1) == 1
        )

    def _forward_fused(self, x: torch.Tensor, residual: Optional[torch.Tensor]):
        # One launch: (x + residual) -> rmsnorm -> low-rank gate -> y * sigmoid.
        # Same in-place contract as the unfused path: with a residual the
        # output overwrites x and residual receives x + residual; without one
        # x is left untouched (the caller keeps it as the residual).
        return lowrank_gated_rmsnorm(
            x,
            self.base_norm.weight,
            self.W_down.weight,
            self.W_up.weight,
            self.base_norm.variance_epsilon,
            residual=residual,
            out=x if residual is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if residual is None:
            assert post_residual_addition is None
            if self._fused_small_batch(x):
                return self._forward_fused(x, None)[0]
            return self._apply_gate(self.base_norm(x))
        if post_residual_addition is None and self._fused_small_batch(x):
            return self._forward_fused(x, residual)
        y, residual = self.base_norm(x, residual, post_residual_addition)
        return self._apply_gate(y), residual

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ):
        """All-reduce + residual + RMSNorm fused (flashinfer/aiter), then the gate."""
        out = self.base_norm.forward_with_allreduce_fusion(
            x, residual, post_residual_addition, use_attn_tp_group
        )
        if residual is None:
            return self._apply_gate(out)
        y, residual = out
        return self._apply_gate(y), residual


class _AXK2ConcatQANorm(RMSNorm):
    """q_a_layernorm that returns ``cat([RMSNorm(q_c), q_c], -1)``.

    The fused q/gate projection consumes both the normed and the raw q-LoRA
    bottleneck, and the DSA indexer consumes only the raw half; widening this
    module's output lets both flow through the unmodified DeepSeek forward
    mixins (which treat the q-LoRA activation as an opaque tensor between
    ``q_a_layernorm`` and ``q_b_proj`` / the indexer).
    """

    def _cat(self, normed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([normed, x], dim=-1)

    def _check(self, residual, post_residual_addition, quant_linear) -> None:
        # q_a_layernorm is always called bare; a residual or a fused-quant
        # linear would mean the concat contract no longer holds.
        assert residual is None and post_residual_addition is None
        assert quant_linear is None, (
            "A.X-K2 q_a_layernorm cannot fuse a quantized linear: the fused "
            "q/gate projection consumes the [normed | raw] concat, not the "
            "normed activation alone."
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual=None,
        post_residual_addition=None,
        quant_linear=None,
    ):
        self._check(residual, post_residual_addition, quant_linear)
        return self._cat(super().forward_cuda(x), x)

    def forward_native(
        self,
        x: torch.Tensor,
        residual=None,
        post_residual_addition=None,
        quant_linear=None,
    ):
        self._check(residual, post_residual_addition, quant_linear)
        return self._cat(super().forward_native(x), x)

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual=None,
        post_residual_addition=None,
        quant_linear=None,
    ):
        self._check(residual, post_residual_addition, quant_linear)
        return self._cat(super().forward_cpu(x), x)


class _AXK2FusedQGateProj(ColumnParallelLinear):
    """Fused query + output-gate projection replacing ``q_b_proj``.

    Consumes the ``2 * q_lora_rank`` concat produced by ``_AXK2ConcatQANorm``.
    Returns the query as ``(tokens, local_heads, qk_head_dim)`` — a same-shape
    no-op for the callers' ``.view(-1, local_heads, qk_head_dim)`` — and
    stashes the gate half until ``o_proj`` consumes it via ``take_gate``.
    """

    def __init__(
        self,
        q_lora_rank: int,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        quant_config,
        prefix: str,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__(
            2 * q_lora_rank,
            num_heads * (qk_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self._pending_gate: Optional[torch.Tensor] = None

    def forward(self, input_: torch.Tensor):
        fused, bias = super().forward(input_)
        head_width = self.qk_head_dim + self.v_head_dim
        fused = fused.view(-1, fused.shape[-1] // head_width, head_width)
        assert self._pending_gate is None, (
            "attention output gate was produced but never consumed by o_proj"
        )
        self._pending_gate = fused[..., self.qk_head_dim :]
        # Contiguous so downstream code sees the same layout as a plain
        # q_b_proj output (attention backends flatten q with .view()).
        return fused[..., : self.qk_head_dim].contiguous(), bias

    def take_gate(self) -> torch.Tensor:
        gate = self._pending_gate
        assert gate is not None, "o_proj ran without a stashed attention output gate"
        self._pending_gate = None
        return gate


class _AXK2GatedOProj(RowParallelLinear):
    """o_proj that applies the stashed sigmoid output gate to its input.

    ``_take_gate`` is bound to the fused q/gate projection's ``take_gate``
    after construction (a bound method, so no duplicate module registration).
    """

    def forward(
        self,
        input_,
        skip_all_reduce=False,
        forward_batch=None,
        output_tensor=None,
    ):
        gate = self._take_gate()
        assert torch.is_tensor(input_) and input_.dim() == 2, (
            "A.X-K2 gated o_proj expects a 2D bf16 attention output; "
            f"got {type(input_)}"
        )
        if input_.is_cuda:
            input_ = fused_sigmoid_mul(input_, gate)
        else:
            input_ = input_ * torch.sigmoid(gate.reshape(input_.shape).float()).to(
                input_.dtype
            )
        return super().forward(
            input_,
            skip_all_reduce=skip_all_reduce,
            forward_batch=forward_batch,
            output_tensor=output_tensor,
        )


class _AXK2SlicedIndexerWqB(ReplicatedLinear):
    """Indexer ``wq_b`` that consumes the raw (pre-norm) half of the q-LoRA concat.

    A.X-K2 feeds the indexer ``q_c = q_a_proj(x)`` where DeepSeek-V3.2 feeds it
    ``q_a_layernorm(q_c)``; with the concat layout ``[normed | raw]`` the raw
    half is the trailing ``input_size`` slice.
    """

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == 2 * self.input_size, (
            f"indexer wq_b expected the {2 * self.input_size}-wide q-LoRA "
            f"concat, got width {x.shape[-1]}"
        )
        return super().forward(x[..., self.input_size :].contiguous())


def _convert_attention_to_axk2(
    attn: DeepseekV2AttentionMLA,
    config,
    quant_config,
    prefix: str,
) -> None:
    """Swap the four leaf modules that differ from DeepSeek-V3.2.

    Runs at init time on the attention instance the DeepSeek classes built;
    TP settings and result reduction are read off the modules being replaced
    so the conversion stays faithful to however the layer was constructed.
    """
    assert attn.use_dsa and attn.q_lora_rank is not None

    attn.q_a_layernorm = _AXK2ConcatQANorm(attn.q_lora_rank, eps=config.rms_norm_eps)

    old_q_b_proj = attn.q_b_proj
    attn.q_b_proj = _AXK2FusedQGateProj(
        q_lora_rank=attn.q_lora_rank,
        num_heads=attn.num_heads,
        qk_head_dim=attn.qk_head_dim,
        v_head_dim=attn.v_head_dim,
        quant_config=attn._get_q_b_proj_quant_config(quant_config),
        prefix=add_prefix("q_b_proj", prefix),
        tp_rank=old_q_b_proj.tp_rank,
        tp_size=old_q_b_proj.tp_size,
    )
    # Init-derived fast-path flags computed from the replaced q_b_proj.
    attn._q_b_proj_verified_shape = False
    attn._use_min_latency_q_b_gemm = False

    old_o_proj = attn.o_proj
    attn.o_proj = _AXK2GatedOProj(
        attn.num_heads * attn.v_head_dim,
        attn.hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=old_o_proj.reduce_results,
        prefix=add_prefix("o_proj", prefix),
        tp_rank=old_o_proj.tp_rank,
        tp_size=old_o_proj.tp_size,
    )
    attn.o_proj._take_gate = attn.q_b_proj.take_gate

    attn.indexer.wq_b = _AXK2SlicedIndexerWqB(
        attn.q_lora_rank,
        attn.indexer.n_heads * attn.indexer.head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("wq_b", add_prefix("indexer", prefix)),
    )


def _rebind_communicator_norm(communicator, attr: str, norm: nn.Module) -> None:
    """Point a LayerCommunicator (and its LayerNorm-SP sibling) at a new norm.

    ``LayerCommunicator`` captures the norm modules by reference at
    construction time, and under LayerNorm SP it builds an all-SCATTERED
    ``_sp_variant`` sibling holding the *same* references. Both have to be
    repointed or the SP region would silently run the ungated RMSNorm.
    """
    while communicator is not None:
        assert hasattr(communicator, attr), (
            f"LayerCommunicator no longer holds {attr!r}; the A.X-K2 gated-norm "
            "swap must be re-checked against the new communicator wiring."
        )
        setattr(communicator, attr, norm)
        communicator = getattr(communicator, "_sp_variant", None)


def _convert_decoder_layer_to_axk2(
    layer: DeepseekV2DecoderLayer,
    config,
    quant_config,
    prefix: str,
) -> None:
    _convert_attention_to_axk2(
        layer.self_attn,
        config=config,
        quant_config=quant_config,
        prefix=add_prefix("self_attn", prefix),
    )
    # input_layernorm is gated on every layer; post_attention_layernorm only
    # on MoE layers. LayerCommunicator holds references to the norms, so
    # refresh those too.
    layer.input_layernorm = AXK2GatedRMSNorm(
        config.hidden_size,
        gate_rank=config.gated_norm_rank,
        eps=config.rms_norm_eps,
        prefix=add_prefix("input_layernorm", prefix),
    )
    _rebind_communicator_norm(
        layer.layer_communicator, "input_layernorm", layer.input_layernorm
    )
    if layer.is_layer_sparse:
        layer.post_attention_layernorm = AXK2GatedRMSNorm(
            config.hidden_size,
            gate_rank=config.gated_norm_rank,
            eps=config.rms_norm_eps,
            prefix=add_prefix("post_attention_layernorm", prefix),
        )
        _rebind_communicator_norm(
            layer.layer_communicator,
            "post_attention_layernorm",
            layer.post_attention_layernorm,
        )


class AXK2ForCausalLM(DeepseekV2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        assert config.attention_output_gate and config.gated_norm, (
            "AXK2ForCausalLM requires attention_output_gate=true and "
            "gated_norm=true; variants without them are not supported."
        )
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        if _is_hip or _is_npu or _is_xpu:
            raise NotImplementedError(
                "A.X-K2 is CUDA-only for now: the ROCm/NPU/XPU fused RMSNorm "
                "paths bypass the concatenated q_a_layernorm this port relies on."
            )
        if get_parallel().attn_cp_size > 1:
            raise NotImplementedError(
                "Attention context parallelism is not supported for A.X-K2 yet."
            )
        if get_parallel().dcp_enabled:
            raise NotImplementedError(
                "DCP is not supported for A.X-K2: the attention output gate is "
                "per-local-head and DCP gathers heads across ranks before o_proj."
            )
        model_prefix = add_prefix("model", prefix)
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            _convert_decoder_layer_to_axk2(
                self.model.layers[layer_id],
                config=config,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{layer_id}", model_prefix),
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        return super().load_weights(self._remap_axk2_names(weights), is_nextn)

    @staticmethod
    def _remap_axk2_names(
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Remap checkpoint names to this module tree.

        The official checkpoint stores the fused q/gate projection as
        ``q_b_proj`` and the gated norms as ``{norm, W_up, W_down}``; a
        checkpoint re-saved through transformers uses ``q_gate_proj`` and
        ``mlp.{fc1, fc2}`` instead. The gated norms' inner RMSNorm lives at
        ``base_norm`` here because the DeepSeek PP loader skips any ``.norm.``
        name on non-last ranks (a final-norm check).
        """
        for name, loaded_weight in weights:
            name = name.replace(".self_attn.q_gate_proj.", ".self_attn.q_b_proj.")
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                name = name.replace(f".{norm_name}.norm.", f".{norm_name}.base_norm.")
                name = name.replace(f".{norm_name}.mlp.fc1.", f".{norm_name}.W_down.")
                name = name.replace(f".{norm_name}.mlp.fc2.", f".{norm_name}.W_up.")
            yield name, loaded_weight


EntryClass = [AXK2ForCausalLM]
