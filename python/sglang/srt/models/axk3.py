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
"""Inference-only SKT A.X-K3 family (AXK3ForCausalLM) on the A.X-K2 port.

A.X-K3 is A.X-K2 with two additions, both switched on by config flags of an
``axk2`` checkpoint (see ``configs/axk3.py``, which routes it here):

1. Index share (``dsa_index_share``): only the ``full_attention`` layers run
   the DSA indexer; each ``sliding_attention`` layer reuses the top-k of the
   full layer before it. The config resolution turns ``layer_types`` into
   ``index_topk_pattern``, which the DeepSeek DSA attention already serves.
2. Gated residual (``gated_residual``, A.X-K3 Flash): the residual stream is
   widened to ``n`` copies of the hidden state. Each sublayer reads its input
   as a gated mix of the per-branch-normalized streams and writes its output
   back with a per-branch gate; this replaces ``input_layernorm`` /
   ``post_attention_layernorm``, and a write-less read replaces the final norm.
   It is the gated-injection hyper-connection SGLang already runs for
   Qwen4-Exp (``GatedResidual`` / ``GatedResidualState``), with the checkpoint
   storing the per-branch RMSNorm gain as ``gamma`` instead of ``1 + weight``.
"""

import re
from functools import partial
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.layer_boundary import GatedResidualState
from sglang.srt.layers.layer_boundary.factories import (
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.models.axk2 import (
    AXK2ForCausalLM,
    _build_axk2_stages,
    _convert_attention_to_axk2,
    _convert_decoder_layer_to_axk2,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

_GATED_RESIDUAL_TENSOR = re.compile(
    r"^(.*gated_residual)\.(gamma|write_bias|read_down\.weight|read_up\.weight"
    r"|write_proj\.weight)$"
)
# Checkpoint projection -> GatedResidual parameter.
_GATED_RESIDUAL_LINEARS = {
    "read_down.weight": "input_mix_weight_down.weight",
    "read_up.weight": "input_mix_weight_up.weight",
    "write_proj.weight": "block_inject_weight.weight",
}


def _hc_config(config) -> HyperConnectionConfig:
    return HyperConnectionConfig(
        hc_count=config.gated_residual_num_streams,
        hidden_size=config.hidden_size,
        params_dtype=torch.get_default_dtype(),
        hc_lowrank=config.gated_residual_rank,
        rms_norm_eps=config.rms_norm_eps,
        hc_per_branch_norm=True,
    )


def _widen_streams(
    hidden_states: torch.Tensor, *, num_streams: int, hidden_size: int
) -> torch.Tensor:
    """The layer stack's input as ``num_streams`` stream-major copies; streams
    that are already widened pass through."""
    if hidden_states.shape[-1] == num_streams * hidden_size:
        return hidden_states
    assert hidden_states.shape[-1] == hidden_size
    return hidden_states.repeat(1, num_streams)


def _axk3_stage_facts(config, layer_id: int, *, residual=None):
    """The stages an A.X-K3 layer declares, from the config alone (the
    model's shared declaration function, see make_layers). ``residual`` is a
    gated-residual layer's own reads and updates of the streams."""
    if not config.gated_residual:
        return DeepseekV2DecoderLayer.stage_facts(config, layer_id)
    if residual is None:
        residual = GatedResidualState.facts()
    is_sparse = DeepseekV2DecoderLayer._is_layer_sparse
    return (
        declare_attn(read=residual.attn_readout, update=residual.attn_update),
        declare_ffn(
            sparse=is_sparse(config, layer_id, is_nextn=False),
            next_layer_sparse=is_sparse(config, layer_id + 1, is_nextn=False),
            read=residual.ffn_readout,
            update=residual.ffn_update,
        ),
    )


def _convert_decoder_layer_to_axk3(
    layer: DeepseekV2DecoderLayer,
    config,
    quant_config,
    prefix: str,
) -> None:
    if not config.gated_residual:
        _convert_decoder_layer_to_axk2(
            layer, config=config, quant_config=quant_config, prefix=prefix
        )
        return
    _convert_attention_to_axk2(
        layer.self_attn,
        config=config,
        quant_config=quant_config,
        prefix=add_prefix("self_attn", prefix),
    )
    # The gated residual reads normalize the streams themselves.
    del layer.input_layernorm
    del layer.post_attention_layernorm
    hc_config = _hc_config(config)
    layer.attn_gated_residual = GatedResidual(hc_config)
    layer.mlp_gated_residual = GatedResidual(hc_config)


def _build_axk3_stages(layer: DeepseekV2DecoderLayer):
    if not layer.config.gated_residual:
        return _build_axk2_stages(layer)
    attn_gr, mlp_gr = layer.attn_gated_residual, layer.mlp_gated_residual
    residual = GatedResidualState(
        expand=partial(
            _widen_streams,
            num_streams=attn_gr.hc_count,
            hidden_size=attn_gr.hidden_size,
        ),
        attn_mix=attn_gr.mix,
        ffn_mix=mlp_gr.mix,
        attn_combine=attn_gr.combine,
        ffn_combine=mlp_gr.combine,
    ).residual_ops()
    attn, ffn = _axk3_stage_facts(layer.config, layer.layer_id, residual=residual)
    # Each read normalizes the streams itself, so neither stage binds a norm.
    return append_stages(
        (attn, None, {"qkv_latent_func": layer.self_attn.prepare_qkv_latent}),
        (ffn, None),
    )


class AXK3FinalGatedRead(nn.Module):
    """The write-less gated read that contracts the streams for the LM head,
    in the final norm's place (``gated_residual_final: gated_read``)."""

    def __init__(self, hc_config: HyperConnectionConfig) -> None:
        super().__init__()
        self.gated_residual = GatedResidual(hc_config, use_combine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gated_residual.mix(hidden_states)[0]


def _remap_gated_residual_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Map the checkpoint's gated-residual tensors onto ``GatedResidual``.

    ``gamma`` is the per-branch RMSNorm gain; ``GroupedGemmaRMSNorm`` scales by
    ``1 + weight``, so it loads ``gamma - 1`` (exact in bf16 for the
    checkpoint's gains). ``write_bias`` is a per-branch write-gate offset
    ``GatedResidual`` does not have; an all-zero one is dropped. The final
    read lives at ``model.norm``, so pipeline ranks without the final norm
    skip it like one.
    """
    for name, loaded_weight in weights:
        name = name.replace("model.final_gated_residual.", "model.norm.gated_residual.")
        match = _GATED_RESIDUAL_TENSOR.match(name)
        if match is None:
            yield name, loaded_weight
            continue
        module, leaf = match.groups()
        if leaf == "gamma":
            gain = (loaded_weight.float() - 1.0).to(loaded_weight.dtype)
            yield f"{module}.hc_norm.weight", gain
        elif leaf == "write_bias":
            if torch.count_nonzero(loaded_weight):
                raise NotImplementedError(
                    f"{name} is nonzero; a gated-residual write bias is not supported."
                )
        else:
            yield f"{module}.{_GATED_RESIDUAL_LINEARS[leaf]}", loaded_weight


class AXK3ForCausalLM(AXK2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        if config.gated_residual:
            if get_parallel().pp_size > 1:
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for the A.X-K3 "
                    "gated residual yet."
                )
            self.model.norm = AXK3FinalGatedRead(_hc_config(config))

    @staticmethod
    def _check_config(config) -> None:
        assert config.attention_output_gate, (
            "AXK3ForCausalLM requires attention_output_gate=true."
        )
        assert config.gated_residual or config.gated_norm, (
            "AXK3ForCausalLM without gated_residual requires gated_norm=true."
        )

    _convert_decoder_layer = staticmethod(_convert_decoder_layer_to_axk3)
    _build_layer_stages = staticmethod(_build_axk3_stages)
    _layer_stage_facts = staticmethod(_axk3_stage_facts)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if self.config.gated_residual:
            weights = _remap_gated_residual_weights(weights)
        return super().load_weights(weights, is_nextn)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list] = None):
        self._reject_aux_hidden_capture()
        super().set_eagle3_layers_to_capture(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids: list):
        self._reject_aux_hidden_capture()
        super().set_dflash_layers_to_capture(layer_ids)

    def _reject_aux_hidden_capture(self) -> None:
        if self.config.gated_residual:
            # The layers' outputs are the widened streams, not hidden states.
            raise NotImplementedError(
                "A.X-K3 gated residual does not expose auxiliary hidden states "
                "(EAGLE3 / DFlash / DSpark drafts) yet."
            )


EntryClass = [AXK3ForCausalLM]
