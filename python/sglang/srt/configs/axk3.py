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
"""Config resolution for SKT A.X-K3 checkpoints.

A.X-K3 checkpoints ship as ``model_type: axk2`` / ``AXK2ForCausalLM`` and turn
on their additions with config flags: ``dsa_index_share`` (layers typed
``sliding_attention`` reuse the top-k of the preceding full layer's indexer)
and ``gated_residual`` (a widened residual stream replaces the pre-norms).
"""

AXK3_ARCH = "AXK3ForCausalLM"
_SLIDING_ATTENTION = "sliding_attention"


def is_axk3_config(config) -> bool:
    return bool(
        getattr(config, "dsa_index_share", False)
        or getattr(config, "gated_residual", False)
    )


def resolve_axk3_config(config) -> None:
    """Derive ``index_topk_pattern`` and route the config to AXK3ForCausalLM.

    The pattern is what the DSA attention, the KV pool and HiSparse read to
    tell full ("F") from top-k-sharing ("S") layers, so it has to be set at
    config load, before any of them is built.
    """
    if not is_axk3_config(config):
        return
    _reject_unsupported(config)
    layer_types = getattr(config, "layer_types", None)
    if config.dsa_index_share and layer_types is not None:
        if layer_types[0] == _SLIDING_ATTENTION:
            raise ValueError(
                "A.X-K3 layer 0 is a sliding_attention layer, which has no "
                "preceding indexer to share the top-k of."
            )
        config.index_topk_pattern = "".join(
            "S" if layer_type == _SLIDING_ATTENTION else "F"
            for layer_type in layer_types
        )
    config.architectures = [AXK3_ARCH]


def _reject_unsupported(config) -> None:
    if getattr(config, "softmax_type", "vanilla") != "vanilla":
        raise ValueError(
            f"A.X-K3 softmax_type={config.softmax_type!r} (learnable attention "
            "sink) is not supported."
        )
    if getattr(config, "use_hysparse", False):
        raise ValueError("A.X-K3 use_hysparse is not supported.")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and _SLIDING_ATTENTION in layer_types:
        # Sliding layers run only as top-k consumers over the whole context;
        # a window shorter than the context would need a real SWA mask.
        if not getattr(config, "dsa_index_share", False):
            raise ValueError(
                "A.X-K3 sliding_attention layers require dsa_index_share=true."
            )
        window = getattr(config, "swa_window_size", None)
        if window is not None and window < config.max_position_embeddings:
            raise ValueError(
                f"A.X-K3 swa_window_size={window} below "
                f"max_position_embeddings={config.max_position_embeddings} "
                "is not supported."
            )
    if getattr(config, "gated_residual", False):
        final = getattr(config, "gated_residual_final", "gated_read")
        if final != "gated_read":
            raise ValueError(f"A.X-K3 gated_residual_final={final!r} is not supported.")
