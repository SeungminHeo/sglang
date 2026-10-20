"""Separate HC gate preparation and combine, with optional next-branch norm.

The decode specialization accepts four branches of BF16 values; gate and apply
take any branch width the split combine supports, apply_norm only 2560. Gate
partials belong to each invocation, so overlapping sublayers never reuse them.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_module(hidden_size: int = 2560):
    args = make_cpp_args(hidden_size)
    return load_jit(
        "hc_combine_decode",
        *args,
        cuda_files=["elementwise/hc_combine_decode.cuh"],
        cuda_wrappers=[
            ("gate", f"HcCombineDecode<{args}>::gate"),
            ("apply", f"HcCombineDecode<{args}>::apply"),
            ("apply_norm", "HcCombineNormDecode<2>::run"),
        ],
    )


def hc_combine_gate(normed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partials = torch.empty(
        (normed.shape[0], 8, 4), device=normed.device, dtype=torch.float32
    )
    _jit_module(normed.shape[-1] // 4).gate(normed, weight, partials)
    return partials


def hc_combine_apply(
    block: torch.Tensor, residual: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    combined = torch.empty_like(residual)
    _jit_module(block.shape[-1]).apply(block, residual, partials, combined)
    return combined


def hc_combine_apply_norm(
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _jit_module().apply_norm(
        block, residual, partials, weight, combined, normalized, eps
    )
    return combined, normalized
