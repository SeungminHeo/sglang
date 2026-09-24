# SPDX-License-Identifier: Apache-2.0
"""Fused low-rank gated RMSNorm (A.X-K2 ``AXK2GatedRMSNorm``) for small batches.

    residual = x + residual                       (optional)
    y        = rmsnorm(residual) * weight
    g        = silu(y @ W_down^T)                 # [tokens, R], R = gated_norm_rank (16)
    out      = y * sigmoid(g @ W_up^T)            # [tokens, N]

One program per token row does all of it in registers, so a decode step pays
one launch per norm instead of flashinfer fused_add_rmsnorm + two cuBLAS
split-K GEMVs (+ reduce) + silu + sigmoid-mul. Every program re-reads the
2 * R * N gate weights (0.46 MB for N=7168, R=16), which is free at decode
batch sizes but wasteful for long prefills; callers keep the unfused path
above ``LOWRANK_GATED_RMSNORM_MAX_TOKENS``.
``w_up`` keeps its PyTorch ``[N, R]`` layout (rows of R contiguous values).
The gate is computed with two 2-D tile reductions ([R, BLOCK_N] and
[BLOCK_N, R]); a per-rank loop of 1-D reductions is ~2.5x slower at M=1.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Above this many rows the unfused path (rmsnorm + cuBLAS GEMVs) is cheaper
# (H200, N=7168, R=16: fused 12 us vs 16.5-19 us up to M=128, crossover ~M=256):
# the fused kernel re-reads the gate weights once per row.
LOWRANK_GATED_RMSNORM_MAX_TOKENS = 128


@triton.jit
def _lowrank_gated_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    wd_ptr,
    wu_ptr,
    out_ptr,
    x_stride,
    res_stride,
    out_stride,
    eps,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        res = tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x = x + res
        # residual receives x + residual (same contract as fused_add_rmsnorm)
        tl.store(
            res_ptr + row * res_stride + cols, x.to(res_ptr.dtype.element_ty), mask=mask
        )

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # The unfused path hands a bf16 y to the gate GEMVs; round the same way so
    # the gate sees identical inputs.
    y = (x * rstd * w).to(out_ptr.dtype.element_ty).to(tl.float32)

    ranks = tl.arange(0, R)
    # z[R] = silu(y @ W_down^T)
    wd = tl.load(
        wd_ptr + ranks[:, None] * N + cols[None, :], mask=mask[None, :], other=0.0
    ).to(tl.float32)
    z = tl.sum(wd * y[None, :], axis=1)
    z = z * tl.sigmoid(z)
    # gate[N] = z @ W_up^T, W_up stored [N, R]
    wu = tl.load(
        wu_ptr + cols[:, None] * R + ranks[None, :], mask=mask[:, None], other=0.0
    ).to(tl.float32)
    gate = tl.sum(wu * z[None, :], axis=1)

    out = y * tl.sigmoid(gate)
    tl.store(
        out_ptr + row * out_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask
    )


def lowrank_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns ``(out, residual)``; ``residual`` is updated in place to
    ``x + residual`` when given. ``out`` defaults to a new tensor like ``x``.

    x: [tokens, N]; weight: [N]; w_down: [R, N]; w_up: [N, R] (nn.Linear layouts).
    R must be a power of two.
    """
    assert x.dim() == 2, x.shape
    tokens, N = x.shape
    R = w_down.shape[0]
    assert R & (R - 1) == 0, f"rank {R} must be a power of two"
    assert w_down.shape == (R, N) and w_up.shape == (N, R), (
        w_down.shape,
        w_up.shape,
        N,
    )
    assert w_down.is_contiguous() and w_up.is_contiguous() and weight.is_contiguous()
    assert x.stride(1) == 1
    if out is None:
        out = torch.empty_like(x)
    assert out.stride(1) == 1 and out.shape == x.shape
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(1) == 1
        res_ptr, res_stride = residual, residual.stride(0)
    else:
        res_ptr, res_stride = x, 0
    if tokens == 0:
        return out, residual
    BLOCK_N = triton.next_power_of_2(N)
    _lowrank_gated_rmsnorm_kernel[(tokens,)](
        x,
        res_ptr,
        weight,
        w_down,
        w_up,
        out,
        x.stride(0),
        res_stride,
        out.stride(0),
        eps,
        N=N,
        R=R,
        BLOCK_N=BLOCK_N,
        HAS_RES=residual is not None,
        num_warps=16 if BLOCK_N > 4096 else (8 if BLOCK_N > 2048 else 4),
    )
    return out, residual
