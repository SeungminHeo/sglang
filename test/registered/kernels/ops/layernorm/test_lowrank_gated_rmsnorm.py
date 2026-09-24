# SPDX-License-Identifier: Apache-2.0
"""Reference tests for the fused low-rank gated RMSNorm (A.X-K2 gated norm)."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.layernorm.lowrank_gated_rmsnorm import (  # noqa: E402
    lowrank_gated_rmsnorm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
EPS = 1e-6


def _reference(x, residual, weight, w_down, w_up, dtype):
    """The unfused AXK2GatedRMSNorm path: fused_add_rmsnorm -> bf16 y ->
    W_down -> silu -> W_up -> y * sigmoid(gate) (fp32 math, dtype storage)."""
    h = x.float() + (residual.float() if residual is not None else 0.0)
    var = h.pow(2).mean(dim=-1, keepdim=True)
    y = (h * torch.rsqrt(var + EPS) * weight.float()).to(dtype)
    gate = F.linear(F.silu(F.linear(y, w_down)), w_up)
    out = (y.float() * torch.sigmoid(gate.float())).to(dtype)
    return out, (h.to(dtype) if residual is not None else None)


@pytest.mark.parametrize("tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("hidden,rank", [(7168, 16), (1024, 4)])
@pytest.mark.parametrize("with_residual", [True, False])
@torch.inference_mode()
def test_lowrank_gated_rmsnorm_matches_unfused(tokens, hidden, rank, with_residual):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=dtype)
    residual = (
        torch.randn(tokens, hidden, device=DEVICE, dtype=dtype)
        if with_residual
        else None
    )
    weight = (1.0 + 0.1 * torch.randn(hidden, device=DEVICE)).to(dtype)
    w_down = (0.05 * torch.randn(rank, hidden, device=DEVICE)).to(dtype)
    w_up = (0.05 * torch.randn(hidden, rank, device=DEVICE)).to(dtype)

    ref_out, ref_res = _reference(x, residual, weight, w_down, w_up, dtype)

    x_in = x.clone()
    res_in = residual.clone() if with_residual else None
    out, res_out = lowrank_gated_rmsnorm(
        x_in,
        weight,
        w_down,
        w_up,
        EPS,
        residual=res_in,
        out=x_in if with_residual else None,
    )

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
    if with_residual:
        assert out.data_ptr() == x_in.data_ptr()
        torch.testing.assert_close(res_out, ref_res, atol=2e-2, rtol=2e-2)
    else:
        assert res_out is None
        torch.testing.assert_close(x_in, x)  # input left untouched


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
