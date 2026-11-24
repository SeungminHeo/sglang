"""Compare Solar's beta=2*sigmoid KDA decode and chain-verify kernels.

Times the complete replaced kernel chains inside CUDA graphs. State resets,
correctness checks, and compilation are outside timing. This measures KDA
kernel latency, not end-to-end serving throughput.

Example:
    python bench_solar_kda_fusion.py --heads 8 16 32 --batch-sizes 1 8 32 64
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.kda_fused_decode import kda_fused_decode
from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update


def make_inputs(batch, heads, steps):
    torch.manual_seed(42)
    dim = heads * 128

    def rand(shape, scale=0.2, dtype=torch.bfloat16):
        return (torch.randn(shape, device="cuda") * scale).to(dtype)

    return {
        "x": rand((batch * steps, 3 * dim)),
        "a": rand((batch * steps, dim)),
        "b": rand((batch * steps, heads), scale=2.0),
        "gate": rand((batch, dim)),
        "w": rand((3 * dim, 4), dtype=torch.float32),
        "A_log": rand((heads,), dtype=torch.float32),
        "dt": rand((dim,), dtype=torch.float32),
        "norm": torch.ones(128, device="cuda"),
        "conv": rand((batch + 2, 3, 3 * dim)),
        "state": rand((batch + 2, heads, 128, 128), 0.02, torch.float32),
        "indices": torch.arange(1, batch + 1, device="cuda", dtype=torch.int32),
        "inter_indices": torch.arange(batch, device="cuda", dtype=torch.int32),
        "cu": torch.arange(batch + 1, device="cuda", dtype=torch.int32) * steps,
    }


def build_runner(inp, batch, heads, steps, phase, fused):
    dim = heads * 128
    conv = inp["conv"].clone()
    state = inp["state"].clone()
    windows = torch.zeros(batch, steps, 3, 3 * dim, device="cuda", dtype=torch.bfloat16)
    checkpoints = torch.zeros(
        batch, steps, heads, 128, 128, device="cuda", dtype=torch.float32
    )
    weights = [x.t().contiguous() for x in inp["w"].split(dim)]
    bias = torch.zeros(3 * dim, device="cuda", dtype=torch.float32)

    def reset():
        conv.copy_(inp["conv"])
        state.copy_(inp["state"])
        windows.zero_()
        checkpoints.zero_()

    def run():
        if phase == "decode":
            if fused:
                return kda_fused_decode(
                    inp["x"],
                    inp["a"],
                    inp["b"],
                    conv,
                    *weights,
                    bias,
                    inp["A_log"],
                    inp["dt"],
                    inp["gate"],
                    inp["norm"],
                    state,
                    inp["indices"],
                    scale=128**-0.5,
                    onorm_eps=1e-5,
                    allow_neg_eigval=True,
                )
            qkv = causal_conv1d_update(
                inp["x"],
                conv.transpose(-1, -2),
                inp["w"],
                None,
                activation="silu",
                conv_state_indices=inp["indices"],
            )
            out = inp["x"].new_empty(batch, 1, heads, 128)
            fused_recurrent_kda_packed_decode(
                qkv,
                inp["a"],
                inp["b"],
                inp["A_log"],
                inp["dt"],
                128**-0.5,
                state,
                out,
                inp["indices"],
                use_qk_l2norm_in_kernel=True,
                allow_neg_eigval=True,
            )
            return rms_norm_gated(
                out.transpose(0, 1),
                inp["gate"].view(1, batch, heads, 128),
                inp["norm"],
                None,
                activation="sigmoid",
                eps=1e-5,
            )
        if fused:
            return fused_kda_conv_gating_verify(
                mixed_qkv=inp["x"],
                conv_weight=inp["w"],
                conv_bias=None,
                conv_state=conv.transpose(-1, -2),
                conv_state_indices=inp["indices"],
                intermediate_conv_window=windows.transpose(-1, -2),
                intermediate_state_indices=inp["inter_indices"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt"],
                ssm_states=state,
                cache_indices=inp["indices"],
                intermediate_states_buffer=checkpoints,
                scale=128**-0.5,
                T=steps,
                num_q_heads=heads,
                num_v_heads=heads,
                head_k_dim=128,
                head_v_dim=128,
                allow_neg_eigval=True,
            )
        qkv = (
            causal_conv1d_update(
                inp["x"].view(batch, steps, 3 * dim).transpose(1, 2),
                conv.transpose(-1, -2),
                inp["w"],
                None,
                activation="silu",
                conv_state_indices=inp["indices"],
                intermediate_conv_window=windows.transpose(-1, -2),
                intermediate_state_indices=inp["inter_indices"],
            )
            .transpose(1, 2)
            .reshape(batch * steps, 3 * dim)
        )
        q, k, v = [x.reshape(1, batch * steps, heads, 128) for x in qkv.split(dim, -1)]
        return fused_sigmoid_gating_delta_rule_update(
            softplus_beta=1.0,
            softplus_threshold=20.0,
            A_log=inp["A_log"],
            dt_bias=inp["dt"],
            q=q,
            k=k,
            v=v,
            a=inp["a"],
            b=inp["b"],
            initial_state_source=state,
            initial_state_indices=inp["indices"],
            cu_seqlens=inp["cu"],
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            allow_neg_eigval=True,
            disable_state_update=True,
            intermediate_states_buffer=checkpoints,
            intermediate_state_indices=inp["inter_indices"],
            cache_steps=steps,
        )

    return run, reset, (conv, state, windows, checkpoints)


def relative_rms(actual, expected):
    delta = actual.float() - expected.float()
    return (
        (delta.square().mean() / expected.float().square().mean().clamp_min(1e-30))
        .sqrt()
        .item()
    )


def benchmark_case(batch, heads, steps, phase, repetitions):
    inp = make_inputs(batch, heads, steps)
    baseline, reset_base, base_buffers = build_runner(
        inp, batch, heads, steps, phase, False
    )
    fused, reset_fused, fused_buffers = build_runner(
        inp, batch, heads, steps, phase, True
    )
    # Compile eagerly, then validate a replay: the serving path uses CUDA
    # graphs, including the read-only committed state during verification.
    baseline()
    fused()
    graphs, outputs = [], []
    for fn, reset in [(baseline, reset_base), (fused, reset_fused)]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        reset()
        graph.replay()
        graphs.append(graph)
        outputs.append(out)
    expected, actual = outputs
    output_error = relative_rms(actual, expected)
    state_idx = 1 if phase == "decode" else 3
    state_error = relative_rms(fused_buffers[state_idx], base_buffers[state_idx])
    if phase == "decode":
        torch.testing.assert_close(fused_buffers[0], base_buffers[0], rtol=0, atol=0)
    if phase == "verify":
        torch.testing.assert_close(fused_buffers[0], inp["conv"], rtol=0, atol=0)
        torch.testing.assert_close(fused_buffers[1], inp["state"], rtol=0, atol=0)
        torch.testing.assert_close(fused_buffers[2], base_buffers[2], rtol=0, atol=0)
    if output_error > 0.001 or state_error > 0.00001:
        raise AssertionError(
            f"{phase}: output relative RMS={output_error}, state={state_error}"
        )

    times = {"baseline_us": [], "fused_us": []}
    # Alternate order to reduce clock/temperature bias. No reset launches are timed.
    for repeat in range(repetitions):
        runners = [
            ("baseline_us", baseline, reset_base),
            ("fused_us", fused, reset_fused),
        ]
        for name, fn, reset in runners[:: 1 if repeat % 2 == 0 else -1]:
            reset()
            times[name].append(triton.testing.do_bench_cudagraph(fn, rep=100) * 1000)
    base_us, fused_us = [
        statistics.median(times[k]) for k in ("baseline_us", "fused_us")
    ]
    return {
        "phase": phase,
        "batch": batch,
        "heads": heads,
        "steps": steps,
        "baseline_us": base_us,
        "fused_us": fused_us,
        "speedup": base_us / fused_us,
        "output_relative_rms": output_error,
        "state_relative_rms": state_error,
        "backend_eligible": phase == "decode" or (batch * heads <= 128 and steps <= 8),
        "samples": times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["decode", "verify"],
        default=["decode", "verify"],
    )
    parser.add_argument("--heads", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--steps", nargs="+", type=int, default=[5])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "beta_scale": 2,
        "timing": "CUDA graph; complete replaced kernel chain",
        "results": [],
    }
    with torch.inference_mode():
        for phase in args.phases:
            for heads in args.heads:
                for batch in args.batch_sizes:
                    for steps in [1] if phase == "decode" else args.steps:
                        row = benchmark_case(
                            batch, heads, steps, phase, args.repetitions
                        )
                        report["results"].append(row)
                        print(json.dumps(row), flush=True)
                        if args.output:
                            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
