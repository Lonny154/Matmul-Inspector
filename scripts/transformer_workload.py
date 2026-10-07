#!/usr/bin/env python3
"""Miniature Transformer block with independently selectable Triton GEMMs."""

import argparse
from dataclasses import dataclass
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_fc1 import FC1_CONFIG, FC2_CONFIG, QKV_CONFIG, triton_gemm


@dataclass(frozen=True)
class ProjectionSelection:
    qkv: bool = False
    fc1: bool = False
    fc2: bool = False

    def label(self) -> str:
        enabled = [
            name
            for name, selected in (
                ("qkv", self.qkv),
                ("fc1", self.fc1),
                ("fc2", self.fc2),
            )
            if selected
        ]
        return "triton-" + "+".join(enabled) if enabled else "pytorch"

    def any_enabled(self) -> bool:
        return self.qkv or self.fc1 or self.fc2


@dataclass(frozen=True)
class CorrectnessResult:
    operation: str
    correct: bool
    max_abs_error: float
    mean_abs_error: float


class MiniTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_size: int):
        super().__init__()

        if hidden_size % num_heads:
            raise ValueError("hidden size must be divisible by number of heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, mlp_size, bias=False)
        self.fc2 = nn.Linear(mlp_size, hidden_size, bias=False)

        # These buffers are populated after the module is moved to CUDA. They
        # are contiguous KxN matrices consumed directly by the Triton kernel.
        self.register_buffer("qkv_weight_t", None, persistent=False)
        self.register_buffer("fc1_weight_t", None, persistent=False)
        self.register_buffer("fc2_weight_t", None, persistent=False)

    def prepare_triton_weights(self) -> None:
        """Transpose projection weights once, outside forward/timed regions."""
        self.qkv_weight_t = self.qkv.weight.T.contiguous()
        self.fc1_weight_t = self.fc1.weight.T.contiguous()
        self.fc2_weight_t = self.fc2.weight.T.contiguous()

    @staticmethod
    def _project(x, pytorch_layer, triton_weight, use_triton, config):
        if not use_triton:
            return pytorch_layer(x)
        if triton_weight is None:
            raise RuntimeError("call prepare_triton_weights() before Triton execution")

        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        output = triton_gemm(x_2d, triton_weight, config)
        return output.reshape(*shape[:-1], triton_weight.shape[1])

    def forward(self, x, projections=ProjectionSelection()):
        batch, seq_len, hidden = x.shape
        residual = x

        with torch.cuda.nvtx.range("layernorm_1"):
            x = self.ln1(x)

        with torch.cuda.nvtx.range("qkv_projection"):
            qkv = self._project(
                x,
                self.qkv,
                self.qkv_weight_t,
                projections.qkv,
                QKV_CONFIG,
            )

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        with torch.cuda.nvtx.range("attention"):
            attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch, seq_len, hidden)

        with torch.cuda.nvtx.range("attention_output_projection"):
            x = self.attn_out(attention)

        x = x + residual
        residual = x

        with torch.cuda.nvtx.range("layernorm_2"):
            x = self.ln2(x)

        with torch.cuda.nvtx.range("mlp_fc1"):
            mlp = self._project(
                x,
                self.fc1,
                self.fc1_weight_t,
                projections.fc1,
                FC1_CONFIG,
            )

        with torch.cuda.nvtx.range("gelu"):
            mlp = F.gelu(mlp)

        with torch.cuda.nvtx.range("mlp_fc2"):
            mlp = self._project(
                mlp,
                self.fc2,
                self.fc2_weight_t,
                projections.fc2,
                FC2_CONFIG,
            )

        return mlp + residual


def compare_tensors(operation, reference, candidate, atol, rtol):
    error = (candidate - reference).abs().float()
    return CorrectnessResult(
        operation=operation,
        correct=torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        max_abs_error=error.max().item(),
        mean_abs_error=error.mean().item(),
    )


def validate_projections(model, x, projections, atol, rtol):
    """Check selected GEMMs and the integrated block against PyTorch."""
    results = []
    with torch.inference_mode():
        qkv_input = model.ln1(x)
        fc1_input = model.ln2(x)
        fc2_input = F.gelu(model.fc1(fc1_input))

        checks = (
            (
                "qkv",
                projections.qkv,
                qkv_input,
                model.qkv,
                model.qkv_weight_t,
                QKV_CONFIG,
            ),
            (
                "fc1",
                projections.fc1,
                fc1_input,
                model.fc1,
                model.fc1_weight_t,
                FC1_CONFIG,
            ),
            (
                "fc2",
                projections.fc2,
                fc2_input,
                model.fc2,
                model.fc2_weight_t,
                FC2_CONFIG,
            ),
        )
        for name, enabled, projection_input, layer, weight_t, config in checks:
            if not enabled:
                continue
            reference = layer(projection_input)
            candidate = model._project(
                projection_input,
                layer,
                weight_t,
                True,
                config,
            )
            results.append(
                compare_tensors(name, reference, candidate, atol=atol, rtol=rtol)
            )

        if projections.any_enabled():
            reference = model(x, ProjectionSelection())
            candidate = model(x, projections)
            results.append(
                compare_tensors(
                    "transformer_block",
                    reference,
                    candidate,
                    atol=atol,
                    rtol=rtol,
                )
            )
    return results


def benchmark_callable_once(function, warmup, iterations):
    """Return mean CUDA-event latency for one repeated callable sample."""
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


def benchmark_once(model, x, projections, warmup, iterations):
    """Return mean CUDA-event latency for one repeated-forward sample."""
    return benchmark_callable_once(
        lambda: model(x, projections),
        warmup,
        iterations,
    )


def benchmark_alternating(model, x, candidate, warmup, iterations, repeats):
    """Alternate baseline/candidate ordering to expose run-order variability."""
    baseline = ProjectionSelection()
    samples = {baseline.label(): [], candidate.label(): []}

    if not candidate.any_enabled():
        for _ in range(repeats):
            samples[baseline.label()].append(
                benchmark_once(model, x, baseline, warmup, iterations)
            )
        return samples

    for repeat in range(repeats):
        order = (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
        for projections in order:
            samples[projections.label()].append(
                benchmark_once(model, x, projections, warmup, iterations)
            )
    return samples


def benchmark_selected_projections(
    model,
    x,
    projections,
    warmup,
    iterations,
    repeats,
):
    """Measure selected projection GEMMs without surrounding block operations."""
    with torch.inference_mode():
        qkv_input = model.ln1(x)
        fc1_input = model.ln2(x)
        fc2_input = F.gelu(model.fc1(fc1_input))

    checks = (
        (
            "qkv",
            projections.qkv,
            qkv_input,
            model.qkv,
            model.qkv_weight_t,
            QKV_CONFIG,
        ),
        (
            "fc1",
            projections.fc1,
            fc1_input,
            model.fc1,
            model.fc1_weight_t,
            FC1_CONFIG,
        ),
        (
            "fc2",
            projections.fc2,
            fc2_input,
            model.fc2,
            model.fc2_weight_t,
            FC2_CONFIG,
        ),
    )
    results = {}
    for name, enabled, projection_input, layer, weight_t, config in checks:
        if not enabled:
            continue
        samples = {"pytorch": [], "triton": []}
        functions = {
            "pytorch": lambda layer=layer, value=projection_input: layer(value),
            "triton": lambda value=projection_input, layer=layer, weight=weight_t, cfg=config: (
                model._project(value, layer, weight, True, cfg)
            ),
        }
        for repeat in range(repeats):
            order = ("pytorch", "triton") if repeat % 2 == 0 else ("triton", "pytorch")
            for implementation in order:
                samples[implementation].append(
                    benchmark_callable_once(
                        functions[implementation],
                        warmup,
                        iterations,
                    )
                )
        results[name] = samples
    return results


def profile_workload(model, x, projections, warmup, iterations):
    """Run only the selected path for external Nsight profiling."""
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, projections)
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("profiled_transformer_iterations"):
            for _ in range(iterations):
                model(x, projections)
        torch.cuda.synchronize()


def print_timing_summary(samples, tokens_per_block):
    for label, values in samples.items():
        mean_ms = statistics.fmean(values)
        median_ms = statistics.median(values)
        blocks_per_second = 1000.0 / median_ms
        print(
            f"{label:24} median={median_ms:.4f} ms "
            f"mean={mean_ms:.4f} ms min={min(values):.4f} ms "
            f"max={max(values):.4f} ms tokens/s="
            f"{tokens_per_block * blocks_per_second:.2f}"
        )


def print_projection_timings(results):
    for operation, samples in results.items():
        pytorch_median = statistics.median(samples["pytorch"])
        triton_median = statistics.median(samples["triton"])
        print(
            f"Projection {operation:4} PyTorch median={pytorch_median * 1000:.2f} us "
            f"Triton median={triton_median * 1000:.2f} us "
            f"ratio={pytorch_median / triton_median:.4f}x"
        )


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq", type=positive_int, default=512)
    parser.add_argument("--hidden", type=positive_int, default=768)
    parser.add_argument("--heads", type=positive_int, default=12)
    parser.add_argument("--mlp", type=positive_int, default=3072)
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--triton-qkv", action="store_true")
    parser.add_argument("--triton-fc1", action="store_true")
    parser.add_argument("--triton-fc2", action="store_true")
    parser.add_argument(
        "--benchmark-projections",
        action="store_true",
        help="also time each selected GEMM outside the full Transformer block",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run only the selected path for external Nsight profiling",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    if args.hidden % args.heads:
        raise ValueError("--hidden must be divisible by --heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = MiniTransformerBlock(args.hidden, args.heads, args.mlp).to(
        device=device,
        dtype=torch.float16,
    )
    model.prepare_triton_weights()
    model.eval()

    x = torch.randn(
        args.batch,
        args.seq,
        args.hidden,
        device=device,
        dtype=torch.float16,
    )
    projections = ProjectionSelection(
        qkv=args.triton_qkv,
        fc1=args.triton_fc1,
        fc2=args.triton_fc2,
    )

    print("Mini Transformer workload")
    print("-------------------------")
    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"Shape:        batch={args.batch} seq={args.seq} hidden={args.hidden}")
    print(f"Heads/MLP:    {args.heads}/{args.mlp}")
    print(f"Candidate:    {projections.label()}")
    print(f"Dtype:        {x.dtype}")

    correctness = validate_projections(
        model,
        x,
        projections,
        atol=args.atol,
        rtol=args.rtol,
    )
    for result in correctness:
        print(
            f"Correctness {result.operation:17} "
            f"{'PASS' if result.correct else 'FAIL'} "
            f"max_abs={result.max_abs_error:.6g} "
            f"mean_abs={result.mean_abs_error:.6g}"
        )
    if any(not result.correct for result in correctness):
        raise RuntimeError("Triton correctness validation failed")

    if args.profile:
        profile_workload(model, x, projections, args.warmup, args.iterations)
        print("Profile workload completed; reported latency belongs to the profiler.")
        return 0

    if args.benchmark_projections:
        projection_samples = benchmark_selected_projections(
            model,
            x,
            projections,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        print_projection_timings(projection_samples)

    samples = benchmark_alternating(
        model,
        x,
        projections,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    print_timing_summary(samples, args.batch * args.seq)
    if projections.any_enabled():
        baseline_median = statistics.median(samples["pytorch"])
        candidate_median = statistics.median(samples[projections.label()])
        print(f"Median ratio (PyTorch/candidate): {baseline_median / candidate_median:.4f}x")
        print("Treat this as a speedup only when repeated runs remain consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
