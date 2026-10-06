#!/usr/bin/env python3
"""Benchmark the existing full-tiled and pretransposed NKI GEMM variants."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Protocol, Sequence


TILE_M = 128
TILE_K = 128
TILE_N = 512
VARIANTS = ("full-tiled", "pretransposed")


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int


@dataclass(frozen=True)
class Config:
    shapes: tuple[Shape, ...]
    variants: tuple[str, ...]
    seed: int
    warmups: int
    iterations: int
    atol: float
    rtol: float
    backend: str
    output: Path


@dataclass
class Execution:
    output: Any
    samples_ms: list[float]


class ExecutionBackend(Protocol):
    """Backend seam for simulator now and Trainium timing/profiling later."""

    name: str
    timing_mode: str
    timing_note: str

    def execute(
        self, variant: str, a: Any, b: Any, warmups: int, iterations: int
    ) -> Execution: ...

    def metadata(self) -> dict[str, Any]: ...


class SimulatorBackend:
    """NKI functional simulator with explicitly non-hardware wall-clock timing."""

    name = "nki-simulator"
    timing_mode = "simulator_wall_clock"
    timing_note = (
        "Host wall-clock time around nki.simulate execution; excludes input creation, "
        "reference GEMM, correctness analysis, and pretranspose preparation. It is not "
        "Trainium kernel performance."
    )

    def __init__(self) -> None:
        try:
            import nki
            from nki_matmul_full_tiled import matmul_kernel_mn_tiled as full_tiled
            from nki_matmul_pretransposed import matmul_kernel_mn_tiled as pretransposed
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "NKI is unavailable. Run this harness in an AWS Neuron/NKI "
                "environment; CPU-only harness tests do not require NKI."
            ) from error
        self._nki = nki
        self._kernels = {
            "full-tiled": nki.simulate(full_tiled),
            "pretransposed": nki.simulate(pretransposed),
        }

    def execute(
        self, variant: str, a: Any, b: Any, warmups: int, iterations: int
    ) -> Execution:
        call = self._kernels[variant]
        # Establish a correctness output before any samples. Simulator setup and
        # first-use effects therefore do not silently become a measured sample.
        output = call(a, b)
        for _ in range(warmups):
            output = call(a, b)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            output = call(a, b)
            stop = time.perf_counter_ns()
            samples.append((stop - start) / 1_000_000.0)
        return Execution(output=output, samples_ms=samples)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "timing_mode": self.timing_mode,
            "nki_version": getattr(self._nki, "__version__", "unknown"),
        }


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def finite_nonnegative(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def parse_int_list(text: str, option: str) -> tuple[int, ...]:
    try:
        values = tuple(positive_int(part.strip()) for part in text.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(f"{option} requires positive integers") from error
    if not values or any(not part.strip() for part in text.split(",")):
        raise argparse.ArgumentTypeError(f"{option} must not be empty")
    return values


def parse_shapes(text: str) -> tuple[Shape, ...]:
    shapes = []
    for token in text.split(","):
        try:
            dimensions = tuple(positive_int(part.strip()) for part in token.lower().split("x"))
        except (ValueError, argparse.ArgumentTypeError) as error:
            raise argparse.ArgumentTypeError(
                "--shapes must be comma-separated MxKxN triples"
            ) from error
        if len(dimensions) != 3:
            raise argparse.ArgumentTypeError(
                "--shapes must be comma-separated MxKxN triples"
            )
        shapes.append(Shape(*dimensions))
    if not shapes:
        raise argparse.ArgumentTypeError("--shapes must not be empty")
    return tuple(shapes)


def validate_shape(shape: Shape) -> None:
    failures = []
    if shape.m % TILE_M:
        failures.append(f"M must be divisible by {TILE_M}")
    if shape.k % TILE_K:
        failures.append(f"K must be divisible by {TILE_K}")
    if shape.n % TILE_N:
        failures.append(f"N must be divisible by {TILE_N}")
    if failures:
        dimensions = f"{shape.m}x{shape.k}x{shape.n}"
        raise ValueError(
            f"Invalid NKI GEMM shape {dimensions}: " + "; ".join(failures)
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    shape_group = result.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--shapes", type=parse_shapes,
        help="Comma-separated MxKxN shapes (default: 128x128x512)",
    )
    shape_group.add_argument(
        "--cartesian", action="store_true",
        help="Use the Cartesian product of --m, --k, and --n lists",
    )
    result.add_argument("--m", default="128", help="Comma-separated M sizes with --cartesian")
    result.add_argument("--k", default="128", help="Comma-separated K sizes with --cartesian")
    result.add_argument("--n", default="512", help="Comma-separated N sizes with --cartesian")
    result.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated variants: full-tiled,pretransposed",
    )
    result.add_argument("--seed", type=nonnegative_int, default=42)
    result.add_argument("--warmups", type=nonnegative_int, default=1)
    result.add_argument("--iterations", type=positive_int, default=3)
    result.add_argument("--atol", type=finite_nonnegative, default=1e-4)
    result.add_argument("--rtol", type=finite_nonnegative, default=1e-4)
    result.add_argument(
        "--backend", default="simulate",
        help="Execution backend (currently: simulate; hardware can implement ExecutionBackend)",
    )
    result.add_argument("--output", type=Path, default=Path("results/nki-gemm-benchmark"))
    return result


def parse_args(arguments: Sequence[str] | None = None) -> Config:
    namespace = parser().parse_args(arguments)
    if namespace.shapes is not None:
        shapes = namespace.shapes
        if namespace.m != "128" or namespace.k != "128" or namespace.n != "512":
            parser().error("--m, --k, and --n cannot be combined with --shapes")
    else:
        m_values = parse_int_list(namespace.m, "--m")
        k_values = parse_int_list(namespace.k, "--k")
        n_values = parse_int_list(namespace.n, "--n")
        shapes = tuple(Shape(*values) for values in itertools.product(m_values, k_values, n_values))
    for shape in shapes:
        validate_shape(shape)
    variants = tuple(part.strip() for part in namespace.variants.split(",") if part.strip())
    invalid_variants = (
        not variants
        or len(set(variants)) != len(variants)
        or any(variant not in VARIANTS for variant in variants)
    )
    if invalid_variants:
        parser().error("--variants must contain unique full-tiled and/or pretransposed values")
    if namespace.backend != "simulate":
        parser().error(
            f"unsupported backend {namespace.backend!r}; add it through ExecutionBackend"
        )
    return Config(
        shapes=shapes,
        variants=variants,
        seed=namespace.seed,
        warmups=namespace.warmups,
        iterations=namespace.iterations,
        atol=namespace.atol,
        rtol=namespace.rtol,
        backend=namespace.backend,
        output=namespace.output,
    )


def timing_statistics(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
    return {
        "sample_count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stddev_ms": statistics.pstdev(samples),
    }


def prepare_variant_inputs(variant: str, a: Any, b: Any) -> tuple[Any, Any]:
    """Adapt logical inputs to a kernel interface outside the timed region."""
    if variant == "pretransposed":
        return a.T.copy(), b
    if variant == "full-tiled":
        return a, b
    raise ValueError(f"Unknown NKI GEMM variant: {variant}")


def run_benchmark(config: Config, backend: ExecutionBackend) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for shape_index, shape in enumerate(config.shapes):
        # A per-shape generator keeps both variants on identical inputs and
        # makes each shape independent of variant selection/order.
        rng = np.random.default_rng(config.seed + shape_index)
        a = rng.standard_normal((shape.m, shape.k), dtype=np.float32)
        b = rng.standard_normal((shape.k, shape.n), dtype=np.float32)
        reference = a @ b
        for variant in config.variants:
            kernel_a, kernel_b = prepare_variant_inputs(variant, a, b)
            execution = backend.execute(
                variant, kernel_a, kernel_b, config.warmups, config.iterations
            )
            output = np.asarray(execution.output, dtype=np.float32)
            if output.shape != reference.shape:
                raise ValueError(
                    f"{variant} returned {output.shape}; expected {reference.shape}"
                )
            error = np.abs(output - reference)
            row = {
                "variant": variant,
                "M": shape.m,
                "K": shape.k,
                "N": shape.n,
                "seed": config.seed + shape_index,
                "backend": backend.name,
                "timing_mode": backend.timing_mode,
                "correct": str(bool(np.allclose(
                    output, reference, atol=config.atol, rtol=config.rtol
                ))).lower(),
                "max_abs_error": float(error.max(initial=0.0)),
                "mean_abs_error": float(error.mean()) if error.size else 0.0,
                "atol": config.atol,
                "rtol": config.rtol,
                "warmups": config.warmups,
                "iterations": config.iterations,
                **timing_statistics(execution.samples_ms),
                "timing_note": backend.timing_note,
            }
            rows.append(row)
            print(
                f"{variant} {shape.m}x{shape.k}x{shape.n}: "
                f"correct={row['correct']} max_abs_error={row['max_abs_error']:.6g} "
                f"median_ms={row['median_ms']:.6g} ({backend.timing_mode})"
            )
    return rows


def write_artifacts(
    config: Config, backend: ExecutionBackend, rows: Sequence[dict[str, Any]]
) -> None:
    config.output.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (config.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "experiment": "nki-gemm",
        "config": {
            **asdict(config),
            "output": str(config.output),
            "shapes": [asdict(shape) for shape in config.shapes],
        },
        "backend": backend.metadata(),
        "timing_note": backend.timing_note,
        "kernel_sources": {
            "full-tiled": "scripts/nki_matmul_full_tiled.py",
            "pretransposed": "scripts/nki_matmul_pretransposed.py",
        },
    }
    (config.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(arguments)
        backend = SimulatorBackend()
        rows = run_benchmark(config, backend)
        write_artifacts(config, backend, rows)
        return 0 if all(row["correct"] == "true" for row in rows) else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"nki-gemm-benchmark: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
