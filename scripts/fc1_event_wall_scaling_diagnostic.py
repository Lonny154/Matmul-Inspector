#!/usr/bin/env python3
"""Compare CUDA-event and enclosing host-wall block times across block sizes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence
import uuid


WORKLOAD = {"m": 512, "k": 768, "n": 3072, "dtype": "float16"}
TRITON_PARAMETERS = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3,
    "group_size_m": 1,
}
CORRECTNESS = {"atol": 0.01, "rtol": 0.01}
BLOCK_SIZES = (50, 500, 5000)
ROUNDS = 12
INPUT_SEED = 0
WORKLOAD_WARMUPS = 100
EVENT_WARMUPS = 10


def _command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _git(root: Path, *args: str) -> str | None:
    result = _command("git", *args, cwd=root)
    return result.stdout.strip() if result.returncode == 0 else None


def _nvidia_query(fields: str) -> dict[str, Any]:
    result = _command("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    return {
        "command": " ".join(result.args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _compute_processes() -> dict[str, Any]:
    result = _command(
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "command": " ".join(result.args),
        "returncode": result.returncode,
        "processes": lines,
        "stderr": result.stderr.strip(),
    }


def build_schedule(rounds: int = ROUNDS) -> list[dict[str, int]]:
    """Cycle through all six permutations to balance block-size positions."""
    permutations = (
        (50, 500, 5000),
        (500, 5000, 50),
        (5000, 50, 500),
        (5000, 500, 50),
        (500, 50, 5000),
        (50, 5000, 500),
    )
    schedule = []
    block_id = 0
    for round_id in range(1, rounds + 1):
        for position, iterations in enumerate(permutations[(round_id - 1) % 6], 1):
            block_id += 1
            schedule.append({
                "block_id": block_id,
                "round": round_id,
                "position_in_round": position,
                "iteration_count": iterations,
            })
    return schedule


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_absolute_deviation": statistics.median(deviations),
    }


def _linear_fit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(row["host_wall_ms_per_block"]) for row in rows]
    ys = [float(row["cuda_event_ms_per_block"]) for row in rows]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    total = sum((y - y_mean) ** 2 for y in ys)
    residual_sum = sum(value * value for value in residuals)
    return {
        "model": "cuda_ms = intercept_ms + slope * host_ms",
        "intercept_ms": intercept,
        "slope": slope,
        "r_squared": 1.0 - residual_sum / total if total else None,
        "residual_ms": _stats(residuals),
        "residual_root_mean_square_ms": math.sqrt(residual_sum / len(residuals)),
        "points": len(rows),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run(output_root: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    process_snapshot = _compute_processes()
    if process_snapshot["returncode"] != 0:
        raise RuntimeError(
            "could not check competing GPU compute processes: "
            f"{process_snapshot['stderr'] or 'unknown nvidia-smi error'}")
    if process_snapshot["processes"]:
        raise RuntimeError(
            "competing GPU compute work is active; no measurements were launched: "
            + "; ".join(process_snapshot["processes"]))

    try:
        import torch
        import triton
        from matmul_inspector import triton_fc1
    except ImportError as error:
        raise RuntimeError("CUDA-enabled PyTorch, Triton, and Matmul Inspector are required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    created = datetime.now(timezone.utc)
    run_id = f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    destination = output_root.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    gpu_snapshot = _nvidia_query(
        "name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu")
    provenance = {
        "created_at_utc": created.isoformat(),
        "source_revision": _git(root, "rev-parse", "HEAD"),
        "source_branch": _git(root, "branch", "--show-current"),
        "source_status_porcelain": _git(root, "status", "--porcelain"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "driver_query": gpu_snapshot,
        "competing_compute_process_check_before_import": process_snapshot,
        "gpu": {
            "device_index": device_index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
    }
    config = {
        "purpose": "standalone CUDA-event versus synchronized host-wall scaling diagnostic",
        "session_budget_impact": "none",
        "implementation": "packaged preallocated Triton FC1 kernel",
        "workload": WORKLOAD,
        "triton_parameters": TRITON_PARAMETERS,
        "correctness": CORRECTNESS,
        "input_seed": INPUT_SEED,
        "block_sizes": list(BLOCK_SIZES),
        "rounds_per_block_size": ROUNDS,
        "workload_warmup_operations": WORKLOAD_WARMUPS,
        "event_warmup_blocks": EVENT_WARMUPS,
        "stream": "one dedicated explicit torch.cuda.Stream for events and workload",
        "timing_sequence": [
            "torch.cuda.synchronize()",
            "host_start_ns = time.perf_counter_ns()",
            "start_event.record(explicit_stream)",
            "submit iteration_count operations on explicit_stream",
            "end_event.record(explicit_stream)",
            "end_event.synchronize()",
            "host_end_ns = time.perf_counter_ns()",
            "start_event.elapsed_time(end_event)",
        ],
        "interpretation": (
            "CUDA-event and host-wall values are different timer measurements. "
            "No cause or correction is assumed."),
    }
    _write_json(destination / "config.json", config)
    _write_json(destination / "provenance.json", provenance)

    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed_all(INPUT_SEED)
    a = torch.randn((WORKLOAD["m"], WORKLOAD["k"]), device=device_index,
                    dtype=torch.float16)
    b = torch.randn((WORKLOAD["k"], WORKLOAD["n"]), device=device_index,
                    dtype=torch.float16)
    output = torch.empty((WORKLOAD["m"], WORKLOAD["n"]), device=device_index,
                         dtype=torch.float16)
    stream = torch.cuda.Stream(device=device_index)
    parameters = triton_fc1.TritonGemmConfig(**TRITON_PARAMETERS)
    grid = (
        triton.cdiv(WORKLOAD["m"], parameters.block_m)
        * triton.cdiv(WORKLOAD["n"], parameters.block_n),
    )

    def launch() -> None:
        triton_fc1.fc1_kernel[grid](
            a, b, output,
            WORKLOAD["m"], WORKLOAD["n"], WORKLOAD["k"],
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=parameters.block_m,
            BLOCK_N=parameters.block_n,
            BLOCK_K=parameters.block_k,
            GROUP_SIZE_M=parameters.group_size_m,
            num_warps=parameters.num_warps,
            num_stages=parameters.num_stages,
        )

    reference = torch.matmul(a, b)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()
    difference = (output.float() - reference.float()).abs()
    correctness = {
        "passed": bool(
            tuple(output.shape) == tuple(reference.shape)
            and torch.isfinite(output).all().item()
            and torch.isfinite(reference).all().item()
            and torch.allclose(output, reference, **CORRECTNESS)),
        "shape_match": tuple(output.shape) == tuple(reference.shape),
        "output_nonfinite_count": int((~torch.isfinite(output)).sum().item()),
        "reference_nonfinite_count": int((~torch.isfinite(reference)).sum().item()),
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        **CORRECTNESS,
    }
    _write_json(destination / "correctness.json", correctness)
    if not correctness["passed"]:
        raise RuntimeError(f"correctness failed; partial artifacts retained at {destination}")

    with torch.cuda.stream(stream):
        for _ in range(WORKLOAD_WARMUPS):
            launch()
    stream.synchronize()

    def measure(iterations: int) -> dict[str, float | int]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start_ns = time.perf_counter_ns()
        start_event.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                launch()
        end_event.record(stream)
        end_event.synchronize()
        host_end_ns = time.perf_counter_ns()
        cuda_ms = float(start_event.elapsed_time(end_event))
        host_ms = (host_end_ns - host_start_ns) / 1e6
        return {
            "host_start_ns": host_start_ns,
            "host_end_ns": host_end_ns,
            "host_wall_ms_per_block": host_ms,
            "cuda_event_ms_per_block": cuda_ms,
            "event_minus_wall_ms_per_block": cuda_ms - host_ms,
            "event_over_wall_ratio": cuda_ms / host_ms,
            "host_wall_ms_per_operation": host_ms / iterations,
            "cuda_event_ms_per_operation": cuda_ms / iterations,
            "event_minus_wall_ms_per_operation": (cuda_ms - host_ms) / iterations,
        }

    for warmup_index in range(EVENT_WARMUPS):
        measure(BLOCK_SIZES[warmup_index % len(BLOCK_SIZES)])

    schedule = build_schedule()
    rows = []
    for entry in schedule:
        rows.append({**entry, **measure(entry["iteration_count"])})

    fields = [
        "block_id", "round", "position_in_round", "iteration_count",
        "host_start_ns", "host_end_ns", "host_wall_ms_per_block",
        "cuda_event_ms_per_block", "event_minus_wall_ms_per_block",
        "event_over_wall_ratio", "host_wall_ms_per_operation",
        "cuda_event_ms_per_operation", "event_minus_wall_ms_per_operation",
    ]
    with (destination / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as stream_file:
        writer = csv.DictWriter(stream_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(destination / "raw_measurements.json", rows)

    by_size = {}
    for size in BLOCK_SIZES:
        selected = [row for row in rows if row["iteration_count"] == size]
        by_size[str(size)] = {
            "host_wall_ms_per_block": _stats(
                [row["host_wall_ms_per_block"] for row in selected]),
            "cuda_event_ms_per_block": _stats(
                [row["cuda_event_ms_per_block"] for row in selected]),
            "event_minus_wall_ms_per_block": _stats(
                [row["event_minus_wall_ms_per_block"] for row in selected]),
            "event_over_wall_ratio": _stats(
                [row["event_over_wall_ratio"] for row in selected]),
            "host_wall_ms_per_operation": _stats(
                [row["host_wall_ms_per_operation"] for row in selected]),
            "cuda_event_ms_per_operation": _stats(
                [row["cuda_event_ms_per_operation"] for row in selected]),
        }
    summary = {
        "block_size_summaries": by_size,
        "descriptive_linear_fit": _linear_fit(rows),
        "classification": "not assigned by the measurement script",
        "classification_note": (
            "Use the paired excess, ratios, and residuals to assess additive, "
            "proportional, mixed, or inconclusive behavior without applying a correction."),
    }
    _write_json(destination / "summary.json", summary)
    result = {
        "schema_version": 1,
        "status": "success",
        "run_id": run_id,
        "run_directory": str(destination),
        "config": config,
        "provenance": provenance,
        "correctness": correctness,
        "actual_execution_order": schedule,
        "summary": summary,
    }
    _write_json(destination / "result.json", result)
    return destination


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/fc1-event-wall-scaling-diagnostics"))
    args = parser.parse_args(arguments)
    try:
        destination = run(args.output_root)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
