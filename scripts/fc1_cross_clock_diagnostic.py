#!/usr/bin/env python3
"""Cross-check CUDA events against perf_counter and CLOCK_MONOTONIC_RAW."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
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
ITERATIONS_PER_BLOCK = 5000
MEASURED_BLOCKS = 12
INPUT_SEED = 0
WORKLOAD_WARMUPS = 100
TIMING_WARMUP_BLOCKS = 3


def _command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _git(root: Path, *args: str) -> str | None:
    result = _command("git", *args, cwd=root)
    return result.stdout.strip() if result.returncode == 0 else None


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _compute_processes() -> dict[str, Any]:
    result = _command(
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits")
    return {
        "command": " ".join(result.args),
        "returncode": result.returncode,
        "processes": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def _gpu_query() -> dict[str, Any]:
    result = _command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits")
    return {
        "command": " ".join(result.args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _clock_info(value: Any) -> dict[str, Any]:
    return {
        "implementation": value.implementation,
        "monotonic": value.monotonic,
        "adjustable": value.adjustable,
        "resolution_seconds": value.resolution,
    }


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values),
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

    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    raw_supported = (
        raw_clock is not None
        and hasattr(time, "clock_gettime_ns")
        and hasattr(time, "clock_getres"))
    if not raw_supported:
        raise RuntimeError(
            "CLOCK_MONOTONIC_RAW is unavailable; monotonic_ns was not substituted")
    try:
        raw_resolution = time.clock_getres(raw_clock)
        time.clock_gettime_ns(raw_clock)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"CLOCK_MONOTONIC_RAW is unavailable: {type(error).__name__}: {error}; "
            "monotonic_ns was not substituted") from error

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
    kernel_release = platform.release()
    provenance = {
        "created_at_utc": created.isoformat(),
        "source_revision": _git(root, "rev-parse", "HEAD"),
        "source_branch": _git(root, "branch", "--show-current"),
        "source_status_porcelain": _git(root, "status", "--porcelain"),
        "os": platform.platform(),
        "kernel_release": kernel_release,
        "proc_version": _read_optional(Path("/proc/version")),
        "wsl": {
            "detected": "microsoft" in kernel_release.lower()
            or "microsoft" in (_read_optional(Path("/proc/version")) or "").lower(),
            "interop_environment_present": "WSL_INTEROP" in __import__("os").environ,
        },
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "clocks": {
            "perf_counter": _clock_info(time.get_clock_info("perf_counter")),
            "clock_monotonic_raw": {
                "supported": True,
                "clock_id": raw_clock,
                "resolution_seconds": raw_resolution,
            },
        },
        "gpu_query": _gpu_query(),
        "competing_compute_process_check_before_import": process_snapshot,
        "gpu": {
            "device_index": device_index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
    }
    timestamp_order = [
        "torch.cuda.synchronize()",
        "perf_start_ns = time.perf_counter_ns()",
        "raw_start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)",
        "perf_after_raw_start_ns = time.perf_counter_ns()",
        "start_event.record(explicit_stream)",
        "submit 5000 operations on explicit_stream",
        "end_event.record(explicit_stream)",
        "end_event.synchronize()",
        "perf_end_ns = time.perf_counter_ns()",
        "raw_end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)",
        "perf_after_raw_end_ns = time.perf_counter_ns()",
        "start_event.elapsed_time(end_event)",
    ]
    config = {
        "purpose": "standalone CUDA-event versus two-host-clock cross-check",
        "session_budget_impact": "none",
        "implementation": "packaged preallocated Triton FC1 kernel",
        "workload": WORKLOAD,
        "triton_parameters": TRITON_PARAMETERS,
        "correctness": CORRECTNESS,
        "input_seed": INPUT_SEED,
        "iterations_per_block": ITERATIONS_PER_BLOCK,
        "measured_blocks": MEASURED_BLOCKS,
        "workload_warmup_operations": WORKLOAD_WARMUPS,
        "timing_warmup_blocks": TIMING_WARMUP_BLOCKS,
        "stream": "one dedicated explicit torch.cuda.Stream for events and workload",
        "timestamp_read_order": timestamp_order,
        "host_read_overhead_definition": (
            "perf_counter span bracketing each CLOCK_MONOTONIC_RAW read; "
            "clock absolute timestamps are never compared"),
        "interpretation": (
            "No previous fitted correction is applied. Agreement does not prove absolute accuracy."),
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
            num_stages=parameters.num_stages)

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

    def measure(block_id: int) -> dict[str, float | int]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        perf_start_ns = time.perf_counter_ns()
        raw_start_ns = time.clock_gettime_ns(raw_clock)
        perf_after_raw_start_ns = time.perf_counter_ns()
        start_event.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(ITERATIONS_PER_BLOCK):
                launch()
        end_event.record(stream)
        end_event.synchronize()
        perf_end_ns = time.perf_counter_ns()
        raw_end_ns = time.clock_gettime_ns(raw_clock)
        perf_after_raw_end_ns = time.perf_counter_ns()
        cuda_ms = float(start_event.elapsed_time(end_event))

        perf_ms = (perf_end_ns - perf_start_ns) / 1e6
        raw_ms = (raw_end_ns - raw_start_ns) / 1e6
        return {
            "block_id": block_id,
            "iteration_count": ITERATIONS_PER_BLOCK,
            "perf_start_ns": perf_start_ns,
            "raw_start_ns": raw_start_ns,
            "perf_after_raw_start_ns": perf_after_raw_start_ns,
            "perf_end_ns": perf_end_ns,
            "raw_end_ns": raw_end_ns,
            "perf_after_raw_end_ns": perf_after_raw_end_ns,
            "start_clock_read_span_ns": perf_after_raw_start_ns - perf_start_ns,
            "end_clock_read_span_ns": perf_after_raw_end_ns - perf_end_ns,
            "perf_counter_ms_per_block": perf_ms,
            "monotonic_raw_ms_per_block": raw_ms,
            "cuda_event_ms_per_block": cuda_ms,
            "raw_minus_perf_ms_per_block": raw_ms - perf_ms,
            "raw_over_perf_ratio": raw_ms / perf_ms,
            "cuda_minus_perf_ms_per_block": cuda_ms - perf_ms,
            "cuda_over_perf_ratio": cuda_ms / perf_ms,
            "cuda_minus_raw_ms_per_block": cuda_ms - raw_ms,
            "cuda_over_raw_ratio": cuda_ms / raw_ms,
            "perf_counter_ms_per_operation": perf_ms / ITERATIONS_PER_BLOCK,
            "monotonic_raw_ms_per_operation": raw_ms / ITERATIONS_PER_BLOCK,
            "cuda_event_ms_per_operation": cuda_ms / ITERATIONS_PER_BLOCK,
        }

    for warmup_id in range(1, TIMING_WARMUP_BLOCKS + 1):
        measure(-warmup_id)
    rows = [measure(block_id) for block_id in range(1, MEASURED_BLOCKS + 1)]

    fields = list(rows[0])
    with (destination / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as stream_file:
        writer = csv.DictWriter(stream_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(destination / "raw_measurements.json", rows)

    summary = {
        "perf_counter_ms_per_block": _stats(
            [row["perf_counter_ms_per_block"] for row in rows]),
        "monotonic_raw_ms_per_block": _stats(
            [row["monotonic_raw_ms_per_block"] for row in rows]),
        "cuda_event_ms_per_block": _stats(
            [row["cuda_event_ms_per_block"] for row in rows]),
        "raw_minus_perf_ms_per_block": _stats(
            [row["raw_minus_perf_ms_per_block"] for row in rows]),
        "raw_over_perf_ratio": _stats([row["raw_over_perf_ratio"] for row in rows]),
        "cuda_minus_perf_ms_per_block": _stats(
            [row["cuda_minus_perf_ms_per_block"] for row in rows]),
        "cuda_over_perf_ratio": _stats([row["cuda_over_perf_ratio"] for row in rows]),
        "cuda_minus_raw_ms_per_block": _stats(
            [row["cuda_minus_raw_ms_per_block"] for row in rows]),
        "cuda_over_raw_ratio": _stats([row["cuda_over_raw_ratio"] for row in rows]),
        "start_clock_read_span_ns": _stats(
            [row["start_clock_read_span_ns"] for row in rows]),
        "end_clock_read_span_ns": _stats(
            [row["end_clock_read_span_ns"] for row in rows]),
        "interpretation_note": (
            "Only elapsed durations are compared. Agreement does not prove absolute accuracy. "
            "No correction factor is applied."),
    }
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "result.json", {
        "schema_version": 1,
        "status": "success",
        "run_id": run_id,
        "run_directory": str(destination),
        "config": config,
        "provenance": provenance,
        "correctness": correctness,
        "summary": summary,
    })
    return destination


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/fc1-cross-clock-diagnostics"))
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
