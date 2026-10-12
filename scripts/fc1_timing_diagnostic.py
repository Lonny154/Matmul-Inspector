#!/usr/bin/env python3
"""Measurement-method diagnostic for the fixed FC1 stages2 experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable
import uuid


WORKLOAD = {"m": 512, "k": 768, "n": 3072, "dtype": "float16"}
TRITON_PARAMETERS = {
    "block_m": 64, "block_n": 128, "block_k": 32,
    "num_warps": 4, "num_stages": 2, "group_size_m": 1,
}
CORRECTNESS = {"atol": 0.01, "rtol": 0.01}
WARMUP_OPERATIONS = 100
OPERATIONS_PER_BLOCK = 500
BLOCKS = 20
INPUT_SEED = 0
SCHEDULE_SEED = 20261012
MODES = ("allocating", "preallocated", "cuda_graph")
IMPLEMENTATIONS = ("baseline", "candidate")


def build_schedule(blocks: int = BLOCKS, seed: int = SCHEDULE_SEED,
                   available: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    """Build deterministic mode shuffles with alternating implementation order."""
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    sequence = 0
    for block in range(1, blocks + 1):
        modes = list(MODES)
        rng.shuffle(modes)
        implementations = list(IMPLEMENTATIONS if block % 2 else reversed(IMPLEMENTATIONS))
        for mode_position, mode in enumerate(modes, 1):
            allowed = set((available or {}).get(mode, IMPLEMENTATIONS))
            for implementation_position, implementation in enumerate(implementations, 1):
                if implementation not in allowed:
                    continue
                sequence += 1
                schedule.append({
                    "sequence": sequence, "block": block, "mode": mode,
                    "mode_position": mode_position, "implementation": implementation,
                    "implementation_position": implementation_position,
                })
    return schedule


def _stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    median = statistics.median(values)
    return {
        "count": len(values), "minimum": min(values), "median": median,
        "mean": mean, "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "coefficient_of_variation": (statistics.stdev(values) / mean
                                      if len(values) > 1 and mean else 0.0),
        "relative_range_over_median": ((max(values) - min(values)) / median if median else 0.0),
    }


def summarize_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in MODES:
        result[mode] = {}
        for implementation in IMPLEMENTATIONS:
            selected = [r for r in rows if r["mode"] == mode and r["implementation"] == implementation]
            if not selected:
                continue
            result[mode][implementation] = {
                "cuda_event_ms_per_operation": _stats([r["cuda_event_ms_per_operation"] for r in selected]),
                "host_submission_ms_per_block": _stats([r["host_submission_ms_per_block"] for r in selected]),
                "host_wall_ms_per_block": _stats([r["host_wall_ms_per_block"] for r in selected]),
                "first_half_cuda_median_ms_per_operation": statistics.median(
                    [r["cuda_event_ms_per_operation"] for r in selected[:len(selected) // 2]]),
                "second_half_cuda_median_ms_per_operation": statistics.median(
                    [r["cuda_event_ms_per_operation"] for r in selected[len(selected) // 2:]]),
            }
    return result


def _load_kernel(root: Path):
    path = root / "scripts" / "triton_fc1.py"
    spec = importlib.util.spec_from_file_location("fc1_timing_diagnostic_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _telemetry(torch, device: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return {
            "sm_clock_khz": int(torch.cuda.clock_rate(device)),
            "temperature_c": int(torch.cuda.temperature(device)),
        }, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _correctness(torch, output, reference) -> dict[str, Any]:
    shape_match = tuple(output.shape) == tuple(reference.shape)
    output_nonfinite = int((~torch.isfinite(output)).sum().item()) if shape_match else None
    reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
    passed = bool(shape_match and output_nonfinite == 0 and reference_nonfinite == 0 and
                  torch.allclose(output, reference, **CORRECTNESS))
    difference = (output.float() - reference.float()).abs() if shape_match else None
    return {
        "passed": passed, "shape_match": shape_match,
        "reference_nonfinite_count": reference_nonfinite,
        "output_nonfinite_count": output_nonfinite,
        "max_absolute_error": float(difference.max().item()) if difference is not None else None,
        "mean_absolute_error": float(difference.mean().item()) if difference is not None else None,
        **CORRECTNESS,
    }


def _measure(torch, submit: Callable[[], Any], operation_count: int) -> dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    submit()
    end.record()
    submission_end = time.perf_counter_ns()
    end.synchronize()
    wall_end = time.perf_counter_ns()
    return {
        "cuda_event_elapsed_ms_per_block": float(start.elapsed_time(end)),
        "cuda_event_ms_per_operation": float(start.elapsed_time(end)) / operation_count,
        "host_submission_ms_per_block": (submission_end - wall_start) / 1e6,
        "host_wall_ms_per_block": (wall_end - wall_start) / 1e6,
    }


def run(output_root: Path, telemetry_enabled: bool = True) -> Path:
    root = Path(__file__).resolve().parents[1]
    try:
        import torch
        import triton
    except ImportError as error:
        raise RuntimeError("CUDA-enabled PyTorch and Triton are required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        "purpose": "separate measurement-method diagnostic; not a session experiment",
        "workload": WORKLOAD, "triton_parameters": TRITON_PARAMETERS,
        "correctness": CORRECTNESS, "input_seed": INPUT_SEED,
        "schedule_seed": SCHEDULE_SEED,
        "ordinary_warmup_operations_per_implementation_mode": WARMUP_OPERATIONS,
        "graph_warmup_replays_per_implementation": WARMUP_OPERATIONS,
        "operations_per_measured_block": OPERATIONS_PER_BLOCK,
        "blocks_per_implementation_mode": BLOCKS, "modes": list(MODES),
        "graph_capture_operations_per_replay": OPERATIONS_PER_BLOCK,
        "timing_boundaries": {
            "cuda_event": "from an event before block submission to an event after all block work",
            "host_submission": "host wall time from before start-event submission through end-event submission; ends before synchronization",
            "host_wall": "host wall time from before start-event submission through end-event completion",
        },
        "interpretation_warning": (
            "Ordinary CUDA-event intervals can include host-induced stream idle gaps. CUDA Graph replay "
            "changes submission behavior. These measurements are not all pure kernel execution time."
        ),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    kernel = _load_kernel(root)
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    provenance = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(root, "status", "--porcelain"),
        "python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda, "device_index": device_index, "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}", "total_memory_bytes": props.total_memory,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed_all(INPUT_SEED)
    a = torch.randn((WORKLOAD["m"], WORKLOAD["k"]), device="cuda", dtype=torch.float16)
    b = torch.randn((WORKLOAD["k"], WORKLOAD["n"]), device="cuda", dtype=torch.float16)
    cfg = kernel.TritonGemmConfig(**TRITON_PARAMETERS)
    baseline_output = torch.empty((WORKLOAD["m"], WORKLOAD["n"]), device="cuda", dtype=torch.float16)
    candidate_output = torch.empty_like(baseline_output)

    def candidate_into():
        grid = (triton.cdiv(WORKLOAD["m"], cfg.block_m) * triton.cdiv(WORKLOAD["n"], cfg.block_n),)
        kernel.fc1_kernel[grid](
            a, b, candidate_output, WORKLOAD["m"], WORKLOAD["n"], WORKLOAD["k"],
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            candidate_output.stride(0), candidate_output.stride(1),
            BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
            GROUP_SIZE_M=cfg.group_size_m, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        return candidate_output

    functions = {
        "allocating": {"baseline": lambda: torch.matmul(a, b),
                       "candidate": lambda: kernel.triton_gemm(a, b, cfg)},
        "preallocated": {"baseline": lambda: torch.matmul(a, b, out=baseline_output),
                         "candidate": candidate_into},
    }
    # Compile every ordinary path, then perform exactly 100 warmup operations per path.
    for mode in ("allocating", "preallocated"):
        for implementation in IMPLEMENTATIONS:
            functions[mode][implementation]()
            torch.cuda.synchronize()
            for _ in range(WARMUP_OPERATIONS):
                functions[mode][implementation]()
            torch.cuda.synchronize()

    graphs: dict[str, Any] = {}
    graph_errors: dict[str, str] = {}
    for implementation in IMPLEMENTATIONS:
        try:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                for _ in range(OPERATIONS_PER_BLOCK):
                    functions["preallocated"][implementation]()
            graphs[implementation] = graph
            for _ in range(WARMUP_OPERATIONS):
                graph.replay()
            torch.cuda.synchronize()
        except Exception as error:
            graph_errors[implementation] = f"{type(error).__name__}: {error}"
    functions["cuda_graph"] = {name: graph.replay for name, graph in graphs.items()}

    reference = torch.matmul(a, b)
    torch.cuda.synchronize()
    correctness: dict[str, Any] = {}
    outputs = {"baseline": baseline_output, "candidate": candidate_output}
    for mode in MODES:
        correctness[mode] = {}
        for implementation in IMPLEMENTATIONS:
            if implementation not in functions[mode]:
                correctness[mode][implementation] = {"supported": False, "error": graph_errors[implementation]}
                continue
            returned = functions[mode][implementation]()
            torch.cuda.synchronize()
            output = outputs[implementation] if mode == "cuda_graph" else returned
            correctness[mode][implementation] = {"supported": True, **_correctness(torch, output, reference)}
    (destination / "correctness.json").write_text(json.dumps(correctness, indent=2) + "\n", encoding="utf-8")
    failures = [(m, i) for m in MODES for i in IMPLEMENTATIONS
                if correctness[m][i].get("supported") and not correctness[m][i]["passed"]]
    if failures:
        raise RuntimeError(f"correctness failed for {failures}; artifacts retained at {destination}")

    available = {mode: [name for name in IMPLEMENTATIONS if name in functions[mode]] for mode in MODES}
    schedule = build_schedule(available=available)
    rows: list[dict[str, Any]] = []
    telemetry_error = None
    telemetry_available = telemetry_enabled

    def submit_repeated(function: Callable[[], Any]) -> None:
        for _ in range(OPERATIONS_PER_BLOCK):
            function()

    for entry in schedule:
        mode, implementation = entry["mode"], entry["implementation"]
        operation_count = OPERATIONS_PER_BLOCK
        if mode == "cuda_graph":
            submit = functions[mode][implementation]
        else:
            fn = functions[mode][implementation]
            submit = lambda fn=fn: submit_repeated(fn)
        measurement = _measure(torch, submit, operation_count)
        telemetry, error = _telemetry(torch, device_index) if telemetry_available else (None, None)
        telemetry_error = telemetry_error or error
        telemetry_available = telemetry_available and error is None
        rows.append({**entry, "operation_count": operation_count,
                     "submission_calls": 1 if mode == "cuda_graph" else OPERATIONS_PER_BLOCK,
                     **measurement, "telemetry_after": telemetry})

    summary = summarize_measurements(rows)
    result = {
        "schema_version": 1, "status": "success", "run_id": run_id,
        "run_directory": str(destination.resolve()), "config": config, "provenance": provenance,
        "correctness": correctness, "graph_capture_errors": graph_errors,
        "telemetry": {"requested": telemetry_enabled, "available": telemetry_available,
                      "error": telemetry_error},
        "actual_execution_order": schedule, "summary": summary,
    }
    (destination / "raw_measurements.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fields = ["sequence", "block", "mode", "mode_position", "implementation",
              "implementation_position", "operation_count", "submission_calls",
              "cuda_event_elapsed_ms_per_block", "cuda_event_ms_per_operation",
              "host_submission_ms_per_block", "host_wall_ms_per_block",
              "sm_clock_khz", "temperature_c"]
    with (destination / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat["sm_clock_khz"] = (row["telemetry_after"] or {}).get("sm_clock_khz")
            flat["temperature_c"] = (row["telemetry_after"] or {}).get("temperature_c")
            writer.writerow(flat)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# FC1 timing-method diagnostic", "", config["interpretation_warning"], "",
             f"Graph replay contains {OPERATIONS_PER_BLOCK} captured operations and is submitted once per measured block.", "",
             "| Mode | Implementation | CUDA median ms/op | CUDA CV | CUDA relative range | Host submit median ms/block | Host wall median ms/block |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for implementation, values in summary[mode].items():
            cuda = values["cuda_event_ms_per_operation"]
            lines.append(f"| {mode} | {implementation} | {cuda['median']:.6f} | {cuda['coefficient_of_variation']:.4f} | {cuda['relative_range_over_median']:.4f} | {values['host_submission_ms_per_block']['median']:.3f} | {values['host_wall_ms_per_block']['median']:.3f} |")
    (destination / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/fc1-timing-diagnostics"))
    parser.add_argument("--no-telemetry", action="store_true")
    args = parser.parse_args()
    try:
        destination = run(args.output_root, telemetry_enabled=not args.no_telemetry)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
