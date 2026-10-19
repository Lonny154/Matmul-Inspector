#!/usr/bin/env python3
"""Bounded cross-clock probe for existing CUDA kernels and graph replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROUNDS = 8
MODES = {"eager_1": (1, 1), "eager_100": (100, 100), "graph_100": (100, 1)}
WORKLOADS = ("add", "fc1")
SLOTS = ("A", "B")  # Identical execution in both slots: a null paired control.


def stats(values: list[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {"count": len(values), "min": min(values), "median": median,
            "max": max(values), "mad": statistics.median(abs(x - median) for x in values)}


def summarize(rows: list[dict]) -> dict:
    """Recompute all descriptive results from retained total block durations."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["workload"], row["mode"]].append(row)
    if set(groups) != {(w, m) for w in WORKLOADS for m in MODES}:
        raise ValueError("missing workload or mode")
    result = {}
    for (workload, mode), selected in sorted(groups.items()):
        if len(selected) != 2 * ROUNDS or {r["round"] for r in selected} != set(range(1, ROUNDS + 1)):
            raise ValueError("incomplete repeated blocks")
        expected_count, expected_submissions = MODES[mode]
        pairs: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in selected:
            if (row["slot"] not in SLOTS or row["slot"] in pairs[row["round"]]
                    or row["operation_count"] != expected_count
                    or row["submission_calls"] != expected_submissions):
                raise ValueError("invalid block identity or count")
            pairs[row["round"]][row["slot"]] = row
        if any(set(pair) != set(SLOTS) for pair in pairs.values()):
            raise ValueError("incomplete A/A pair")
        entry = {field: stats([r[field] for r in selected]) for field in
                 ("cuda_event_ms", "perf_counter_ms", "monotonic_raw_ms",
                  "event_minus_perf_ms", "event_minus_raw_ms", "raw_over_perf_ratio")}
        entry["paired_A_minus_B_ms"] = {
            clock: stats([pairs[i]["A"][clock] - pairs[i]["B"][clock]
                          for i in range(1, ROUNDS + 1)])
            for clock in ("cuda_event_ms", "perf_counter_ms", "monotonic_raw_ms")}
        entry["first_half_event_ms"] = statistics.median(
            r["cuda_event_ms"] for r in selected if r["round"] <= ROUNDS // 2)
        entry["second_half_event_ms"] = statistics.median(
            r["cuda_event_ms"] for r in selected if r["round"] > ROUNDS // 2)
        result[f"{workload}/{mode}"] = entry
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str) -> dict:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def gpu_snapshot() -> dict:
    return command("nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
                   "--format=csv,noheader,nounits")


def measure(torch, stream, operation, *, workload: str, mode: str, round_id: int,
            slot: str, graph=None) -> dict:
    operation_count, submissions = MODES[mode]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    perf_start = time.perf_counter_ns()
    raw_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    start.record(stream)
    with torch.cuda.stream(stream):
        if graph is None:
            for _ in range(operation_count):
                operation()
        else:
            graph.replay()
    end.record(stream)
    perf_submission_end = time.perf_counter_ns()
    end.synchronize()
    raw_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    perf_end = time.perf_counter_ns()
    event_ms = float(start.elapsed_time(end))
    perf_ms = (perf_end - perf_start) / 1e6
    raw_ms = (raw_end - raw_start) / 1e6
    return {"workload": workload, "mode": mode, "round": round_id, "slot": slot,
            "operation_count": operation_count, "submission_calls": submissions,
            "cuda_event_ms": event_ms, "perf_counter_ms": perf_ms,
            "monotonic_raw_ms": raw_ms,
            "host_submission_ms": (perf_submission_end - perf_start) / 1e6,
            "event_minus_perf_ms": event_ms - perf_ms,
            "event_minus_raw_ms": event_ms - raw_ms,
            "raw_over_perf_ratio": raw_ms / perf_ms,
            "perf_start_ns": perf_start, "perf_submission_end_ns": perf_submission_end,
            "perf_end_ns": perf_end, "raw_start_ns": raw_start, "raw_end_ns": raw_end}


def run(output: Path) -> None:
    if output.exists():
        raise ValueError("output already exists")
    processes = command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits")
    if processes["returncode"] or processes["stdout"]:
        raise RuntimeError("GPU process check failed or another compute process is active")
    import torch
    import triton
    from matmul_inspector import triton_fc1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not hasattr(time, "CLOCK_MONOTONIC_RAW"):
        raise RuntimeError("CLOCK_MONOTONIC_RAW unavailable")
    root = Path(__file__).resolve().parents[1]
    revision = command("git", "rev-parse", "HEAD")["stdout"]
    output.mkdir(parents=True)
    write_json(output / "config.json", {
        "workloads": list(WORKLOADS), "modes": MODES, "rounds": ROUNDS,
        "slots": "identical A/A null control with AB/BA order",
        "warmup_calls_per_workload": 30,
        "graph_capture_operations": 100,
        "timing_order": ["device synchronize", "perf start", "RAW start",
                         "event start on explicit stream", "submit operations on same stream",
                         "event end on explicit stream", "perf submission end",
                         "end event synchronize", "RAW end", "perf end"],
        "normalization": "all raw durations are total ms per block; divide by operation_count only after inspection",
    })
    write_json(output / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision,
        "script_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "kernel_release": platform.release(), "pytorch": torch.__version__,
        "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "driver_and_clocks_before": gpu_snapshot(),
        "competing_compute_processes_before": processes,
        "perf_clock": time.get_clock_info("perf_counter").implementation,
    })
    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    x, y = (torch.randn(4096, device="cuda") for _ in range(2))
    z = torch.empty_like(x)
    a = torch.randn((512, 768), device="cuda", dtype=torch.float16)
    b = torch.randn((768, 3072), device="cuda", dtype=torch.float16)
    c = torch.empty((512, 3072), device="cuda", dtype=torch.float16)
    cfg = triton_fc1.TritonGemmConfig(block_m=64, block_n=128, block_k=32,
                                      num_warps=4, num_stages=3, group_size_m=1)
    grid = (triton.cdiv(512, 64) * triton.cdiv(3072, 128),)

    def add() -> None:
        torch.add(x, y, out=z)

    def fc1() -> None:
        triton_fc1.fc1_kernel[grid](a, b, c, 512, 3072, 768,
                                   a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                   c.stride(0), c.stride(1), BLOCK_M=cfg.block_m,
                                   BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                                   GROUP_SIZE_M=cfg.group_size_m,
                                   num_warps=cfg.num_warps, num_stages=cfg.num_stages)

    operations = {"add": add, "fc1": fc1}
    graphs = {}
    reference = torch.matmul(a, b)
    with torch.cuda.stream(stream):
        for name, operation in operations.items():
            for _ in range(30):
                operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(100):
                    operation()
            graphs[name] = graph
            graph.replay()
            stream.synchronize()
    correctness = {
        "add": {"passed": bool(torch.allclose(z, x + y)), "elements": 4096},
        "fc1": {"passed": bool(torch.allclose(c, reference, atol=0.01, rtol=0.01)),
                "max_absolute_error": float((c.float() - reference.float()).abs().max().item()),
                "atol": 0.01, "rtol": 0.01}}
    write_json(output / "correctness.json", correctness)
    if not all(item["passed"] for item in correctness.values()):
        raise RuntimeError("correctness failed; partial evidence retained")
    rows, telemetry = [], []
    for round_id in range(1, ROUNDS + 1):
        workloads = WORKLOADS if round_id % 2 else reversed(WORKLOADS)
        mode_names = list(MODES)
        mode_names = mode_names[(round_id - 1) % 3:] + mode_names[:(round_id - 1) % 3]
        for workload in workloads:
            for mode in mode_names:
                slots = SLOTS if round_id % 2 else reversed(SLOTS)
                for slot in slots:
                    rows.append(measure(torch, stream, operations[workload], workload=workload,
                                        mode=mode, round_id=round_id, slot=slot,
                                        graph=graphs[workload] if mode == "graph_100" else None))
        telemetry.append({"round": round_id, "gpu": gpu_snapshot()})
    (output / "raw_blocks.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    write_json(output / "telemetry.json", telemetry)
    write_json(output / "summary.json", summarize(rows))
    manifest = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "hash_manifest.json", manifest)


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    actual = {path.name for path in output.iterdir() if path.is_file() and path.name != "hash_manifest.json"}
    if set(manifest) != actual or any(sha256(output / name) != digest for name, digest in manifest.items()):
        raise ValueError("file manifest mismatch")
    rows = [json.loads(line) for line in (output / "raw_blocks.jsonl").read_text().splitlines()]
    if summarize(rows) != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary mismatch")
    return {"status": "PASS", "blocks": len(rows), "files": len(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error("choose exactly one of --output or --verify")
    try:
        if args.verify:
            print(json.dumps(verify(args.verify)))
        else:
            run(args.output)
            print(json.dumps(verify(args.output)))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"timing probe failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
