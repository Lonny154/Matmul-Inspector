#!/usr/bin/env python3
"""Correctness-gated RTX GPU study of a few selected HA-Schedule-2 points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile

from matmul_inspector.tensor_game_schedule_gpu_study import (
    build_lowering_map, join_static_measurements, load_selected_points, summarize_timings,
)


REPO = Path(__file__).resolve().parents[1]
HA2 = REPO / "artifacts/tensor-game/hardware-aware-schedule-search-v1"
FROZEN = REPO / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"
RTOL = 5e-3
ATOL = 5e-2


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gpu_profile(torch, execute) -> list[dict]:
    from torch.profiler import DeviceType, ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        execute()
        torch.cuda.synchronize()
    return [{"kernel_name": event.name, "profiler_device_duration_us": float(event.device_time_total)}
            for event in recorded.events() if event.device_type == DeviceType.CUDA]


def _compiler_capture(compiled, label: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    asm = compiled.asm
    result = {"label": label, "kernel_name": compiled.name,
              "registers_per_thread": int(compiled.n_regs),
              "static_shared_memory_bytes": int(compiled.metadata.shared),
              "artifacts": {}}
    for kind in ("ttir", "ttgir", "ptx"):
        content = asm.get(kind)
        if content is not None:
            data = content.encode() if isinstance(content, str) else bytes(content)
            path = directory / f"{label}.{kind}"
            path.write_bytes(data)
            result["artifacts"][kind] = {"path": str(path.relative_to(directory.parent)),
                                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    cubin = asm.get("cubin")
    if cubin is not None:
        data = bytes(cubin)
        result["cubin_sha256"] = hashlib.sha256(data).hexdigest()
        result["cubin_bytes"] = len(data)
        with tempfile.NamedTemporaryFile(suffix=".cubin") as temporary:
            temporary.write(data)
            temporary.flush()
            sass = _command("nvdisasm", "--print-code", temporary.name)
        if sass:
            path = directory / f"{label}.sass"
            path.write_text(sass + "\n")
            result["artifacts"]["sass"] = {"path": str(path.relative_to(directory.parent)),
                                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "bytes": path.stat().st_size}
        else:
            result["sass_status"] = "UNAVAILABLE_OR_DISASSEMBLY_FAILED"
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = ("matrix_size", "round", "iteration", "order_position", "candidate", "algorithm",
               "schedule_hash", "elapsed_ms")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pair_summary(summary: list[dict]) -> list[dict]:
    by_key = {(row["matrix_size"], row["candidate"]): row for row in summary}
    result = []
    for size in sorted({row["matrix_size"] for row in summary}):
        stored = by_key.get((size, "rank8_stored"))
        recompute = by_key.get((size, "rank8_recomputed"))
        if stored and recompute:
            ratios = [a / b for a, b in zip(stored["round_medians_ms"], recompute["round_medians_ms"])]
            rng = random.Random(1107 + size)
            bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
            result.append({"matrix_size": size,
                           "stored_median_ms": stored["median_ms"],
                           "recomputed_median_ms": recompute["median_ms"],
                           "recomputed_minus_stored_ms": recompute["median_ms"] - stored["median_ms"],
                           "paired_stored_over_recomputed_round_ratios": ratios,
                           "median_paired_ratio": statistics.median(ratios),
                           "paired_ratio_bootstrap_95pct": [bootstrap[250], bootstrap[9750]],
                           "ratio_min": min(ratios), "ratio_max": max(ratios)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--correctness-seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--ha2-artifact", type=Path, default=HA2)
    parser.add_argument("--frozen-artifact", type=Path, default=FROZEN)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    if args.smoke:
        args.sizes = [128]
        args.warmups = 2
        args.iterations = 5
        args.rounds = 2
    if any(size < 2 or size % 2 for size in args.sizes):
        parser.error("sizes must be positive even matrix dimensions")
    if len(args.correctness_seeds) < 5:
        parser.error("at least five correctness seeds are required")
    if min(args.warmups, args.iterations, args.rounds) < 1:
        parser.error("warmups, iterations, and rounds must be positive")
    selected = load_selected_points(args.ha2_artifact, args.frozen_artifact)
    if any(size not in item["static_per_size"] for item in selected for size in args.sizes):
        parser.error("all sizes must be present in the retained HA-Schedule-2 study")
    # Import GPU dependencies only after CPU-safe selection has succeeded.
    try:
        import torch
        import triton
    except ImportError as exc:
        parser.error(f"PyTorch and Triton are required: {exc}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no GPU timing artifacts were generated")
    from matmul_inspector.tensor_game_triton_lowering import create_lowering, reference_lowering
    from matmul_inspector.tensor_game_schedule_gpu_runtime import (
        allocate_workspace, execute_selected, expected_kernel_names,
    )
    from matmul_inspector.tensor_game import RankOneAction

    frozen_record = json.loads(args.frozen_artifact.read_text().splitlines()[0])
    frozen_actions = tuple(RankOneAction(**action) for action in frozen_record["actions"])
    lowerings = {"conventional": reference_lowering("conventional"),
                 "strassen": reference_lowering("strassen"),
                 "project-rank8": create_lowering("project-rank8", frozen_actions)}
    included = [item for item in selected if item["gpu_mode"] is not None]
    item_by_name = {item["name"]: item for item in included}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    hardware = {"gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": torch.cuda.get_device_capability(0),
                "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "nvidia_smi_query": _command("nvidia-smi", "--query-gpu=driver_version,clocks.gr,clocks.mem,power.limit",
                                              "--format=csv,noheader,nounits")}
    software = {"python": platform.python_version(), "torch": torch.__version__,
                "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                "driver": _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                "environment": {name: os.environ.get(name) for name in
                                ("CUDA_VISIBLE_DEVICES", "TRITON_CACHE_DIR", "PYTORCH_CUDA_ALLOC_CONF")}}
    try:
        commit = _command("git", "rev-parse", "HEAD") or "unknown"
        correctness: list[dict] = []
        raw: list[dict] = []
        launch_records: list[dict] = []
        telemetry: list[dict] = []
        compiler: list[dict] = []
        workspaces: dict[tuple[int, str], dict] = {}
        benchmark_inputs: dict[int, tuple] = {}
        passed: dict[tuple[int, str], bool] = {}
        args.output.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="ha-schedule3-triton-") as cache:
            os.environ["TRITON_CACHE_DIR"] = cache
            for size in args.sizes:
                for item in included:
                    workspaces[(size, item["name"])] = allocate_workspace(
                        lowerings[item["algorithm"]], size, item["gpu_mode"])
                for seed in args.correctness_seeds:
                    torch.manual_seed(seed)
                    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
                    right = torch.randn((size, size), device="cuda", dtype=torch.float16)
                    reference = left @ right
                    outputs = {}
                    for item in included:
                        name = item["name"]
                        try:
                            output = execute_selected(lowerings[item["algorithm"]], left, right,
                                                      workspaces[(size, name)], item["gpu_mode"]).clone()
                            torch.cuda.synchronize()
                            difference = (output.float() - reference.float()).abs()
                            success = bool(torch.allclose(output, reference, rtol=RTOL, atol=ATOL))
                            outputs[name] = output
                            row = {"matrix_size": size, "seed": seed, "candidate": name,
                                   "schedule_hash": item["static_per_size"][size]["schedule_hash"],
                                   "max_absolute_error": float(difference.max()),
                                   "mean_absolute_error": float(difference.mean()),
                                   "rtol": RTOL, "atol": ATOL, "passed": success}
                        except Exception as exc:
                            success = False
                            row = {"matrix_size": size, "seed": seed, "candidate": name,
                                   "schedule_hash": item["static_per_size"][size]["schedule_hash"],
                                   "rtol": RTOL, "atol": ATOL, "passed": False,
                                   "error": f"{type(exc).__name__}: {exc}"}
                        correctness.append(row)
                        passed[(size, name)] = passed.get((size, name), True) and success
                    if "rank8_stored" in outputs and "rank8_recomputed" in outputs:
                        pair_difference = (outputs["rank8_stored"].float() - outputs["rank8_recomputed"].float()).abs()
                        pair_passed = bool(torch.allclose(outputs["rank8_stored"],
                                                          outputs["rank8_recomputed"],
                                                          rtol=RTOL, atol=ATOL))
                        correctness.append({"matrix_size": size, "seed": seed, "candidate": "rank8_pair_agreement",
                                            "max_absolute_error": float(pair_difference.max()),
                                            "mean_absolute_error": float(pair_difference.mean()),
                                            "rtol": RTOL, "atol": ATOL,
                                            "passed": pair_passed})
                        if not pair_passed:
                            passed[(size, "rank8_stored")] = False
                            passed[(size, "rank8_recomputed")] = False
                torch.manual_seed(args.seed + size)
                benchmark_inputs[size] = (torch.randn((size, size), device="cuda", dtype=torch.float16),
                                          torch.randn((size, size), device="cuda", dtype=torch.float16))
            for size in args.sizes:
                left, right = benchmark_inputs[size]
                functions = {item["name"]: (lambda item=item: execute_selected(
                    lowerings[item["algorithm"]], left, right, workspaces[(size, item["name"])], item["gpu_mode"]))
                             for item in included if passed[(size, item["name"])]}
                functions["pytorch_cublas"] = lambda: left @ right
                for function in functions.values():
                    for _ in range(args.warmups):
                        function()
                torch.cuda.synchronize()
                for round_index in range(args.rounds):
                    pair = [name for name in ("rank8_stored", "rank8_recomputed") if name in functions]
                    if round_index % 2:
                        pair.reverse()
                    blocks = ([pair] if pair else []) + [[name] for name in functions if name not in pair]
                    random.Random(args.seed + size * 100 + round_index).shuffle(blocks)
                    order = [name for block in blocks for name in block]
                    for iteration in range(args.iterations):
                        for position, name in enumerate(order):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            functions[name]()
                            end.record()
                            end.synchronize()
                            raw.append({"matrix_size": size, "round": round_index,
                                        "iteration": iteration, "order_position": position,
                                        "candidate": name,
                                        "algorithm": item_by_name[name]["algorithm"]
                                        if name in item_by_name else "pytorch_cublas",
                                        "schedule_hash": item_by_name[name]["static_per_size"][size]["schedule_hash"]
                                        if name in item_by_name else "EXTERNAL_CUBLAS",
                                        "elapsed_ms": start.elapsed_time(end)})
                    telemetry.append({"matrix_size": size, "round": round_index,
                                      "nvidia_smi": _command(
                                          "nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu",
                                          "--format=csv,noheader,nounits")})
                for name, function in functions.items():
                    events = _gpu_profile(torch, function)
                    expected_names = (expected_kernel_names(len(lowerings[item_by_name[name]["algorithm"]].actions),
                                                             item_by_name[name]["gpu_mode"])
                                      if name in item_by_name else [])
                    launch_records.append({"matrix_size": size, "candidate": name,
                                           "actual_kernel_launches": len(events),
                                           "expected_source_launches": len(expected_names) if expected_names else None,
                                           "source_kernel_names_match":
                                           [event["kernel_name"] for event in events] == expected_names
                                           if expected_names else None,
                                           "per_launch": events})
                if size == max(args.sizes):
                    for name in ("rank8_stored", "rank8_recomputed"):
                        if name not in functions:
                            continue
                        captured = {}
                        item = item_by_name[name]
                        execute_selected(lowerings[item["algorithm"]], left, right,
                                         workspaces[(size, name)], item["gpu_mode"], capture=captured)
                        torch.cuda.synchronize()
                        for kind, compiled in captured.items():
                            compiler.append(_compiler_capture(compiled, f"{name}_{kind}_{size}",
                                                              args.output / "compiler"))
        summary = summarize_timings(raw, {row["candidate"] for row in raw})
        paired = _pair_summary(summary)
        static_vs_measured = join_static_measurements(selected, args.sizes, summary, launch_records)
        config = {"sizes": args.sizes, "correctness_seeds": args.correctness_seeds,
                  "rtol": RTOL, "atol": ATOL, "warmups": args.warmups,
                  "iterations_per_round": args.iterations, "rounds": args.rounds,
                  "seed": args.seed, "input_dtype": "fp16", "product_accumulation_dtype": "fp32",
                  "output_dtype": "fp16", "timing": "CUDA events, complete device operation",
                  "round_replicate": "median of per-operation CUDA events",
                  "ha2_artifact": str(args.ha2_artifact), "frozen_artifact": str(args.frozen_artifact)}
        provenance = {"source_commit": commit, "ha2_hash_manifest_sha256": hashlib.sha256(
            (args.ha2_artifact / "hash_manifest.json").read_bytes()).hexdigest(),
            "frozen_artifact_sha256": hashlib.sha256(args.frozen_artifact.read_bytes()).hexdigest(),
            "gpu_used": True}
        aggregate = {"correctness_rows": len(correctness),
                     "correctness_passed": sum(row["passed"] for row in correctness),
                     "correctness_failed": sum(not row["passed"] for row in correctness),
                     "timed_candidates": sorted({row["candidate"] for row in raw}),
                     "timing_samples": len(raw), "paired_recompute": paired,
                     "implementability_excluded": [item["name"] for item in selected if item["gpu_mode"] is None],
                     "decision": "PENDING_ANALYSIS"}
        def write(name: str, payload: object) -> None:
            (args.output / name).write_bytes(encoded(payload))

        write("config.json", config)
        write("provenance.json", provenance)
        write("hardware.json", hardware)
        write("software_versions.json", software)
        write("selected_schedules.json", [{key: value for key, value in item.items() if key != "static_per_size"}
                                          | {"static_per_size": {size: {"schedule_hash": data["schedule_hash"],
                                                                  "metrics": data["metrics"]}
                                                                 for size, data in item["static_per_size"].items()}}
                                          for item in selected])
        write("lowering_map.json", build_lowering_map(selected))
        (args.output / "correctness.jsonl").write_bytes(b"".join(encoded(row) for row in correctness))
        _write_csv(args.output / "raw_timings.csv", raw)
        write("timing_summary.json", summary)
        write("launch_accounting.json", launch_records)
        write("round_telemetry.json", telemetry)
        write("compiler_artifacts_manifest.json", compiler)
        write("static_vs_measured.json", static_vs_measured)
        write("aggregate_summary.json", aggregate)
        (args.output / "study_notes.md").write_text(
            "# HA-Schedule-3 GPU study\n\nCUDA-event times are complete device-operation times, not host invocation times. "
            "Modeled HBM bytes and ON_CHIP capacity are abstract. See docs/hardware_aware_schedule_gpu_study.md.\n")
        hashes = {path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(args.output.rglob("*")) if path.is_file()}
        write("hash_manifest.json", hashes)
        print(f"wrote {len(raw)} raw timings and {len(correctness)} correctness rows to {args.output}; "
              f"failures: {aggregate['correctness_failed']}")
        return 0 if aggregate["correctness_failed"] == 0 else 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    raise SystemExit(main())
