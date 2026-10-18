#!/usr/bin/env python3
"""Bounded recompute-vs-store mechanism study; A/B are fresh GPU processes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

from matmul_inspector.tensor_game_schedule_gpu_study import load_selected_points, verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_mechanism_study import (
    COMMON, DUMMY, RECOMPUTED, STORED, classify_kernel_roles,
    classify_mechanism, compare_sessions, compiler_diff_summary,
    join_static_measured, paired_summary, summarize_raw, validate_topology,
)


REPO = Path(__file__).resolve().parents[1]
HA2 = REPO / "artifacts/tensor-game/hardware-aware-schedule-search-v1"
HA3 = REPO / "artifacts/tensor-game/hardware-aware-schedule-gpu-v1"
FROZEN = REPO / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"
RTOL = 5e-3
ATOL = 5e-2
PROFILE_METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sectors.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ("session", "matrix_size", "phase", "variant", "round", "iteration",
              "order_position", "elapsed_ms")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [{**row, "matrix_size": int(row["matrix_size"]), "round": int(row["round"]),
                 "iteration": int(row["iteration"]), "order_position": int(row["order_position"]),
                 "elapsed_ms": float(row["elapsed_ms"])} for row in csv.DictReader(handle)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in rows))


def command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def frozen_lowering():
    from matmul_inspector.tensor_game import RankOneAction
    from matmul_inspector.tensor_game_triton_lowering import create_lowering

    row = json.loads(FROZEN.read_text().splitlines()[0])
    return create_lowering("project-rank8", tuple(RankOneAction(**item) for item in row["actions"]))


def time_one(torch, function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def benchmark_pair(torch, *, session: str, size: int, phase: str,
                   functions: dict, warmups: int, iterations: int,
                   rounds: int, seed: int) -> list[dict]:
    for function in functions.values():
        for _ in range(warmups):
            function()
    torch.cuda.synchronize()
    rows = []
    for round_index in range(rounds):
        order = [STORED, RECOMPUTED] if round_index % 2 == 0 else [RECOMPUTED, STORED]
        for iteration in range(iterations):
            for position, name in enumerate(order):
                rows.append({"session": session, "matrix_size": size, "phase": phase,
                             "variant": name, "round": round_index, "iteration": iteration,
                             "order_position": position, "elapsed_ms": time_one(torch, functions[name])})
    return rows


def benchmark_components(torch, *, session: str, size: int, phase: str,
                         functions: dict, warmups: int, iterations: int,
                         rounds: int, seed: int) -> list[dict]:
    for function in functions.values():
        for _ in range(warmups):
            function()
    torch.cuda.synchronize()
    rows = []
    for round_index in range(rounds):
        order = list(functions)
        random.Random(seed + size * 31 + round_index).shuffle(order)
        for iteration in range(iterations):
            for position, name in enumerate(order):
                rows.append({"session": session, "matrix_size": size, "phase": phase,
                             "variant": name, "round": round_index, "iteration": iteration,
                             "order_position": position, "elapsed_ms": time_one(torch, functions[name])})
    return rows


def benchmark_graph_modes(torch, *, session: str, size: int, eager: dict, graphs: dict,
                          warmups: int, iterations: int, rounds: int, seed: int) -> list[dict]:
    """Pair both paths inside each round, then alternate eager/graph block order."""
    groups = {"graph_eager": eager, "graph_replay": graphs}
    for functions in groups.values():
        for function in functions.values():
            for _ in range(warmups):
                function()
    torch.cuda.synchronize()
    rows = []
    for round_index in range(rounds):
        group_order = list(groups)
        random.Random(seed + size * 47 + round_index).shuffle(group_order)
        pair_order = [STORED, RECOMPUTED] if round_index % 2 == 0 else [RECOMPUTED, STORED]
        for iteration in range(iterations):
            position = 0
            for group in group_order:
                for mode in pair_order:
                    rows.append({"session": session, "matrix_size": size, "phase": group,
                                 "variant": mode, "round": round_index, "iteration": iteration,
                                 "order_position": position,
                                 "elapsed_ms": time_one(torch, groups[group][mode])})
                    position += 1
    return rows


def gpu_kernel_names(torch, function) -> list[str]:
    from torch.profiler import DeviceType, ProfilerActivity, profile

    function()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        function()
        torch.cuda.synchronize()
    return [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]


def correctness_row(torch, session: str, size: int, seed: int,
                    candidate: str, output, reference) -> dict:
    difference = (output.float() - reference.float()).abs()
    return {"session": session, "matrix_size": size, "seed": seed,
            "candidate": candidate, "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()), "rtol": RTOL,
            "atol": ATOL, "passed": bool(torch.allclose(output, reference,
                                                        rtol=RTOL, atol=ATOL))}


def worker(args) -> int:
    import torch
    import triton
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
    from matmul_inspector.tensor_game_schedule_mechanism_runtime import (
        execute_staged, run_common_gemms, run_consumer_product, run_consumers,
        run_dummy_launch, run_materialization, run_reconstruction,
    )
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU worker cannot run")
    session = args.worker_session
    lowering = frozen_lowering()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        start_time = now()
        correctness = []
        complete = []
        stages = []
        graphs_raw = []
        dummy_raw = []
        topology = {}
        common_hashes = {}
        compiler_entries = []
        graph_status = {}
        dummy_status = {}
        telemetry = []
        hardware = {"gpu_name": torch.cuda.get_device_name(0),
                    "compute_capability": torch.cuda.get_device_capability(0),
                    "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "nvidia_smi": command("nvidia-smi", "--query-gpu=driver_version,power.limit,clocks.gr,clocks.mem",
                                           "--format=csv,noheader,nounits")}
        software = {"python": platform.python_version(), "torch": torch.__version__,
                    "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                    "environment": {key: os.environ.get(key) for key in
                                    ("CUDA_VISIBLE_DEVICES", "TRITON_CACHE_DIR", "PYTORCH_CUDA_ALLOC_CONF")}}
        for size in args.sizes:
            workspaces = {
                STORED: allocate_workspace(lowering, size, "stored_form"),
                RECOMPUTED: allocate_workspace(lowering, size, "recomputed_form"),
            }
            # Gate all sizes before the first timing sample in this process.
            for seed in args.correctness_seeds:
                torch.manual_seed(seed)
                left = torch.randn((size, size), device="cuda", dtype=torch.float16)
                right = torch.randn((size, size), device="cuda", dtype=torch.float16)
                reference = left @ right
                baseline_products = {}
                outputs = {}
                for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
                    outputs[mode] = execute_selected(lowering, left, right,
                                                     workspaces[mode], gpu_mode).clone()
                    baseline_products[mode] = [product.clone() for product in workspaces[mode]["products"][:2]]
                    correctness.append(correctness_row(torch, session, size, seed, mode,
                                                       outputs[mode], reference))
                    staged = execute_staged(lowering, left, right,
                                            workspaces[mode], mode).clone()
                    correctness.append(correctness_row(torch, session, size, seed,
                                                       mode + "_stage_equivalence", staged, outputs[mode]))
                    for index in (0, 1):
                        difference = (workspaces[mode]["products"][index] - baseline_products[mode][index]).abs()
                        correctness.append({"session": session, "matrix_size": size, "seed": seed,
                                            "candidate": mode + f"_consumer_{index}_equivalence",
                                            "max_absolute_error": float(difference.max()),
                                            "mean_absolute_error": float(difference.mean()),
                                            "rtol": 0, "atol": 0,
                                            "passed": bool(torch.equal(workspaces[mode]["products"][index],
                                                                       baseline_products[mode][index]))})
                correctness.append(correctness_row(torch, session, size, seed, "rank8_pair_agreement",
                                                   outputs[STORED], outputs[RECOMPUTED]))
            if any(not row["passed"] for row in correctness):
                write_jsonl(args.output / f"correctness_session_{session.lower()}.jsonl", correctness)
                return 2
            torch.manual_seed(args.seed + size)
            left = torch.randn((size, size), device="cuda", dtype=torch.float16)
            right = torch.randn((size, size), device="cuda", dtype=torch.float16)
            functions = {
                STORED: lambda: execute_selected(lowering, left, right, workspaces[STORED], "stored_form"),
                RECOMPUTED: lambda: execute_selected(lowering, left, right,
                                                     workspaces[RECOMPUTED], "recomputed_form"),
            }
            # Compile, then check the real launch order and hash common kernels.
            for function in functions.values():
                function()
            torch.cuda.synchronize()
            captured = {}
            for mode, gpu_mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
                compiled = {}
                execute_selected(lowering, left, right, workspaces[mode], gpu_mode, capture=compiled)
                captured[mode] = compiled
                names = gpu_kernel_names(torch, functions[mode])
                topology[mode] = validate_topology(mode, names)
            roles = classify_kernel_roles(topology[STORED], topology[RECOMPUTED])
            common_hashes[size] = {role: {
                mode: hashlib.sha256(bytes(captured[mode][role].asm["cubin"])).hexdigest()
                for mode in (STORED, RECOMPUTED)} for role in COMMON}
            if any(len(set(hashes.values())) != 1 for hashes in common_hashes[size].values()):
                raise RuntimeError(f"common compiled kernel differs at N={size}")
            complete.extend(benchmark_pair(torch, session=session, size=size, phase="complete",
                                           functions=functions, warmups=args.warmups,
                                           iterations=args.iterations, rounds=args.rounds, seed=args.seed))
            telemetry.append({"session": session, "matrix_size": size, "after_complete": now(),
                              "nvidia_smi": command("nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu",
                                                    "--format=csv,noheader,nounits")})
            if session == "B":
                continue
            # The stored consumer input is made once outside each timed block.
            run_materialization(lowering, left, workspaces[STORED])
            consumer_functions = {
                STORED: lambda: run_consumers(lowering, left, right, workspaces[STORED], STORED),
                RECOMPUTED: lambda: run_consumers(lowering, left, right, workspaces[RECOMPUTED], RECOMPUTED),
            }
            stages.extend(benchmark_pair(torch, session=session, size=size,
                                         phase="stage_consumer", functions=consumer_functions,
                                         warmups=args.warmups, iterations=args.stage_iterations,
                                         rounds=args.stage_rounds, seed=args.seed))
            for index in (0, 1):
                individual = {
                    STORED: lambda index=index: run_consumer_product(
                        lowering, left, right, workspaces[STORED], STORED, index),
                    RECOMPUTED: lambda index=index: run_consumer_product(
                        lowering, left, right, workspaces[RECOMPUTED], RECOMPUTED, index),
                }
                stages.extend(benchmark_pair(torch, session=session, size=size,
                                             phase=f"stage_consumer_{index}", functions=individual,
                                             warmups=args.warmups, iterations=args.stage_iterations,
                                             rounds=args.stage_rounds, seed=args.seed))
            components = {
                "materialization": lambda: run_materialization(lowering, left, workspaces[STORED]),
                "common_gemms": lambda: run_common_gemms(lowering, left, right, workspaces[STORED]),
                "reconstruction": lambda: run_reconstruction(lowering, left, workspaces[STORED]),
            }
            stages.extend(benchmark_components(torch, session=session, size=size,
                                               phase="stage_component", functions=components,
                                               warmups=args.warmups, iterations=args.stage_iterations,
                                               rounds=args.stage_rounds, seed=args.seed))
            if size == max(args.sizes):
                # Both specializations and every common kernel are captured.
                for mode in (STORED, RECOMPUTED):
                    stage_handles = {}
                    execute_staged(lowering, left, right, workspaces[mode], mode, capture=stage_handles)
                    for role in (("materialization",) if mode == STORED else ()) + ("consumer_0", "consumer_1"):
                        compiler_entries.append(_compiler_capture(
                            stage_handles[role], f"{mode}_{role}_{size}", args.output / "compiler"))
                    if mode == STORED:
                        for role in COMMON:
                            compiler_entries.append(_compiler_capture(
                                stage_handles[role], f"common_{role}_{size}", args.output / "compiler"))
            # Graphs reuse the exact same preallocated buffers and production functions.
            try:
                graphs = {}
                for mode in (STORED, RECOMPUTED):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        functions[mode]()
                    graphs[mode] = graph
                    graph.replay()
                torch.cuda.synchronize()
                reference = left @ right
                for mode in (STORED, RECOMPUTED):
                    row = correctness_row(torch, session, size, args.seed + size,
                                          mode + "_graph_replay", workspaces[mode]["output"], reference)
                    correctness.append(row)
                    if not row["passed"]:
                        raise ValueError(f"CUDA Graph replay failed correctness for {mode}")
                graph_status[size] = {"supported": True, "correctness": "PASS"}
                graph_functions = {mode: graphs[mode].replay for mode in (STORED, RECOMPUTED)}
                graphs_raw.extend(benchmark_graph_modes(
                    torch, session=session, size=size, eager=functions, graphs=graph_functions,
                    warmups=args.warmups, iterations=args.diagnostic_iterations,
                    rounds=args.diagnostic_rounds, seed=args.seed))
            except Exception as exc:
                graph_status[size] = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
            # A separate one-word scratch kernel is diagnostic only.
            scratch = torch.ones((1,), device="cuda", dtype=torch.int32)
            before = scratch.clone()
            dummy_function = lambda: (run_dummy_launch(scratch), functions[RECOMPUTED]())[1]
            try:
                output = dummy_function().clone()
                reference = functions[RECOMPUTED]().clone()
                dummy_row = correctness_row(torch, session, size, args.seed + size,
                                            DUMMY, output, reference)
                dummy_row["passed"] &= bool(torch.equal(scratch, before))
                correctness.append(dummy_row)
                if not dummy_row["passed"]:
                    raise ValueError("dummy diagnostic modified output or scratch")
                topology[DUMMY] = validate_topology(DUMMY, gpu_kernel_names(torch, dummy_function))
                dummy_status[size] = {"supported": True, "correctness": "PASS",
                                      "label": "DIAGNOSTIC_ONLY"}
                dummy_raw.extend(benchmark_components(
                    torch, session=session, size=size, phase="launch_diagnostic",
                    functions={STORED: functions[STORED], RECOMPUTED: functions[RECOMPUTED],
                               DUMMY: dummy_function,
                               "tiny_dummy_DIAGNOSTIC_ONLY": lambda: run_dummy_launch(scratch)},
                    warmups=args.warmups, iterations=args.diagnostic_iterations,
                    rounds=args.diagnostic_rounds, seed=args.seed))
            except Exception as exc:
                dummy_status[size] = {"supported": False, "error": f"{type(exc).__name__}: {exc}",
                                      "label": "DIAGNOSTIC_ONLY"}
            if any(not row["passed"] for row in correctness):
                # Primary timings remain, but failed diagnostics are excluded later.
                pass
        write_csv(args.output / f"complete_timings_session_{session.lower()}.csv", complete)
        write_jsonl(args.output / f"correctness_session_{session.lower()}.jsonl", correctness)
        write_json(args.output / f"session_{session.lower()}_metadata.json",
                   {"session": session, "started_utc": start_time, "ended_utc": now(),
                    "pid": os.getpid(), "hardware": hardware, "software": software,
                    "topology": topology, "common_cubin_hashes": common_hashes,
                    "roles": roles, "graph_status": graph_status, "dummy_status": dummy_status,
                    "telemetry": telemetry})
        if session == "A":
            write_csv(args.output / "stage_timings.csv", stages)
            write_csv(args.output / "graph_timings.csv", graphs_raw)
            write_csv(args.output / "launch_diagnostic_timings.csv", dummy_raw)
            write_json(args.output / "compiler_artifacts_manifest.json", compiler_entries)
        print(f"session {session}: {len(complete)} complete timing rows, {len(correctness)} correctness rows")
        return 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def ncu_target(args) -> int:
    import torch
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for profiler target")
    lowering = frozen_lowering()
    torch.manual_seed(args.seed + args.ncu_size)
    left = torch.randn((args.ncu_size, args.ncu_size), device="cuda", dtype=torch.float16)
    right = torch.randn((args.ncu_size, args.ncu_size), device="cuda", dtype=torch.float16)
    workspace = allocate_workspace(lowering, args.ncu_size, args.ncu_target)
    execute_selected(lowering, left, right, workspace, args.ncu_target)
    torch.cuda.synchronize()
    return 0


def profile_pair(args) -> dict:
    if args.smoke or args.skip_profiler:
        return {"status": "SKIPPED", "reason": "smoke or explicit --skip-profiler"}
    if not shutil.which("ncu"):
        return {"status": "UNAVAILABLE", "reason": "ncu is not installed"}
    queried = command("ncu", "--query-metrics", "--query-metrics-mode", "all")
    available = {line.split()[0] for line in queried.splitlines()} if queried else set()
    metrics = [metric for metric in PROFILE_METRICS if metric in available]
    if not metrics:
        return {"status": "UNAVAILABLE", "reason": "requested metrics not available", "query_succeeded": bool(queried)}
    size = 512 if 512 in args.sizes else max(args.sizes)
    captures = []
    for repeat in range(2):
        for mode in ("stored_form", "recomputed_form"):
            cmd = ["ncu", "--target-processes", "all", "--metrics", ",".join(metrics),
                   "--kernel-name", "regex:(_materialize_u_form|_shared_form_gemm)", "--csv",
                   sys.executable, str(Path(__file__).resolve()), "--ncu-target", mode,
                   "--ncu-size", str(size), "--seed", str(args.seed)]
            try:
                completed = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                           check=False, timeout=180)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                captures.append({"repeat": repeat, "mode": mode, "status": "FAILED", "error": str(exc)})
                continue
            lines = completed.stdout.splitlines()
            header = next((index for index, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
            if completed.returncode or header is None:
                captures.append({"repeat": repeat, "mode": mode, "status": "FAILED",
                                 "returncode": completed.returncode,
                                 "error_excerpt": (completed.stderr + completed.stdout)[-2000:]})
                continue
            rows = list(csv.DictReader(lines[header:]))
            captures.append({"repeat": repeat, "mode": mode, "status": "PASS",
                             "kernels": rows, "command": cmd[:7] + ["<Python study target>"]})
    status = "PASS" if all(row["status"] == "PASS" for row in captures) else "PARTIAL_OR_FAILED"
    return {"status": status, "size": size, "tool_version": command("ncu", "--version"),
            "metric_names": metrics, "query_succeeded": bool(queried),
            "captures": captures, "limitation": "First-execution profiler replay in separate processes; not benchmark timing."}


def profiler_index(profile: dict) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for capture in profile.get("captures", []):
        if capture.get("status") != "PASS":
            continue
        totals: dict[str, float] = {}
        occupancy = []
        for row in capture["kernels"]:
            try:
                value = float(row["Metric Value"].replace(",", ""))
            except (ValueError, KeyError):
                continue
            if row["Metric Name"] == "sm__warps_active.avg.pct_of_peak_sustained_active":
                occupancy.append({"kernel_name": row["Kernel Name"], "percent": value})
            else:
                totals[row["Metric Name"]] = totals.get(row["Metric Name"], 0.0) + value
        by_mode.setdefault(capture["mode"], []).append(
            {"additive_changed_kernel_totals": totals, "occupancy_per_kernel": occupancy,
             "kernel_names": list(dict.fromkeys(row["Kernel Name"] for row in capture["kernels"]))})
    return {(profile["size"], STORED if mode == "stored_form" else RECOMPUTED): values
            for mode, values in by_mode.items()} if "size" in profile else {}


def orchestrate(args) -> int:
    if args.output.exists():
        raise ValueError("output directory already exists")
    selected = load_selected_points(HA2, FROZEN)
    verify_artifact_hashes(HA3)
    if any(size not in selected[0]["static_per_size"] for size in args.sizes):
        raise ValueError("sizes must occur in retained HA-Schedule-2 evidence")
    args.output.mkdir(parents=True)
    provenance = {"started_utc": now(), "source_commit": command("git", "rev-parse", "HEAD"),
                  "ha3_hash_manifest_sha256": hashlib.sha256((HA3 / "hash_manifest.json").read_bytes()).hexdigest(),
                  "ha2_hash_manifest_sha256": hashlib.sha256((HA2 / "hash_manifest.json").read_bytes()).hexdigest(),
                  "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest()}
    write_json(args.output / "config.json", {
        "sizes": args.sizes, "correctness_seeds": args.correctness_seeds,
        "warmups": args.warmups, "iterations_per_round": args.iterations,
        "rounds": args.rounds, "stage_iterations_per_round": args.stage_iterations,
        "stage_rounds": args.stage_rounds,
        "diagnostic_iterations_per_round": args.diagnostic_iterations,
        "diagnostic_rounds": args.diagnostic_rounds, "seed": args.seed,
        "rtol": RTOL, "atol": ATOL, "dtype": "fp16 input/output, fp32 product accumulation",
        "timing": "CUDA events around individual stages or complete invocation",
        "session_processes": "A and B are distinct fresh Python subprocesses"})
    write_json(args.output / "provenance.json", provenance)
    for session in ("A", "B"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-session", session,
               "--output", str(args.output), "--sizes", *map(str, args.sizes),
               "--correctness-seeds", *map(str, args.correctness_seeds),
               "--warmups", str(args.warmups), "--iterations", str(args.iterations),
               "--rounds", str(args.rounds), "--stage-iterations", str(args.stage_iterations),
               "--stage-rounds", str(args.stage_rounds),
               "--diagnostic-iterations", str(args.diagnostic_iterations),
               "--diagnostic-rounds", str(args.diagnostic_rounds), "--seed", str(args.seed)]
        result = subprocess.run(cmd, cwd=REPO, check=False)
        if result.returncode:
            write_json(args.output / "run_failure.json", {"session": session,
                       "exit_code": result.returncode, "time_utc": now()})
            return result.returncode
    profile = profile_pair(args)
    write_json(args.output / "profiler_results.json", profile)
    metadata_a = read_json(args.output / "session_a_metadata.json")
    metadata_b = read_json(args.output / "session_b_metadata.json")
    write_json(args.output / "hardware.json", metadata_a["hardware"])
    write_json(args.output / "software_versions.json", metadata_a["software"])
    topology = {mode: metadata_a["topology"][mode] for mode in (STORED, RECOMPUTED)}
    roles = classify_kernel_roles(topology[STORED], topology[RECOMPUTED])
    common_validation = {size: hashes for size, hashes in metadata_a["common_cubin_hashes"].items()}
    if any(len(set(role_hashes.values())) != 1 for hashes in common_validation.values()
           for role_hashes in hashes.values()):
        raise ValueError("common kernels did not compile identically")
    write_json(args.output / "topology.json", {"launches": topology, "role_classification": roles,
               "common_cubin_hashes_by_size": common_validation,
               "session_b_launches": metadata_b["topology"]})
    complete_raw = (read_csv(args.output / "complete_timings_session_a.csv") +
                    read_csv(args.output / "complete_timings_session_b.csv"))
    stage_raw = read_csv(args.output / "stage_timings.csv")
    graph_raw = read_csv(args.output / "graph_timings.csv")
    dummy_raw = read_csv(args.output / "launch_diagnostic_timings.csv")
    correctness = [json.loads(line) for session in ("a", "b")
                   for line in (args.output / f"correctness_session_{session}.jsonl").read_text().splitlines()]
    write_jsonl(args.output / "correctness.jsonl", correctness)
    complete_summary = summarize_raw(complete_raw)
    stage_summary = summarize_raw(stage_raw)
    graph_summary = summarize_raw(graph_raw)
    dummy_summary = summarize_raw(dummy_raw)
    complete_pairs = paired_summary(complete_summary, "complete")
    consumer_pairs = paired_summary(stage_summary, "stage_consumer")
    graph_eager_pairs = paired_summary(graph_summary, "graph_eager")
    graph_pairs = paired_summary(graph_summary, "graph_replay")
    dummy_pairs = paired_summary(dummy_summary, "launch_diagnostic", STORED, DUMMY)
    sessions = compare_sessions(complete_pairs)
    write_json(args.output / "timing_summary.json", {"complete": complete_summary,
               "stages": stage_summary, "graph": graph_summary, "launch_diagnostic": dummy_summary,
               "complete_pairs": complete_pairs, "consumer_pairs": consumer_pairs,
               "graph_eager_pairs": graph_eager_pairs, "graph_pairs": graph_pairs,
               "dummy_pairs": dummy_pairs,
               "session_comparison": sessions})
    write_json(args.output / "cuda_graph_results.json", {"status": metadata_a["graph_status"],
               "eager_and_graph_summaries": graph_summary, "graph_pairs": graph_pairs})
    write_json(args.output / "launch_overhead_diagnostic.json", {
        "label": "DIAGNOSTIC_ONLY", "status": metadata_a["dummy_status"],
        "timings": dummy_summary, "dummy_pairs": dummy_pairs,
        "note": "The tiny kernel touches only a scratch word. This is launch-scale context, not a subtracted production cost."})
    compiler_entries = read_json(args.output / "compiler_artifacts_manifest.json")
    by_label = {row["label"]: row for row in compiler_entries}
    size = max(args.sizes)
    stored = by_label[f"{STORED}_consumer_0_{size}"]
    recomputed = by_label[f"{RECOMPUTED}_consumer_0_{size}"]
    left_sass = (args.output / stored["artifacts"]["sass"]["path"]).read_text()
    right_sass = (args.output / recomputed["artifacts"]["sass"]["path"]).read_text()
    stored_1 = by_label[f"{STORED}_consumer_1_{size}"]
    recomputed_1 = by_label[f"{RECOMPUTED}_consumer_1_{size}"]
    write_json(args.output / "compiler_comparison.json", {
        "consumer_0": compiler_diff_summary(stored, recomputed, left_sass, right_sass),
        "consumer_1": compiler_diff_summary(
            stored_1, recomputed_1,
            (args.output / stored_1["artifacts"]["sass"]["path"]).read_text(),
            (args.output / recomputed_1["artifacts"]["sass"]["path"]).read_text()),
        "common_cubin_hashes_by_size": common_validation,
        "all_compiled_kernels": compiler_entries})
    static_join = join_static_measured(
        selected, args.sizes, complete_summary,
        [row for row in stage_summary if row["phase"] == "stage_consumer"],
        topology, profiler_index(profile))
    write_json(args.output / "static_vs_measured.json", static_join)
    materialization = [row for row in stage_summary if row["variant"] == "materialization"]
    classification = classify_mechanism(sessions, complete_pairs, consumer_pairs,
                                        materialization, graph_pairs, dummy_pairs,
                                        graph_eager_pairs)
    aggregate = {"classification": classification, "correctness_rows": len(correctness),
                 "correctness_failed": sum(not row["passed"] for row in correctness),
                 "complete_raw_rows": len(complete_raw), "stage_raw_rows": len(stage_raw),
                 "graph_raw_rows": len(graph_raw), "dummy_raw_rows": len(dummy_raw),
                 "session_process_ids": [metadata_a["pid"], metadata_b["pid"]],
                 "session_comparison": sessions, "profiler_status": profile["status"]}
    write_json(args.output / "aggregate_summary.json", aggregate)
    (args.output / "study_notes.md").write_text(
        "# HA-Schedule-4 mechanism study\n\n"
        f"Implementation commit: `{provenance['source_commit']}`.\n"
        f"Primary classification: `{classification['primary']}`. Decision: `{classification['decision']}`.\n\n"
        "See docs/hardware_aware_schedule_mechanism_study.md for interpretation.\n")
    hashes = {path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "hash_manifest.json", hashes)
    print(f"wrote HA-Schedule-4 evidence to {args.output}; classification: {classification['primary']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--correctness-seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--stage-iterations", type=int, default=100)
    parser.add_argument("--stage-rounds", type=int, default=15)
    parser.add_argument("--diagnostic-iterations", type=int, default=100)
    parser.add_argument("--diagnostic-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-session", choices=("A", "B"), help=argparse.SUPPRESS)
    parser.add_argument("--ncu-target", choices=("stored_form", "recomputed_form"), help=argparse.SUPPRESS)
    parser.add_argument("--ncu-size", type=int, default=512, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ncu_target:
        return ncu_target(args)
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.resolve()
    if args.smoke:
        args.sizes = [128]
        args.warmups = 2
        args.iterations = 5
        args.rounds = 2
        args.stage_iterations = 5
        args.stage_rounds = 2
        args.diagnostic_iterations = 5
        args.diagnostic_rounds = 2
    if any(size < 2 or size % 2 for size in args.sizes):
        parser.error("sizes must be positive even matrix dimensions")
    if len(args.correctness_seeds) < 5:
        parser.error("five deterministic correctness seeds are required")
    if min(args.warmups, args.iterations, args.rounds, args.stage_iterations,
           args.stage_rounds, args.diagnostic_iterations, args.diagnostic_rounds) < 1:
        parser.error("timing counts must be positive")
    if args.worker_session:
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
