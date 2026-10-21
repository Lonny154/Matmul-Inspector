#!/usr/bin/env python3
"""Run a bounded HA-Schedule-10 CPU replay or CUDA experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from matmul_inspector.tensor_game_schedule_reselection import (
    MODES, compare_reselection, load_retained_corpus, make_observation,
    select_batch1, select_batch2, validate_observations,
)
from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_ir import translate_schedule, validate_ir
from matmul_inspector.tensor_game_schedule_ir_planner import plan_ir
from matmul_inspector.tensor_game_schedule_model import validate_schedule
from scripts.run_hardware_aware_schedule_lowering_study import (
    FROZEN, REPO, make_dag, project_actions, source_rows,
)

SEEDS = (11, 23, 37)
CONFIG = {"batch1_limit": 8, "batch2_limit": 4, "matrix_size": 256,
          "modes": MODES, "warmups": 3, "rounds": 5, "iterations_per_round": 5,
          "benchmark_seed": 104729, "correctness_seeds": SEEDS,
          "rtol": 0.005, "atol": 0.05, "timing": "CUDA event, device execution"}


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def cpu_gate(selected: list[dict], source: dict[str, dict], actions) -> list[dict]:
    results = []
    for row in selected:
        original = source[row["schedule_hash"]]
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(original["schedule"])
        legality = validate_schedule(dag, schedule)
        ir = translate_schedule(dag, schedule)
        validation = validate_ir(dag, schedule, ir)
        plan = plan_ir(dag, schedule, ir, lowering_name=original["lowering_name"],
                       static_metrics=original["metrics"], calibration_features=original["features"])
        status = "PASS" if legality.valid and validation["status"] == "IR_VALID" and (
            plan.mapping_status == "EXACT_MAPPING" and
            ir.ir_hash == row["ir_hash"] and plan.lowering_plan_hash == row["lowering_plan_hash"]
            ) else "FAIL"
        results.append({"schedule_hash": row["schedule_hash"], "legality": legality.valid,
                        "ir_status": validation["status"], "mapping_status": plan.mapping_status,
                        "ir_hash": ir.ir_hash, "lowering_plan_hash": plan.lowering_plan_hash,
                        "status": status})
    if any(row["status"] != "PASS" for row in results):
        raise ValueError("selected schedule failed CPU legality/IR/lowerability gate")
    return results


def gpu_batch(selected: list[dict], source: dict[str, dict], actions, label: str):
    """Use the HA-9 runtime and HA-4 matched CUDA-event benchmark helpers."""
    import torch
    import triton
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_ir_runtime import (
        allocate_backend, audit_ir_workspace, execute_backend,
    )
    from matmul_inspector.tensor_game_schedule_mechanism_study import summarize_raw
    from scripts.run_hardware_aware_schedule_mechanism_study import benchmark_components

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    correctness, topology, raw, summaries, observations = [], [], [], [], []
    functions: dict[str, dict[str, object]] = {mode: {} for mode in MODES}
    grouped: dict[str, list[dict]] = defaultdict(list)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row in selected:
            original = source[row["schedule_hash"]]
            dag = make_dag(row["algorithm"], row["matrix_size"], actions)
            schedule = _schedule(original["schedule"])
            ir = translate_schedule(dag, schedule)
            plan = plan_ir(dag, schedule, ir, lowering_name=original["lowering_name"],
                           static_metrics=original["metrics"], calibration_features=original["features"])
            workspace = allocate_backend(dag, schedule, ir, plan, lowering_name=original["lowering_name"])
            audit = audit_ir_workspace(dag, schedule, ir, plan, workspace)
            if audit["status"] != "PASS":
                raise ValueError("GPU workspace audit failed")
            size = row["matrix_size"]
            for seed in SEEDS:
                torch.manual_seed(seed)
                left = torch.randn((size, size), device="cuda", dtype=torch.float16)
                right = torch.randn_like(left)
                actual = execute_backend(dag, schedule, ir, plan, left, right, workspace,
                                         lowering_name=original["lowering_name"])
                reference = left @ right
                torch.cuda.synchronize()
                error = (actual - reference).abs().float()
                correctness.append({"batch": label, "schedule_hash": row["schedule_hash"],
                    "seed": seed, "passed": bool(torch.allclose(actual, reference,
                    rtol=CONFIG["rtol"], atol=CONFIG["atol"])),
                    "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
                    "rtol": CONFIG["rtol"], "atol": CONFIG["atol"]})
            if not all(item["passed"] for item in correctness if item["schedule_hash"] == row["schedule_hash"]):
                continue
            captured = []
            execute_backend(dag, schedule, ir, plan, left, right, workspace,
                            lowering_name=original["lowering_name"], capture=captured)
            compiler_hashes = [{"group_id": item["group_id"], "primitive": item["role"],
                "cubin_sha256": hashlib.sha256(bytes(item["compiled"].asm["cubin"])).hexdigest()}
                for item in captured]
            def eager(dag=dag, schedule=schedule, ir=ir, plan=plan, left=left, right=right,
                      workspace=workspace, name=original["lowering_name"]):
                return execute_backend(dag, schedule, ir, plan, left, right, workspace,
                                       lowering_name=name)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
                eager()
                torch.cuda.synchronize()
            actual_names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
            expected = [entry["kernel"] for entry in plan.launch_sequence]
            agrees = len(expected) == len(actual_names) and all(
                name.startswith(kernel) for name, kernel in zip(actual_names, expected))
            topology.append({"batch": label, "schedule_hash": row["schedule_hash"],
                             "expected_kernel_names": expected, "actual_kernel_names": actual_names,
                             "topology_agrees": agrees, "workspace_audit": audit,
                             "compiler_hashes": compiler_hashes})
            if not agrees:
                continue
            eager()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eager()
            graph.replay()
            torch.cuda.synchronize()
            graph_pass = bool(torch.allclose(workspace["output"], reference,
                                            rtol=CONFIG["rtol"], atol=CONFIG["atol"]))
            correctness.append({"batch": label, "schedule_hash": row["schedule_hash"],
                                "seed": SEEDS[-1], "mode": "CUDA_GRAPH", "passed": graph_pass,
                                "rtol": CONFIG["rtol"], "atol": CONFIG["atol"]})
            functions["EAGER"][row["schedule_hash"]] = eager
            if graph_pass:
                functions["CUDA_GRAPH"][row["schedule_hash"]] = graph.replay
            grouped[row["algorithm"]].append(row)
        for mode in MODES:
            for algorithm, group in grouped.items():
                active = {row["schedule_hash"]: functions[mode][row["schedule_hash"]]
                          for row in group if row["schedule_hash"] in functions[mode]}
                if not active:
                    continue
                timed = benchmark_components(torch, session=label, size=CONFIG["matrix_size"],
                    phase=f"{algorithm}:{mode}", functions=active, warmups=CONFIG["warmups"],
                    iterations=CONFIG["iterations_per_round"], rounds=CONFIG["rounds"],
                    seed=CONFIG["benchmark_seed"])
                raw.extend({**item, "execution_mode": mode, "algorithm": algorithm}
                           for item in timed)
                summary = summarize_raw(timed)
                summaries.extend({**item, "execution_mode": mode, "algorithm": algorithm}
                                 for item in summary)
                by_hash = {item["variant"]: item for item in summary}
                active_rows = [row for row in group if row["schedule_hash"] in active]
                for i, left_row in enumerate(active_rows):
                    for right_row in active_rows[i + 1:]:
                        observations.append(make_observation(left_row, right_row, mode=mode,
                            left_ms=by_hash[left_row["schedule_hash"]]["round_medians_ms"],
                            right_ms=by_hash[right_row["schedule_hash"]]["round_medians_ms"],
                            provenance={"evidence_kind": "FRESH_GPU", "batch": label,
                                        "device": torch.cuda.get_device_name(),
                                        "compute_capability": torch.cuda.get_device_capability(),
                                        "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                                        "torch": torch.__version__, "triton": triton.__version__,
                                        "cuda_runtime": torch.version.cuda},
                            benchmark=CONFIG,
                            correctness={row["schedule_hash"]: "PASS" for row in active_rows}))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return correctness, topology, raw, summaries, observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true", help="Run fresh CUDA validation and measurement")
    parser.add_argument("--observations", type=Path, help="Replay a JSON list of real or labeled synthetic observations")
    args = parser.parse_args()
    if args.gpu and args.observations:
        parser.error("--gpu and --observations are mutually exclusive")
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory already exists")
    candidates, source_provenance = load_retained_corpus(REPO)
    source = {row["schedule_hash"]: row for row in source_rows()}
    actions = project_actions(FROZEN)
    batch1 = select_batch1(candidates, size=CONFIG["matrix_size"])
    gate1 = cpu_gate(batch1, source, actions)
    observations = json.loads(args.observations.read_text()) if args.observations else []
    evidence_kind = "REPLAY_INPUT" if args.observations else "NONE"
    gpu_correctness, topology, raw, summaries = [], [], [], []
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use CPU replay or run on a CUDA host")
        checks, launches, samples, summary, observations = gpu_batch(batch1, source, actions, "batch1")
        gpu_correctness.extend(checks); topology.extend(launches)
        raw.extend(samples); summaries.extend(summary)
        evidence_kind = "FRESH_GPU"
    for mode in MODES:
        validate_observations(observations, candidates, mode)
    frozen = {mode: select_batch2(candidates, batch1, [], mode=mode) for mode in MODES}
    updated = {mode: select_batch2(candidates, batch1, observations, mode=mode) for mode in MODES}
    comparisons = {mode: compare_reselection(frozen[mode], updated[mode]) for mode in MODES}
    # The eager choice defines the four-configuration physical Batch 2 budget.
    # Graph reselection is retained as a mode-specific counterfactual if it differs.
    batch2 = [next(row for row in candidates if row["schedule_hash"] == item["schedule_hash"])
              for item in updated["EAGER"]]
    gate2 = cpu_gate(batch2, source, actions)
    if args.gpu:
        # A shared candidate may serve both modes. Unique configuration count remains bounded by 8.
        checks, launches, samples, summary, _ = gpu_batch(batch2, source, actions, "batch2")
        gpu_correctness.extend(checks); topology.extend(launches)
        raw.extend(samples); summaries.extend(summary)
    output.mkdir(parents=True)
    write(output / "config.json", CONFIG)
    write(output / "provenance.json", {**source_provenance,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selector_sha256": hashlib.sha256((REPO / "src/matmul_inspector/tensor_game_schedule_reselection.py").read_bytes()).hexdigest(),
        "evidence_kind": evidence_kind,
        "observation_input_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest()
        if args.observations else None})
    write_rows(output / "candidate_universe.jsonl", candidates)
    write(output / "batch1_selection.json", {"rule": "N=256, four rank-8, two conventional, two Strassen; "
          "greedy structural token coverage, hash tie-break", "candidates": batch1})
    write_rows(output / "cpu_gates.jsonl", [{"batch": "batch1", **row} for row in gate1] +
               [{"batch": "batch2", **row} for row in gate2])
    write_rows(output / "gpu_correctness.jsonl", gpu_correctness)
    write_rows(output / "gpu_topology.jsonl", topology)
    write_rows(output / "raw_device_timing.jsonl", raw)
    write_rows(output / "timing_summary.jsonl", summaries)
    write(output / "calibration_before.json", {"observations": [], "status": "NO_IR_WIDE_MEASUREMENTS"})
    write(output / "calibration_after.json", {"observations": observations,
          "status": "MEASURED" if evidence_kind == "FRESH_GPU" else
                    "REPLAY_ONLY" if observations else "PENDING_GPU"})
    write(output / "batch2_frozen.json", frozen)
    write(output / "batch2_updated.json", updated)
    write(output / "reselection_comparison.json", comparisons)
    measured_hashes = {row["schedule_hash"] for row in batch1 + batch2}
    gpu_all_pass = (args.gpu and all(
        {row["seed"] for row in gpu_correctness if row["schedule_hash"] == schedule_hash
         and row.get("mode") is None and row["passed"]} >= set(SEEDS)
        and any(row["schedule_hash"] == schedule_hash and row.get("mode") == "CUDA_GRAPH"
                and row["passed"] for row in gpu_correctness)
        and any(row["schedule_hash"] == schedule_hash and row["topology_agrees"]
                for row in topology) for schedule_hash in measured_hashes)
        and all(any(row["mode"] == mode for row in observations) for mode in MODES))
    write(output / "decision.json", {"classification": "A_EMPIRICALLY_DEMONSTRATED"
          if gpu_all_pass and any(item["changed"] for item in comparisons.values())
          else "C_NO_MEANINGFUL_RESELECTION" if gpu_all_pass else
          "B_MECHANICALLY_DEMONSTRATED_ONLY" if observations and any(
              item["changed"] for item in comparisons.values()) and not args.gpu else
          "D_HARDWARE_CALIBRATION_PENDING",
          "gpu_available": args.gpu, "evidence_kind": evidence_kind,
          "gpu_correctness_and_topology_complete": gpu_all_pass,
          "selected_batch1": len(batch1), "selected_batch2_unique": len(batch2),
          "changed_by_mode": {mode: comparisons[mode]["changed"] for mode in MODES}})
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir()) if path.is_file()}
    write(output / "hash_manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": json.loads((output / "decision.json").read_text())},
                     sort_keys=True))


if __name__ == "__main__":
    main()
