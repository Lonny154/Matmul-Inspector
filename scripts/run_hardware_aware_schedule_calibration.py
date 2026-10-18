#!/usr/bin/env python3
"""Rank retained legal schedules with size- and execution-mode-specific evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from matmul_inspector.tensor_game_schedule_calibration import (
    MODES, build_profiles, calibrated_rank, leave_one_session_out, pareto_hashes,
    schedule_features, static_rank, validation_assessment, verify_profile_sources,
)
from matmul_inspector.tensor_game_schedule_gpu_study import (
    _schedule, load_selected_points, verify_artifact_hashes,
)
from matmul_inspector.tensor_game_schedule_model import (
    BlockConfig, POLICIES, build_execution_dag, evaluate_schedule, make_schedule,
    reference_dag, validate_schedule,
)
from scripts.run_hardware_aware_schedule_study import project_actions


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts/tensor-game"
HA1 = ROOT / "hardware-aware-schedule-v1"
HA2 = ROOT / "hardware-aware-schedule-search-v1"
HA3 = ROOT / "hardware-aware-schedule-gpu-v1"
HA4 = ROOT / "hardware-aware-schedule-mechanism-v1"
FROZEN = ROOT / "hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"
SIZES = (128, 256, 512, 1024)
ALGORITHMS = ("conventional", "strassen", "project-rank8")


def read_json(path: Path):
    return json.loads(path.read_text())


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in rows))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_data() -> dict:
    manifests = {name: verify_artifact_hashes(path) for name, path in
                 (("HA1", HA1), ("HA2", HA2), ("HA3", HA3), ("HA4", HA4))}
    selected = load_selected_points(HA2, FROZEN)
    if read_json(HA4 / "aggregate_summary.json")["correctness_failed"]:
        raise ValueError("HA-Schedule-4 source correctness failed")
    session_a = read_json(HA4 / "session_a_metadata.json")
    session_b = read_json(HA4 / "session_b_metadata.json")
    if session_a["pid"] == session_b["pid"] or session_a["roles"] != session_b["roles"]:
        raise ValueError("HA-Schedule-4 fresh-session or topology provenance is invalid")
    if any(not record["supported"] for record in session_a["graph_status"].values()):
        raise ValueError("retained graph evidence is incomplete")
    files = ("timing_summary.json", "stage_timings.csv", "launch_diagnostic_timings.csv",
             "topology.json", "hardware.json", "software_versions.json", "compiler_comparison.json",
             "session_a_metadata.json", "session_b_metadata.json")
    source_hashes = {"hash_manifest.json": sha(HA4 / "hash_manifest.json"),
                     **{name: manifests["HA4"][name] for name in files}}
    return {"manifests": manifests, "selected": selected, "source_hashes": source_hashes,
            "source_manifest": {name: {"artifact": str(path.relative_to(REPO)),
                                       "manifest_sha256": sha(path / "hash_manifest.json"),
                                       "entry_count": len(manifests[name])}
                                for name, path in (("HA1", HA1), ("HA2", HA2), ("HA3", HA3), ("HA4", HA4))},
            "frozen_sha256": sha(FROZEN)}


def compiler_resources() -> dict:
    saved = read_json(HA4 / "compiler_comparison.json")
    return {role: {"registers_per_thread": saved[role]["registers_per_thread"],
                   "static_shared_memory_bytes": saved[role]["static_shared_memory_bytes"],
                   "cubin_sha256": [saved[role]["stored_cubin_sha256"],
                                    saved[role]["recomputed_cubin_sha256"]],
                   "capture_matrix_size": 1024}
            for role in ("consumer_0", "consumer_1")}


def calibration_profiles(source: dict) -> list[dict]:
    timing = read_json(HA4 / "timing_summary.json")
    profiles = build_profiles(
        timing, timing["stages"], timing["launch_diagnostic"],
        read_json(HA4 / "hardware.json"), read_json(HA4 / "software_versions.json"),
        str(HA4.relative_to(REPO)), source["source_hashes"],
        read_json(HA4 / "topology.json")["launches"], compiler_resources())
    for profile in profiles:
        verify_profile_sources(profile, REPO)
    return profiles


def collect_candidates(source: dict) -> list[dict]:
    ha2_config = read_json(HA2 / "config.json")
    if (ha2_config["matrix_sizes"] != list(SIZES) or ha2_config["input_dtype"] != "fp16"
            or ha2_config["accumulation_dtype"] != "fp32" or ha2_config["output_dtype"] != "fp16"):
        raise ValueError("retained CP-SAT corpus has a different size or dtype configuration")
    actions = project_actions(FROZEN)
    corpus = read_lines(HA2 / "schedules.jsonl")
    known = {(row["algorithm"], int(size), per_size["schedule_hash"]): row["name"]
             for row in source["selected"] if row["gpu_mode"] is not None
             for size, per_size in row["static_per_size"].items()}
    # Deduplicate on canonical semantic schedule hash, not solver-run identity.
    by_key: dict[tuple[str, int, str], dict] = {}
    for size in SIZES:
        config = BlockConfig(size, "fp16", "fp32", "fp16")
        for algorithm in ALGORITHMS:
            dag = (build_execution_dag(algorithm, actions, config) if algorithm == "project-rank8"
                   else reference_dag(algorithm, config))
            entries = [("retained_cp_sat", row["source_run_id"], _schedule(row["schedule"]))
                       for row in corpus if row["algorithm"] == algorithm and row["matrix_size"] == size]
            entries += [("manual_control", policy, make_schedule(dag, policy)) for policy in POLICIES]
            for source_kind, source_id, schedule in entries:
                check = validate_schedule(dag, schedule)
                if not check.valid:
                    raise ValueError(f"invalid schedule {algorithm}/{size}/{source_id}: {check.issues}")
                metrics = evaluate_schedule(dag, schedule)
                if metrics["validation_status"] != "PASS":
                    raise ValueError("independent schedule evaluator did not pass")
                key = (algorithm, size, schedule.schedule_hash)
                if key in by_key:
                    continue
                lowering_name = known.get(key)
                by_key[key] = {
                    "algorithm": algorithm, "matrix_size": size,
                    "schedule_hash": schedule.schedule_hash, "schedule": asdict(schedule),
                    "source": source_kind, "source_id": source_id,
                    "policy": schedule.policy, "legality": "PASS",
                    "lowering_name": lowering_name,
                    "lowering_status": "APPROXIMATELY_IMPLEMENTABLE" if lowering_name else "UNCALIBRATED_LOWERING",
                    "metrics": metrics,
                    "features": schedule_features(dag, schedule, metrics, lowering_name),
                }
    return sorted(by_key.values(), key=lambda row: (row["algorithm"], row["matrix_size"], row["schedule_hash"]))


def rank_study(candidates: list[dict], profiles: list[dict]) -> dict:
    by_profile = {(row["matrix_size"], row["execution_mode"]): row for row in profiles}
    hardware, software = profiles[0]["hardware"], profiles[0]["software"]
    static_rows, eager_rows, graph_rows, comparisons = [], [], [], []
    for algorithm in ALGORITHMS:
        for size in SIZES:
            group = [row for row in candidates if row["algorithm"] == algorithm and row["matrix_size"] == size]
            static = static_rank(group)
            eager = calibrated_rank(group, by_profile[(size, "EAGER")], hardware=hardware, software=software)
            graph = calibrated_rank(group, by_profile[(size, "CUDA_GRAPH")], hardware=hardware, software=software)
            static_rows += static
            eager_rows += [{"algorithm": algorithm, "matrix_size": size, **row} for row in eager]
            graph_rows += [{"algorithm": algorithm, "matrix_size": size, **row} for row in graph]
            mapped = [row for row in group if row["lowering_name"]]
            mapped_static = static_rank(mapped)
            def best(rows: list[dict], field: str) -> list[str]:
                ranks = [row[field] for row in rows if row[field] is not None]
                return sorted(row["schedule_hash"] for row in rows if row[field] == min(ranks)) if ranks else []
            rank_by = {row["schedule_hash"]: row["static_rank"] for row in static}
            mapped_rank_by = {row["schedule_hash"]: row["static_rank"] for row in mapped_static}
            eager_by = {row["schedule_hash"]: row for row in eager}
            graph_by = {row["schedule_hash"]: row for row in graph}
            pair = {row["lowering_name"]: row for row in mapped if row["lowering_name"] in
                    ("rank8_stored", "rank8_recomputed")}
            comparisons.append({
                "algorithm": algorithm, "matrix_size": size, "candidate_count": len(group),
                "calibrated_lowering_count": len(mapped), "pareto_hashes": pareto_hashes(group),
                "static_best_all": best(static, "static_rank"),
                "static_best_mapped": best(mapped_static, "static_rank"),
                "eager_best_mapped": best(eager, "calibrated_rank"),
                "graph_best_mapped": best(graph, "calibrated_rank"),
                "eager_confidence": by_profile[(size, "EAGER")]["confidence"],
                "graph_confidence": by_profile[(size, "CUDA_GRAPH")]["confidence"],
                "rank_diagnostics": [{"schedule_hash": row["schedule_hash"],
                                      "lowering_name": row["lowering_name"],
                                      "static_rank_all": rank_by[row["schedule_hash"]],
                                      "static_rank_mapped": mapped_rank_by.get(row["schedule_hash"]),
                                      "eager_rank_mapped": eager_by[row["schedule_hash"]]["calibrated_rank"],
                                      "graph_rank_mapped": graph_by[row["schedule_hash"]]["calibrated_rank"],
                                      "features": row["features"]} for row in group],
                "rank8_pair": {name: {"schedule_hash": row["schedule_hash"],
                                        "static_rank_mapped": mapped_rank_by[row["schedule_hash"]],
                                        "eager_rank_mapped": eager_by[row["schedule_hash"]]["calibrated_rank"],
                                        "graph_rank_mapped": graph_by[row["schedule_hash"]]["calibrated_rank"]}
                               for name, row in pair.items()},
            })
    return {"static": static_rows, "eager": eager_rows, "graph": graph_rows,
            "comparisons": comparisons}


def gpu_worker(args) -> int:
    import torch
    import triton
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
    from matmul_inspector.tensor_game_schedule_mechanism_study import (
        RECOMPUTED, STORED, paired_summary, summarize_raw,
    )
    from scripts.run_hardware_aware_schedule_mechanism_study import (
        benchmark_graph_modes, benchmark_pair, frozen_lowering,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("fresh validation requires CUDA")
    expected_hardware = read_json(HA4 / "hardware.json")
    if torch.cuda.get_device_name(0) != expected_hardware["gpu_name"] or list(torch.cuda.get_device_capability(0)) != expected_hardware["compute_capability"]:
        raise ValueError("fresh validation GPU does not match calibration hardware")
    source_software = read_json(HA4 / "software_versions.json")
    if (torch.__version__ != source_software["torch"] or triton.__version__ != source_software["triton"]
            or torch.version.cuda != source_software["cuda_runtime"]):
        raise ValueError("fresh validation software differs from calibration")
    lowering = frozen_lowering()
    rows, checks, status = [], [], {}
    started = datetime.now(timezone.utc).isoformat()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for size in SIZES:
            workspaces = {STORED: allocate_workspace(lowering, size, "stored_form"),
                          RECOMPUTED: allocate_workspace(lowering, size, "recomputed_form")}
            correct = True
            for seed in (11, 23, 37, 53, 71):
                torch.manual_seed(seed)
                left = torch.randn((size, size), device="cuda", dtype=torch.float16)
                right = torch.randn((size, size), device="cuda", dtype=torch.float16)
                reference = left @ right
                for name, mode in ((STORED, "stored_form"), (RECOMPUTED, "recomputed_form")):
                    result = execute_selected(lowering, left, right, workspaces[name], mode)
                    passed = bool(torch.allclose(result, reference, rtol=5e-3, atol=5e-2))
                    checks.append({"matrix_size": size, "seed": seed, "variant": name,
                                   "phase": "eager", "passed": passed})
                    correct &= passed
            if not correct:
                status[str(size)] = {"eager": "CORRECTNESS_FAILED", "graph": "NOT_RUN"}
                continue
            torch.manual_seed(args.seed + size)
            left = torch.randn((size, size), device="cuda", dtype=torch.float16)
            right = torch.randn((size, size), device="cuda", dtype=torch.float16)
            eager = {STORED: lambda: execute_selected(lowering, left, right, workspaces[STORED], "stored_form"),
                     RECOMPUTED: lambda: execute_selected(lowering, left, right, workspaces[RECOMPUTED], "recomputed_form")}
            graphs = {}
            try:
                for name in (STORED, RECOMPUTED):
                    eager[name]()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        eager[name]()
                    graph.replay()
                    graphs[name] = graph
                torch.cuda.synchronize()
                reference = left @ right
                graph_correct = True
                for name in (STORED, RECOMPUTED):
                    passed = bool(torch.allclose(workspaces[name]["output"], reference, rtol=5e-3, atol=5e-2))
                    checks.append({"matrix_size": size, "seed": args.seed + size,
                                   "variant": name, "phase": "graph", "passed": passed})
                    graph_correct &= passed
                if not graph_correct:
                    rows.extend(benchmark_pair(torch, session="FRESH", size=size,
                                               phase="graph_eager", functions=eager,
                                               warmups=args.warmups, iterations=args.iterations,
                                               rounds=args.rounds, seed=args.seed))
                    status[str(size)] = {"eager": "PASS", "graph": "CORRECTNESS_FAILED"}
                    continue
                graph_functions = {name: graphs[name].replay for name in graphs}
                rows.extend(benchmark_graph_modes(torch, session="FRESH", size=size,
                                                  eager=eager, graphs=graph_functions,
                                                  warmups=args.warmups, iterations=args.iterations,
                                                  rounds=args.rounds, seed=args.seed))
                status[str(size)] = {"eager": "PASS", "graph": "PASS"}
            except Exception as exc:
                rows.extend(benchmark_pair(torch, session="FRESH", size=size,
                                           phase="graph_eager", functions=eager,
                                           warmups=args.warmups, iterations=args.iterations,
                                           rounds=args.rounds, seed=args.seed))
                status[str(size)] = {"eager": "PASS", "graph": "UNSUPPORTED",
                                     "error": f"{type(exc).__name__}: {exc}"}
        summary = summarize_raw(rows)
        write_json(args.output / "gpu_validation.json", {
            "hardware": expected_hardware, "software": source_software,
            "actual_gpu_name": torch.cuda.get_device_name(0),
            "actual_compute_capability": torch.cuda.get_device_capability(0),
            "actual_torch": torch.__version__, "actual_triton": triton.__version__,
            "process_id": os.getpid(), "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "correctness": checks, "timing_summaries": summary,
            "eager_pairs": paired_summary(summary, "graph_eager"),
            "graph_pairs": paired_summary(summary, "graph_replay"),
            "protocol": {"warmups": args.warmups, "iterations_per_round": args.iterations,
                         "rounds": args.rounds, "cuda_events": True, "seeds": [11, 23, 37, 53, 71]},
        })
        write_lines(args.output / "gpu_validation_raw.jsonl", rows)
        return 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true", help="CPU-only structural smoke; no decision gate")
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.warmups, args.iterations, args.rounds) < 1:
        parser.error("timing counts must be positive")
    if args.gpu_worker:
        return gpu_worker(args)
    if args.output.exists():
        parser.error("output directory already exists")
    source = source_data()
    profiles = calibration_profiles(source)
    candidates = collect_candidates(source)
    ranking = rank_study(candidates, profiles)
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", {"algorithms": ALGORITHMS, "matrix_sizes": SIZES,
               "modes": MODES, "candidate_source": "retained HA-Schedule-2 CP-SAT plus manual controls",
               "no_new_solver_runs": True, "gpu_validation": not args.skip_gpu,
               "warmups": args.warmups, "iterations": args.iterations, "rounds": args.rounds})
    write_json(args.output / "provenance.json", {"source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_rank8_sha256": source["frozen_sha256"]})
    write_json(args.output / "calibration_source_manifest.json", source["source_manifest"])
    write_json(args.output / "calibration_profiles.json", profiles)
    write_lines(args.output / "candidate_schedules.jsonl", candidates)
    write_lines(args.output / "schedule_features.jsonl", [
        {"algorithm": row["algorithm"], "matrix_size": row["matrix_size"],
         "schedule_hash": row["schedule_hash"], "legality": row["legality"],
         "lowering_name": row["lowering_name"], "features": row["features"]} for row in candidates])
    write_json(args.output / "static_rankings.json", ranking["static"])
    write_json(args.output / "eager_calibrated_rankings.json", ranking["eager"])
    write_json(args.output / "graph_calibrated_rankings.json", ranking["graph"])
    write_json(args.output / "ranking_comparison.json", ranking["comparisons"])
    write_json(args.output / "leave_one_session_out.json", leave_one_session_out(
        read_json(HA4 / "timing_summary.json")["complete_pairs"]))
    if args.skip_gpu:
        write_json(args.output / "gpu_validation.json", {"status": "SKIPPED_BY_REQUEST"})
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
               "--gpu-worker", "--warmups", str(args.warmups), "--iterations", str(args.iterations),
               "--rounds", str(args.rounds), "--seed", str(args.seed)]
        result = subprocess.run(cmd, cwd=REPO, check=False)
        if result.returncode:
            write_json(args.output / "gpu_validation.json", {"status": "WORKER_FAILED",
                       "returncode": result.returncode})
    fresh = read_json(args.output / "gpu_validation.json")
    fresh_eager = {row["matrix_size"]: row for row in fresh.get("eager_pairs", [])}
    fresh_graph = {row["matrix_size"]: row for row in fresh.get("graph_pairs", [])}
    assessment = [{"matrix_size": profile["matrix_size"], "execution_mode": profile["execution_mode"],
                   "profile_id": profile["profile_id"], **validation_assessment(
                       profile, (fresh_eager if profile["execution_mode"] == "EAGER" else fresh_graph).get(
                           profile["matrix_size"]))} for profile in profiles]
    write_json(args.output / "validation_assessment.json", assessment)
    comparisons = ranking["comparisons"]
    pair_groups = [row for row in comparisons if row["algorithm"] == "project-rank8"]
    selection_changes = sum(row["eager_best_mapped"] != row["graph_best_mapped"] for row in pair_groups)
    rank_changes = sum(item["static_rank_mapped"] != item["eager_rank_mapped"]
                       for row in pair_groups for item in row["rank8_pair"].values())
    rank_changes += sum(item["static_rank_mapped"] != item["graph_rank_mapped"]
                        for row in pair_groups for item in row["rank8_pair"].values())
    full_validation = all(row["status"] == "DIRECTION_AGREES" for row in assessment)
    any_contradiction = any(row["status"] in ("CONTRADICTED", "NOT_REPRODUCED") for row in assessment)
    gate = ("CALIBRATION_UNSTABLE" if any_contradiction else
            "MODE_SPECIFIC_SIGNAL" if full_validation and selection_changes else
            "INCONCLUSIVE")
    readiness = ("READY_FOR_CLOSED_LOOP_PROTOTYPE" if full_validation and selection_changes else
                 "NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE")
    aggregate = {"candidate_count": len(candidates),
                 "candidate_count_by_algorithm": {algorithm: len([row for row in candidates
                                                if row["algorithm"] == algorithm and row["matrix_size"] == 128])
                                                  for algorithm in ALGORITHMS},
                 "mapped_candidate_count": sum(row["lowering_name"] is not None for row in candidates),
                 "rank8_rank_changes": rank_changes, "rank8_eager_graph_selection_changes": selection_changes,
                 "rank8_static_eager_agreement_count": sum(row["static_best_mapped"] == row["eager_best_mapped"]
                                                           for row in pair_groups),
                 "rank8_static_graph_agreement_count": sum(row["static_best_mapped"] == row["graph_best_mapped"]
                                                           for row in pair_groups),
                 "rank8_eager_graph_agreement_count": 4 - selection_changes,
                 "decision": gate, "secondary_decision": "CALIBRATED_SEARCH_SIGNAL" if full_validation and rank_changes else None,
                 "closed_loop_readiness": readiness,
                 "readiness_reason": ("Known lowerings, correctness gates, and mode-specific calibration are available; "
                                      "unmapped schedules remain out of the executable loop."
                                      if readiness == "READY_FOR_CLOSED_LOOP_PROTOTYPE" else
                                      "Fresh validation or lowering coverage is insufficient for an automated reselection loop."),
                 "validation_assessment": assessment}
    write_json(args.output / "aggregate_summary.json", aggregate)
    (args.output / "study_notes.md").write_text(
        "# HA-Schedule-5 calibrated schedule search\n\n"
        f"Decision: `{gate}`. Closed-loop readiness: `{readiness}`.\n"
        "Ordinal calibration applies only to validated GPU-lowering mappings. "
        "No candidate outside that set receives a fabricated physical launch count or latency estimate. "
        "See docs/hardware_aware_schedule_calibration.md for results and limitations.\n")
    hashes = {path.relative_to(args.output).as_posix(): sha(path)
              for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "hash_manifest.json", hashes)
    print(f"{len(candidates)} legal candidates; {gate}; {readiness} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
