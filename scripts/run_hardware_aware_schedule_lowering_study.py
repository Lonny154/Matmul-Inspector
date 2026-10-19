#!/usr/bin/env python3
"""Classify HA-5 schedules, validate bounded GPU templates, and replicate graph timing."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule, verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_lowering import (
    aggregate_graph_sessions, benchmarkable, capability_manifest, classify_schedule,
    coverage_summary, plan_record,
)
from matmul_inspector.tensor_game_schedule_model import BlockConfig, build_execution_dag, reference_dag
from scripts.run_hardware_aware_schedule_calibration import FROZEN, project_actions

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "artifacts/tensor-game/hardware-aware-schedule-calibration-v4"
ROOT = REPO / "artifacts/tensor-game"
SIZES = (128, 256, 512, 1024)
SEEDS = (11, 23, 37)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows() -> list[dict]:
    verify_artifact_hashes(SOURCE)
    rows = [json.loads(line) for line in (SOURCE / "candidate_schedules.jsonl").read_text().splitlines() if line]
    if len(rows) != 156 or Counter((row["algorithm"], row["matrix_size"]) for row in rows) != {
        (algorithm, size): count for size in SIZES for algorithm, count in
        (("conventional", 10), ("strassen", 12), ("project-rank8", 17))}:
        raise ValueError("HA-5 source corpus differs from retained 39-per-size corpus")
    return rows


def make_dag(algorithm: str, size: int, actions):
    config = BlockConfig(size, "fp16", "fp32", "fp16")
    return (build_execution_dag(algorithm, actions, config) if algorithm == "project-rank8"
            else reference_dag(algorithm, config))


def graph_sessions(new: dict | None, audits: list[dict] | None = None) -> list[dict]:
    sessions = []
    old = read_json(ROOT / "hardware-aware-schedule-mechanism-v1/timing_summary.json")
    sources = [("HA4-A", old["graph_pairs"])]
    for index in range(1, 5):
        validation = read_json(ROOT / f"hardware-aware-schedule-calibration-v{index}/gpu_validation.json")
        sources.append((f"HA5-v{index}", validation["graph_pairs"]))
    for index, audit in enumerate(audits or [], 1):
        sources.append((f"HA6-audit{index}", audit["graph_pairs"]))
    if new and isinstance(new.get("graph_pairs"), list):
        sources.append(("HA6-final", new["graph_pairs"]))
    for source, pairs in sources:
        for pair in pairs:
            sessions.append({"session_id": source, "matrix_size": pair["matrix_size"],
                             "median_stored_over_recomputed_ratio": pair["median_stored_over_recomputed_ratio"],
                             "ratio_bootstrap_95pct": pair["ratio_bootstrap_95pct"],
                             "source": ("hardware-aware-schedule-mechanism-v1" if source == "HA4-A" else
                                        f"hardware-aware-schedule-calibration-v{source[-1]}" if source.startswith("HA5") else
                                        f"graph_development_audit/session_{source.removeprefix('HA6-audit')}.json"
                                        if source.startswith("HA6-audit") else "graph_replication.json")})
    return sessions


def gpu_worker(output: Path, rows: list[dict]) -> None:
    import torch
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_gpu_runtime import (
        allocate_workspace, execute_selected, expected_kernel_names,
    )
    from matmul_inspector.tensor_game_schedule_lowering_runtime import allocate_materialized, execute_materialized
    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture
    from scripts.run_hardware_aware_schedule_mechanism_study import frozen_lowering

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --skip-gpu for CPU-only structural evidence")
    actions = project_actions(FROZEN)
    checks, topology, smoke, compiler_metadata = [], [], [], []
    lowerings = {"conventional": reference_lowering("conventional"),
                 "strassen": reference_lowering("strassen"), "project-rank8": frozen_lowering()}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row in rows:
            dag = make_dag(row["algorithm"], row["matrix_size"], actions)
            schedule = _schedule(row["schedule"])
            plan = classify_schedule(dag, schedule, lowering_name=row["lowering_name"],
                                     static_metrics=row["metrics"], calibration_features=row["features"])
            if plan.mapping_status == "UNSUPPORTED":
                continue
            if plan.mapping_status == "EXACT_MAPPING":
                workspace = allocate_materialized(dag)
                run = lambda left, right: execute_materialized(dag, schedule, plan, left, right, workspace)
                expected = [launch["kernel"] for launch in plan.launch_sequence]
            else:
                mode = {"conventional_reference": "reference", "strassen_reference": "reference",
                        "rank8_stored": "stored_form", "rank8_recomputed": "recomputed_form"}[row["lowering_name"]]
                workspace = allocate_workspace(lowerings[row["algorithm"]], row["matrix_size"], mode)
                run = lambda left, right: execute_selected(lowerings[row["algorithm"]], left, right, workspace, mode)
                expected = expected_kernel_names(len(dag.terms), mode)
            for seed in SEEDS:
                torch.manual_seed(seed)
                left = torch.randn((row["matrix_size"], row["matrix_size"]), device="cuda", dtype=torch.float16)
                right = torch.randn_like(left)
                actual = run(left, right)
                reference = left @ right
                torch.cuda.synchronize()
                error = (actual - reference).abs().float()
                checks.append({"schedule_hash": plan.schedule_hash, "lowering_plan_hash": plan.lowering_plan_hash,
                               "algorithm": plan.algorithm, "matrix_size": row["matrix_size"], "seed": seed,
                               "rtol": 0.005, "atol": 0.05, "max_absolute_error": float(error.max()),
                               "mean_absolute_error": float(error.mean()),
                               "passed": bool(torch.allclose(actual, reference, rtol=0.005, atol=0.05))})
            correct = benchmarkable(plan, checks)
            # Profile a warm implementation. These are actual device launch events,
            # not Python call counts or timing samples.
            run(left, right)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
                run(left, right)
                torch.cuda.synchronize()
            names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
            matches = len(names) == len(expected) and all(name.startswith(want) for name, want in zip(names, expected))
            topology.append({"schedule_hash": plan.schedule_hash, "lowering_plan_hash": plan.lowering_plan_hash,
                             "algorithm": plan.algorithm, "matrix_size": row["matrix_size"],
                             "family_id": plan.family_id, "mapping_status": plan.mapping_status,
                             "expected_launch_count": len(expected), "actual_launch_count": len(names),
                             "expected_kernel_names": expected, "actual_kernel_names": names,
                             "expected_roles": [launch["role"] for launch in plan.launch_sequence],
                             "topology_agrees": matches,
                             "correctness_status": "PASS" if correct else "LOWERING_INCORRECT"})
            if plan.mapping_status == "EXACT_MAPPING" and not matches:
                raise RuntimeError(f"exact topology mismatch for {plan.schedule_hash}")
            if (correct and row["algorithm"] == "project-rank8" and row["matrix_size"] == 256
                    and row["policy"] == "MATERIALIZE_ALL"):
                captured = []
                execute_materialized(dag, schedule, plan, left, right, workspace, capture=captured)
                first_by_role = {launch["role"]: launch["compiled"] for launch in reversed(captured)}
                for role, compiled in sorted(first_by_role.items()):
                    compiler_metadata.append(_compiler_capture(compiled, f"materialized_{role.lower()}_256",
                                                               output / "compiler"))
            if correct and row["matrix_size"] == 256 and (row["policy"] == "MATERIALIZE_ALL" or row["lowering_name"]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(5):
                    start.record(); run(left, right); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end))
                smoke.append({"schedule_hash": plan.schedule_hash, "family_id": plan.family_id,
                              "mapping_status": plan.mapping_status, "matrix_size": 256,
                              "event_ms": times, "status": "SMOKE_ONLY_NOT_RANKING_EVIDENCE"})
        write_rows(output / "correctness.jsonl", checks)
        write_rows(output / "launch_topology.jsonl", topology)
        write_json(output / "representative_gpu_smoke.json", {"candidates": smoke,
                   "note": "Five unpaired operations per path; execution smoke, not a schedule ranking."})
        write_json(output / "compiler_artifacts_manifest.json", compiler_metadata)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--audit-graph-artifact", type=Path, action="append", default=[],
                        help="Retain and aggregate preliminary HA-6 graph sessions; may be repeated")
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.gpu_worker:
        gpu_worker(output, source_rows())
        return 0
    if output.exists():
        parser.error("output directory already exists")
    audit_sources = [path.resolve() for path in args.audit_graph_artifact]
    if len(set(audit_sources)) != len(audit_sources):
        parser.error("duplicate audit artifact")
    audits = []
    for path in audit_sources:
        verify_artifact_hashes(path)
        audit = read_json(path / "graph_replication.json")
        if len(audit.get("graph_pairs", [])) != 4 or any(not row["passed"] for row in audit["correctness"]):
            raise ValueError(f"incomplete audit graph session: {path}")
        audits.append(audit)
    rows = source_rows()
    actions = project_actions(FROZEN)
    plans = []
    classifications = []
    for row in rows:
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        plan = classify_schedule(dag, _schedule(row["schedule"]), lowering_name=row["lowering_name"],
                                 static_metrics=row["metrics"], calibration_features=row["features"])
        plans.append(plan)
        classifications.append({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                                "matrix_size": row["matrix_size"], "policy": row["policy"],
                                "family_id": plan.family_id, "support_status": plan.support_status,
                                "mapping_status": plan.mapping_status,
                                "reasons": plan.unsupported_features, "lowering_plan_hash": plan.lowering_plan_hash,
                                "static_metrics_hash": plan.static_metrics_hash,
                                "calibration_features_hash": plan.calibration_features_hash})
    output.mkdir(parents=True)
    write_json(output / "config.json", {"source": str(SOURCE.relative_to(REPO)), "matrix_sizes": SIZES,
                                      "correctness_seeds": SEEDS, "gpu_enabled": not args.skip_gpu,
                                      "audit_graph_session_count": len(audits)})
    write_json(output / "provenance.json", {"source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_hash_manifest_sha256": sha(SOURCE / "hash_manifest.json"),
        "audit_source_hashes": [sha(path / "graph_replication.json") for path in audit_sources]})
    if audits:
        audit_dir = output / "graph_development_audit"
        audit_dir.mkdir()
        for index, path in enumerate(audit_sources, 1):
            shutil.copyfile(path / "graph_replication.json", audit_dir / f"session_{index}.json")
            shutil.copyfile(path / "gpu_validation_raw.jsonl", audit_dir / f"session_{index}_raw.jsonl")
    write_rows(output / "schedule_classification.jsonl", classifications)
    write_rows(output / "lowering_plans.jsonl", [plan_record(plan) for plan in plans])
    capabilities = capability_manifest()
    write_json(output / "lowering_capabilities.json", capabilities)
    write_json(output / "lowering_families.json", capabilities["families"])
    coverage = {str(size): coverage_summary([plan for row, plan in zip(rows, plans) if row["matrix_size"] == size])
                for size in SIZES}
    write_json(output / "coverage_summary.json", coverage)
    reasons = Counter(reason for plan in plans if plan.mapping_status == "UNSUPPORTED"
                      for reason in plan.unsupported_features)
    write_json(output / "unsupported_reasons.json", reasons)
    if args.skip_gpu:
        write_rows(output / "correctness.jsonl", [])
        write_rows(output / "launch_topology.jsonl", [])
        write_json(output / "representative_gpu_smoke.json", {"status": "SKIPPED_NO_GPU"})
        write_json(output / "compiler_artifacts_manifest.json", {"status": "SKIPPED_NO_GPU"})
        write_json(output / "graph_replication.json", {"status": "SKIPPED_NO_GPU"})
    else:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--gpu-worker", "--output", str(output)],
                       cwd=REPO, check=True)
        # A second fresh process reuses the established HA-5 graph protocol.
        subprocess.run([sys.executable, str(REPO / "scripts/run_hardware_aware_schedule_calibration.py"),
                        "--gpu-worker", "--output", str(output)], cwd=REPO, check=True)
        new = read_json(output / "gpu_validation.json")
        write_json(output / "graph_replication.json", {"process_id": new["process_id"],
                   "started_utc": new["started_utc"], "ended_utc": new["ended_utc"],
                   "hardware": new["hardware"], "software": new["software"],
                   "protocol": new["protocol"], "status": new["status"],
                   "correctness": new["correctness"], "graph_pairs": new["graph_pairs"],
                   "eager_pairs": new["eager_pairs"], "timing_summaries": new["timing_summaries"],
                   "raw_sha256": sha(output / "gpu_validation_raw.jsonl")})
    new_graph = read_json(output / "graph_replication.json") if not args.skip_gpu else None
    graph_summary = aggregate_graph_sessions(graph_sessions(new_graph, audits))
    write_json(output / "graph_evidence_summary.json", graph_summary)
    correctness = [json.loads(line) for line in (output / "correctness.jsonl").read_text().splitlines() if line]
    topology = [json.loads(line) for line in (output / "launch_topology.jsonl").read_text().splitlines() if line]
    passing = {plan.schedule_hash for plan in plans if benchmarkable(plan, correctness)}
    failed = {row["schedule_hash"] for row in correctness if not row["passed"]}
    for record in classifications:
        record["validation_status"] = ("LOWERING_INCORRECT" if record["schedule_hash"] in failed else
                                       "PASS" if record["schedule_hash"] in passing else
                                       "UNSUPPORTED" if record["mapping_status"] == "UNSUPPORTED" else "NOT_RUN")
    write_rows(output / "schedule_classification.jsonl", classifications)
    for size in SIZES:
        subset = [row for row in classifications if row["matrix_size"] == size]
        coverage[str(size)]["validated_exact"] = sum(row["mapping_status"] == "EXACT_MAPPING" and
                                                   row["validation_status"] == "PASS" for row in subset)
        coverage[str(size)]["validated_approximate"] = sum(row["mapping_status"] == "APPROXIMATE_MAPPING" and
                                                         row["validation_status"] == "PASS" for row in subset)
        coverage[str(size)]["lowering_incorrect"] = sum(row["validation_status"] == "LOWERING_INCORRECT"
                                                    for row in subset)
        for algorithm, algorithm_coverage in coverage[str(size)]["by_algorithm"].items():
            algorithm_rows = [row for row in subset if row["algorithm"] == algorithm]
            algorithm_coverage["validated_exact"] = sum(row["mapping_status"] == "EXACT_MAPPING" and
                row["validation_status"] == "PASS" for row in algorithm_rows)
            algorithm_coverage["validated_approximate"] = sum(row["mapping_status"] == "APPROXIMATE_MAPPING" and
                row["validation_status"] == "PASS" for row in algorithm_rows)
            algorithm_coverage["lowering_incorrect"] = sum(row["validation_status"] == "LOWERING_INCORRECT"
                for row in algorithm_rows)
    write_json(output / "coverage_summary.json", coverage)
    exact_passing = sum(plan.mapping_status == "EXACT_MAPPING" and plan.schedule_hash in passing for plan in plans)
    blockers = []
    if exact_passing < 20 * len(SIZES):
        blockers.append("faithful validated coverage is below 20/39 per size")
    if not args.skip_gpu and any(row.get("topology_agrees") is False for row in topology):
        blockers.append("at least one mapped topology disagrees")
    if args.skip_gpu:
        blockers.append("GPU correctness, topology, and fresh graph session not run")
    criteria = {name: True for name in ("candidate_generation", "schedule_legality", "static_characterization",
        "calibration", "execution_mode_specific_ranking", "lowerability_query", "automatic_lowering_supported_families",
        "correctness_gating", "gpu_benchmarking", "compiler_profiler_evidence", "failure_recording",
        "provenance", "confidence_downgrade", "candidate_reselection_support")}
    criteria["automatic_lowering_supported_families"] = exact_passing > 0
    criteria["correctness_gating"] = bool(correctness)
    criteria["gpu_benchmarking"] = bool(read_json(output / "representative_gpu_smoke.json").get("candidates"))
    readiness = {"status": "NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE", "decision": "LOWERING_COVERAGE_STILL_LIMITED",
                 "criteria": criteria, "blockers": blockers, "validated_exact_schedule_records": exact_passing,
                 "graph_sessions_per_size": {row["matrix_size"]: row["session_count"] for row in graph_summary}}
    write_json(output / "closed_loop_readiness.json", readiness)
    write_json(output / "aggregate_summary.json", {"coverage": coverage, "correctness_checks": len(correctness),
        "passing_schedule_hashes": len(passing), "topology_records": len(topology),
        "graph_confidence": {row["matrix_size"]: row["confidence"] for row in graph_summary},
        "readiness": readiness})
    (output / "README.md").write_text("# HA-Schedule-6 retained evidence\n\nClassification and plans link every HA-5 schedule hash to a bounded template. Exact coverage requires correctness and topology gates. GPU smoke timings are not ranking evidence. Graph replication uses the unchanged HA-5 protocol in a new process. See docs/hardware_aware_schedule_lowering.md.\n")
    write_json(output / "hash_manifest.json", {str(path.relative_to(output)): sha(path)
                                               for path in sorted(output.rglob("*"))
                                               if path.is_file() and path.name != "hash_manifest.json"})
    print(json.dumps({"output": str(output), "coverage_128": coverage["128"], "readiness": readiness["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
