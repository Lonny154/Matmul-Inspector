#!/usr/bin/env python3
"""Retained HA-7 mixed-group coverage, correctness, topology, and bounded smoke."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule, verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_lowering import classify_schedule, plan_record
from matmul_inspector.tensor_game_schedule_mixed_lowering import (
    FAMILY_MIXED, classify_mixed_schedule, mixed_capability_manifest,
)
from scripts.run_hardware_aware_schedule_lowering_study import (
    FROZEN, REPO, make_dag, project_actions, source_rows,
)

SIZES = (128, 256, 512, 1024)
SEEDS = (11, 23, 37)
EXTRA_SEEDS = (41, 53)
PREVIOUS = REPO / "artifacts/tensor-game/hardware-aware-schedule-lowering-v1"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_patterns(records):
    signatures = Counter()
    blockers = Counter()
    mixed_by_size = Counter()
    isolated_blockers = Counter()
    for row, dag, schedule, old, new in records:
        if old.mapping_status != "UNSUPPORTED":
            continue
        nodes = {node.node_id: node for node in dag.nodes}
        found = False
        for group in schedule.groups:
            ops = group.operations
            if any(nodes[op.node_id].operation == "MATMUL" for op in ops) and any(
                    nodes[op.node_id].operation == "ACCUMULATE" for op in ops):
                found = True
                signature = (tuple(sorted(op.location for op in ops if nodes[op.node_id].operation == "LINEAR_FORM")),
                             tuple(op.location for op in ops if nodes[op.node_id].operation == "MATMUL"),
                             tuple(sorted(op.location for op in ops if nodes[op.node_id].operation == "ACCUMULATE")),
                             sum(nodes[op.node_id].operation == "OUTPUT" for op in ops))
                if row["matrix_size"] == 128:
                    signatures[str(signature)] += 1
        if found:
            mixed_by_size[row["matrix_size"]] += 1
        for reason in new.unsupported_features:
            blockers[reason] += 1
        if new.mapping_status == "UNSUPPORTED" and len(new.unsupported_features) == 1:
            isolated_blockers[new.unsupported_features[0]] += 1
    return {"unsupported_with_mixed_group_by_size": dict(mixed_by_size),
            "distinct_n128_group_signatures": len(signatures),
            "n128_group_signatures": dict(signatures),
            "isolated_remaining_blockers": dict(isolated_blockers),
            "all_remaining_blocker_occurrences": dict(blockers)}


def gpu_validation(output, records):
    import torch
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_mixed_runtime import allocate_mixed, execute_mixed
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture
    from scripts.run_hardware_aware_schedule_mechanism_study import frozen_lowering

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --skip-gpu for structural evidence")
    checks, topology, resources, smoke = [], [], [], []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, dag, schedule, _, plan in records:
            if plan.family_id != FAMILY_MIXED:
                continue
            workspace = allocate_mixed(dag, schedule, plan)
            def run(left, right, capture=None):
                return execute_mixed(dag, schedule, plan, left, right, workspace, capture=capture)
            representative = row["matrix_size"] == 256 and row["policy"] == "FUSE_OUTPUT_RECONSTRUCTION"
            seeds = (*SEEDS, *EXTRA_SEEDS) if representative else SEEDS
            for seed in seeds:
                torch.manual_seed(seed)
                left = torch.randn((row["matrix_size"], row["matrix_size"]), device="cuda", dtype=torch.float16)
                right = torch.randn_like(left)
                actual = run(left, right)
                reference = left @ right
                torch.cuda.synchronize()
                error = (actual - reference).abs().float()
                checks.append({"schedule_hash": plan.schedule_hash, "lowering_plan_hash": plan.lowering_plan_hash,
                               "algorithm": row["algorithm"], "matrix_size": row["matrix_size"],
                               "seed": seed, "rtol": 0.005, "atol": 0.05,
                               "max_absolute_error": float(error.max()),
                               "mean_absolute_error": float(error.mean()),
                               "passed": bool(torch.allclose(actual, reference, rtol=0.005, atol=0.05))})
            run(left, right)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
                run(left, right)
                torch.cuda.synchronize()
            actual_names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
            expected_names = [item["kernel"] for item in plan.launch_sequence]
            agrees = len(actual_names) == len(expected_names) and all(
                actual.startswith(expected) for actual, expected in zip(actual_names, expected_names))
            topology.append({"schedule_hash": plan.schedule_hash, "lowering_plan_hash": plan.lowering_plan_hash,
                             "algorithm": row["algorithm"], "matrix_size": row["matrix_size"],
                             "expected_launch_count": len(expected_names), "actual_launch_count": len(actual_names),
                             "expected_roles": [item["role"] for item in plan.launch_sequence],
                             "expected_kernel_names": expected_names, "actual_kernel_names": actual_names,
                             "topology_agrees": agrees,
                             "product_hbm_workspace_count": sum(item["role"] == FAMILY_MIXED and
                                 item["fused_gemm_id"] in workspace["values"] for item in plan.launch_sequence)})
            if representative:
                captured = []
                run(left, right, captured)
                fused = next(item["compiled"] for item in captured if item["role"] == FAMILY_MIXED)
                resource = _compiler_capture(fused, f"mixed_{row['algorithm']}_256", output / "compiler")
                resource.update({"schedule_hash": plan.schedule_hash, "algorithm": row["algorithm"],
                                 "matrix_size": 256, "kernel_count": len(captured)})
                resources.append(resource)
            if representative and all(item["passed"] for item in checks if item["schedule_hash"] == plan.schedule_hash):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(5):
                    start.record(); run(left, right); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end))
                smoke.append({"candidate": f"{row['algorithm']}_mixed", "schedule_hash": plan.schedule_hash,
                              "matrix_size": 256, "event_ms": times})
        lower = frozen_lowering()
        for mode in ("stored_form", "recomputed_form"):
            workspace = allocate_workspace(lower, 256, mode)
            torch.manual_seed(11)
            left = torch.randn((256, 256), device="cuda", dtype=torch.float16)
            right = torch.randn_like(left)
            run = lambda: execute_selected(lower, left, right, workspace, mode)
            actual = run()
            correct = bool(torch.allclose(actual, left @ right, rtol=0.005, atol=0.05))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = []
            if correct:
                for _ in range(5):
                    start.record(); run(); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end))
            smoke.append({"candidate": f"rank8_{mode}", "matrix_size": 256,
                          "correct": correct, "event_ms": times})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    write_rows(output / "correctness.jsonl", checks)
    write_rows(output / "launch_topology.jsonl", topology)
    write_rows(output / "compiler_resources.jsonl", resources)
    write_json(output / "performance_smoke.json", {"status": "SMOKE_ONLY_NOT_RANKING_EVIDENCE",
                                                "candidates": smoke})
    write_json(output / "gpu_environment.json", {"device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton_version": __import__("triton").__version__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--refresh-derived", action="store_true",
                        help="Recompute CPU classifications from intact retained GPU evidence")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.refresh_derived:
        parser.error("output directory already exists")
    if args.refresh_derived:
        if args.skip_gpu or not output.is_dir():
            parser.error("--refresh-derived requires an existing GPU study directory")
        verify_artifact_hashes(output)
    rows = source_rows()
    actions = project_actions(FROZEN)
    records = []
    for row in rows:
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        old = classify_schedule(dag, schedule, **kwargs)
        new = classify_mixed_schedule(dag, schedule, **kwargs)
        records.append((row, dag, schedule, old, new))
    if args.refresh_derived:
        current_hashes = {new.lowering_plan_hash for _, _, _, _, new in records if new.family_id == FAMILY_MIXED}
        for filename in ("correctness.jsonl", "launch_topology.jsonl"):
            if any(item["lowering_plan_hash"] not in current_hashes for item in read_rows(output / filename)):
                raise ValueError(f"retained {filename} does not match current mixed plans")
    if not args.refresh_derived:
        output.mkdir(parents=True)
        write_json(output / "config.json", {"source_corpus": "hardware-aware-schedule-calibration-v4",
                                      "baseline": str(PREVIOUS.relative_to(REPO)),
                                      "matrix_sizes": SIZES, "correctness_seeds": SEEDS,
                                      "representative_extra_seeds": EXTRA_SEEDS,
                                      "rtol": 0.005, "atol": 0.05,
                                      "gpu_enabled": not args.skip_gpu})
        write_json(output / "provenance.json", {"source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_hash_manifest_sha256": sha(REPO / "artifacts/tensor-game/hardware-aware-schedule-calibration-v4/hash_manifest.json"),
        "baseline_hash_manifest_sha256": sha(PREVIOUS / "hash_manifest.json")})
    else:
        previous_provenance = read_json(output / "provenance.json")
        previous_provenance["derived_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        previous_provenance["derived_reason_model_version"] = "ha-schedule-7-v1"
        write_json(output / "provenance.json", previous_provenance)
    write_json(output / "pre_coverage.json", read_json(PREVIOUS / "coverage_summary.json"))
    write_json(output / "mixed_group_patterns.json", group_patterns(records))
    write_json(output / "lowering_family.json", mixed_capability_manifest())
    write_rows(output / "lowering_plans.jsonl", [plan_record(new) for _, _, _, _, new in records])
    if args.refresh_derived:
        pass  # Correctness, topology, resources, and smoke are retained intact.
    elif args.skip_gpu:
        write_rows(output / "correctness.jsonl", [])
        write_rows(output / "launch_topology.jsonl", [])
        write_rows(output / "compiler_resources.jsonl", [])
        write_json(output / "performance_smoke.json", {"status": "SKIPPED_NO_GPU"})
    else:
        gpu_validation(output, records)
    checks = read_rows(output / "correctness.jsonl")
    topology = {item["schedule_hash"]: item for item in read_rows(output / "launch_topology.jsonl")}
    previous_status = {item["schedule_hash"]: item["validation_status"] for item in
                       read_rows(PREVIOUS / "schedule_classification.jsonl")}
    checks_by_hash = defaultdict(list)
    for item in checks:
        checks_by_hash[item["schedule_hash"]].append(item)
    classification = []
    for row, _, _, old, new in records:
        if new.family_id == FAMILY_MIXED:
            tested = checks_by_hash[new.schedule_hash]
            status = ("NOT_RUN" if not tested else "PASS" if len(tested) >= 3 and
                      all(item["passed"] for item in tested) and
                      topology[new.schedule_hash]["topology_agrees"] and
                      topology[new.schedule_hash]["product_hbm_workspace_count"] == 0 else "LOWERING_INCORRECT")
        else:
            status = previous_status[new.schedule_hash]
        classification.append({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                               "matrix_size": row["matrix_size"], "policy": row["policy"],
                               "old_mapping_status": old.mapping_status,
                               "family_id": new.family_id, "mapping_status": new.mapping_status,
                               "validation_status": status, "lowering_plan_hash": new.lowering_plan_hash,
                               "reasons": new.unsupported_features})
    write_rows(output / "classification_after.jsonl", classification)
    coverage = {}
    for size in SIZES:
        subset = [item for item in classification if item["matrix_size"] == size]
        def counts(items):
            return {"legal": len(items), "structural_exact": sum(x["mapping_status"] == "EXACT_MAPPING" for x in items),
                    "exact_passing": sum(x["mapping_status"] == "EXACT_MAPPING" and x["validation_status"] == "PASS" for x in items),
                    "approximate_passing": sum(x["mapping_status"] == "APPROXIMATE_MAPPING" and x["validation_status"] == "PASS" for x in items),
                    "unsupported": sum(x["mapping_status"] == "UNSUPPORTED" for x in items),
                    "lowering_incorrect": sum(x["validation_status"] == "LOWERING_INCORRECT" for x in items)}
        coverage[str(size)] = {**counts(subset), "by_algorithm": {algorithm: counts([x for x in subset if x["algorithm"] == algorithm])
                              for algorithm in ("conventional", "strassen", "project-rank8")}}
    write_json(output / "coverage_summary.json", coverage)
    reason_counts = Counter(reason for item in classification if item["mapping_status"] == "UNSUPPORTED"
                            for reason in item["reasons"])
    write_json(output / "remaining_unsupported_reasons.json", dict(reason_counts))
    new_exact = [item for item in classification if item["family_id"] == FAMILY_MIXED]
    exact_passing = sum(item["exact_passing"] for item in coverage.values())
    decision = ("MIXED_GROUP_COVERAGE_UNLOCKED" if exact_passing >= 80 else
                "NUMERICAL_LIMITED" if sum(x["validation_status"] == "LOWERING_INCORRECT" for x in new_exact) > len(new_exact) // 2 else
                "LOWERING_BACKEND_STILL_FRAGMENTED")
    blockers = []
    if exact_passing < 80:
        blockers.append("exact-passing coverage is below 20/39 per size or 50% overall")
    if reason_counts:
        blockers.append("remaining legal schedules need additional lowering semantics")
    if any(item["validation_status"] == "LOWERING_INCORRECT" for item in classification):
        blockers.append("Strassen exact mappings fail fixed-tolerance correctness at N=512/1024")
    if args.skip_gpu:
        blockers.append("new mixed kernels were not GPU-validated")
    readiness = {"status": "NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE", "decision": decision,
                 "validated_exact_schedule_records": exact_passing,
                 "recommended_next_backend_capability": "bounded HBM-backed mixed-group contract",
                 "criteria": {name: True for name in ("schedule_generation", "legality_validation", "static_metrics",
                    "calibrated_ranking", "eager_graph_separation", "lowerability_query", "lowering_plan_generation",
                    "generic_family_lowering", "correctness_gate", "bounded_benchmarking", "compiler_profiler_evidence",
                    "failure_retention", "provenance", "confidence_downgrade", "candidate_reselection")},
                 "blockers": blockers}
    if args.skip_gpu:
        for key in ("correctness_gate", "bounded_benchmarking", "compiler_profiler_evidence"):
            readiness["criteria"][key] = False
    write_json(output / "closed_loop_readiness.json", readiness)
    write_json(output / "aggregate_summary.json", {"decision": decision, "coverage": coverage,
        "new_exact_structural": len(new_exact),
        "new_exact_passing": sum(item["validation_status"] == "PASS" for item in new_exact),
        "new_exact_incorrect": sum(item["validation_status"] == "LOWERING_INCORRECT" for item in new_exact),
        "new_approximate": sum(old.mapping_status == "UNSUPPORTED" and new.mapping_status == "APPROXIMATE_MAPPING"
                               for _, _, _, old, new in records),
        "correctness_checks": len(checks), "topology_records": len(topology),
        "dominant_remaining_reason": reason_counts.most_common(1), "readiness": readiness})
    (output / "README.md").write_text("# HA-Schedule-7 retained evidence\n\nOne bounded mixed GEMM/reconstruction template is evaluated against all 156 retained HA-5 schedule records. Exact-passing coverage requires fixed-tolerance correctness and profiler launch-topology agreement. Previous HA-6 validations are reused by hash. Smoke timings are not ranking evidence. See docs/hardware_aware_mixed_group_lowering.md.\n")
    write_json(output / "hash_manifest.json", {str(path.relative_to(output)): sha(path)
                                               for path in sorted(output.rglob("*"))
                                               if path.is_file() and path.name != "hash_manifest.json"})
    print(json.dumps({"output": str(output), "decision": decision,
                      "exact_passing": exact_passing, "new_exact": len(new_exact)}, sort_keys=True))


if __name__ == "__main__":
    main()
