#!/usr/bin/env python3
"""Classify and validate bounded intentional HBM lifetimes on retained schedules."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule, verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_hbm_lowering import (
    FAMILY_HBM, HBMBackedPlan, classify_hbm_schedule, hbm_capability_manifest,
    reference_transition_checks, validate_hbm_plan,
)
from matmul_inspector.tensor_game_schedule_mixed_lowering import classify_mixed_schedule
from scripts.run_hardware_aware_schedule_lowering_study import (
    FROZEN, REPO, make_dag, project_actions, source_rows,
)

SIZES = (128, 256, 512, 1024)
SEEDS = (11, 23, 37)
EXTRA_SEEDS = (41, 53)
SOURCE = REPO / "artifacts/tensor-game/hardware-aware-schedule-calibration-v4"
PREVIOUS = REPO / "artifacts/tensor-game/hardware-aware-mixed-group-lowering-v1"
HA6 = REPO / "artifacts/tensor-game/hardware-aware-schedule-lowering-v1"


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(plan):
    return plan.schedule_plan if isinstance(plan, HBMBackedPlan) else plan


def build_records():
    verify_artifact_hashes(SOURCE)
    verify_artifact_hashes(PREVIOUS)
    verify_artifact_hashes(HA6)
    actions = project_actions(FROZEN)
    records = []
    for row in source_rows():
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        old = classify_mixed_schedule(dag, schedule, **kwargs)
        new = classify_hbm_schedule(dag, schedule, **kwargs)
        records.append((row, dag, schedule, old, new))
    if len(records) != 156:
        raise ValueError("retained corpus is not 156 schedule-size records")
    return records


def lifetime_patterns(records):
    by_size = Counter()
    combinations = Counter()
    ordered_group_shapes = Counter()
    theoretical = Counter()
    for row, dag, schedule, old, new in records:
        if old.mapping_status != "UNSUPPORTED":
            continue
        nodes = {node.node_id: node for node in dag.nodes}
        producers = {op.node_id: index for index, group in enumerate(schedule.groups)
                     for op in group.operations if op.location == "HBM"}
        types = set()
        for index, group in enumerate(schedule.groups):
            if row["matrix_size"] == 128 and len(group.operations) > 1:
                shape = tuple((nodes[op.node_id].operation, op.location) for op in group.operations)
                ordered_group_shapes[str(shape)] += 1
            for op in group.operations:
                for dependency in nodes[op.node_id].dependencies:
                    if dependency in producers and producers[dependency] < index:
                        types.add({"LINEAR_FORM": "PREPARED_FORM", "MATMUL": "GEMM_PRODUCT",
                                   "ACCUMULATE": "PARTIAL_OUTPUT"}[nodes[dependency].operation])
        key = tuple(sorted(types))
        combinations[(row["matrix_size"], key)] += 1
        by_size[row["matrix_size"]] += 1
        if isinstance(new, HBMBackedPlan):
            theoretical[row["matrix_size"]] += 1
    return {"unsupported_before_by_size": dict(by_size),
            "cross_group_class_combinations": [
                {"matrix_size": size, "value_classes": kinds, "count": count}
                for (size, kinds), count in sorted(combinations.items())],
            "distinct_n128_ordered_multi_operation_shapes": len(ordered_group_shapes),
            "n128_ordered_multi_operation_shapes": dict(ordered_group_shapes),
            "bounded_structural_gain_by_size": dict(theoretical)}


def selected_schedule_coverage(classification):
    """Apply new correctness status to frozen HA-5 selections, without reranking."""
    exact = {row["schedule_hash"] for row in classification if row["mapping_status"] == "EXACT_MAPPING"
             and row["validation_status"] == "PASS"}
    comparison = read_json(SOURCE / "ranking_comparison.json")
    static = read_json(SOURCE / "static_rankings.json")
    eager = read_json(SOURCE / "eager_calibrated_rankings.json")
    graph = read_json(SOURCE / "graph_calibrated_rankings.json")
    entries = []
    for item in comparison:
        algorithm, size = item["algorithm"], item["matrix_size"]
        local_static = sorted((row for row in static if row["algorithm"] == algorithm and
                               row["matrix_size"] == size), key=lambda row: (row["static_rank"], row["schedule_hash"]))
        selections = {"static_top1": [row["schedule_hash"] for row in local_static[:1]],
                      "static_top3": [row["schedule_hash"] for row in local_static[:3]],
                      "static_pareto": item["pareto_hashes"]}
        for name, source in (("eager_old_mapped_top1", eager), ("graph_old_mapped_top1", graph)):
            chosen = [row for row in source if row["algorithm"] == algorithm and row["matrix_size"] == size
                      and row.get("calibrated_rank") == 1]
            selections[name] = [row["schedule_hash"] for row in chosen]
        for name, hashes in selections.items():
            entries.append({"algorithm": algorithm, "matrix_size": size, "selection": name,
                            "selected_count": len(hashes), "exact_passing_count": sum(h in exact for h in hashes),
                            "selected_hashes": hashes, "exact_passing_hashes": [h for h in hashes if h in exact],
                            "scope": "HA5_OLD_MAPPED_SUBSET_ONLY" if name.startswith(("eager_", "graph_"))
                            else "ALL_RETAINED_LEGAL_STATIC"})
    totals = {}
    for name in sorted({entry["selection"] for entry in entries}):
        subset = [entry for entry in entries if entry["selection"] == name]
        totals[name] = {"selected": sum(entry["selected_count"] for entry in subset),
                        "exact_passing": sum(entry["exact_passing_count"] for entry in subset)}
    return {"entries": entries, "totals": totals,
            "caveat": "HA-5 eager/graph ranks cover only old measured mappings. New mappings are not recalibrated or ranked here."}


def backend_architecture_decision(classification):
    remaining = [row for row in classification if row["matrix_size"] == 128 and
                 row["mapping_status"] == "UNSUPPORTED"]
    combinations = {tuple(sorted(row["reasons"])) for row in remaining}
    interacting = sum(len(row["reasons"]) >= 2 for row in remaining)
    decision = ("GENERAL_LOWERING_IR_NEEDED" if len(combinations) >= 4 and interacting >= 6 else
                "TEMPLATE_BACKEND_STILL_VIABLE")
    return {"decision": decision, "remaining_n128_schedules": len(remaining),
            "distinct_blocker_combinations": len(combinations),
            "schedules_with_multiple_blockers": interacting,
            "criterion": "general IR if >=4 blocker combinations and >=6/size schedules have interacting blockers",
            "recommendation": "Design a small general schedule-to-kernel IR before more family templates" if
            decision == "GENERAL_LOWERING_IR_NEEDED" else "Study the largest bounded remaining family"}


def validated_new_status(checks, topology, buffer_audit):
    if not checks:
        return "NOT_RUN"
    return ("PASS" if len({item["seed"] for item in checks}) >= 3 and
            all(item["passed"] for item in checks) and topology.get("topology_agrees") is True and
            buffer_audit.get("status") == "PASS" else "LOWERING_INCORRECT")


def gpu_validation(output, records):
    import torch
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_hbm_runtime import (
        allocate_hbm, audit_hbm_workspace, execute_hbm,
    )
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture
    from scripts.run_hardware_aware_schedule_mechanism_study import frozen_lowering

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --skip-gpu for structural evidence")
    checks, topology, resources, smoke, audits = [], [], [], [], []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, dag, schedule, _, plan in records:
            if not isinstance(plan, HBMBackedPlan):
                continue
            workspace = allocate_hbm(dag, schedule, plan)
            audits.append({"schedule_hash": row["schedule_hash"], "lowering_plan_hash": plan.lowering_plan_hash,
                           "algorithm": row["algorithm"], "matrix_size": row["matrix_size"],
                           **audit_hbm_workspace(dag, schedule, plan, workspace)})
            def run(left, right, capture=None):
                return execute_hbm(dag, schedule, plan, left, right, workspace, capture=capture)
            # These frozen N=256 hashes isolate product/partial and
            # prepared-form paths. The policy selects the Strassen form case.
            representative = row["matrix_size"] == 256 and (
                row["schedule_hash"][:8] in {"e0e8c4bf", "1406976b"} or
                row["policy"] == "FUSE_INPUT_PREPARATION")
            for seed in (*SEEDS, *(EXTRA_SEEDS if representative else ())):
                torch.manual_seed(seed)
                left = torch.randn((row["matrix_size"], row["matrix_size"]), device="cuda", dtype=torch.float16)
                right = torch.randn_like(left)
                actual = run(left, right)
                reference = left @ right
                torch.cuda.synchronize()
                error = (actual - reference).abs().float()
                checks.append({"schedule_hash": row["schedule_hash"], "lowering_plan_hash": plan.lowering_plan_hash,
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
            expected_names = [launch["kernel"] for launch in plan.schedule_plan.launch_sequence]
            agrees = len(actual_names) == len(expected_names) and all(
                actual.startswith(expected) for actual, expected in zip(actual_names, expected_names))
            captured = []
            run(left, right, captured)
            kernel_hashes = [{"group_id": item["group_id"], "role": item["role"],
                              "cubin_sha256": hashlib.sha256(bytes(item["compiled"].asm["cubin"])).hexdigest()}
                             for item in captured if item["role"] in (FAMILY_HBM, "HBM_FINALIZE")]
            topology.append({"schedule_hash": row["schedule_hash"], "lowering_plan_hash": plan.lowering_plan_hash,
                             "algorithm": row["algorithm"], "matrix_size": row["matrix_size"],
                             "expected_launch_count": len(expected_names), "actual_launch_count": len(actual_names),
                             "expected_roles": [launch["role"] for launch in plan.schedule_plan.launch_sequence],
                             "expected_kernel_names": expected_names, "actual_kernel_names": actual_names,
                             "relevant_kernel_hashes": kernel_hashes, "topology_agrees": agrees})
            if representative:
                first = next(item["compiled"] for item in captured if item["role"] == FAMILY_HBM)
                resource = _compiler_capture(first, f"hbm_{row['algorithm']}_256", output / "compiler")
                resource.update({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                                 "matrix_size": 256, "kernel_count": len(captured)})
                resources.append(resource)
            if representative and all(item["passed"] for item in checks if item["schedule_hash"] == row["schedule_hash"]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(5):
                    start.record(); run(left, right); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end))
                smoke.append({"candidate": f"{row['algorithm']}_hbm_{row['schedule_hash'][:8]}",
                              "schedule_hash": row["schedule_hash"], "matrix_size": 256, "event_ms": times})
        anchors = (("conventional_reference", reference_lowering("conventional"), "reference"),
                   ("rank8_stored", frozen_lowering(), "stored_form"),
                   ("rank8_recomputed", frozen_lowering(), "recomputed_form"))
        for name, lowering, mode in anchors:
            workspace = allocate_workspace(lowering, 256, mode)
            torch.manual_seed(11)
            left = torch.randn((256, 256), device="cuda", dtype=torch.float16)
            right = torch.randn_like(left)
            run = lambda: execute_selected(lowering, left, right, workspace, mode)
            actual = run()
            correct = bool(torch.allclose(actual, left @ right, rtol=0.005, atol=0.05))
            times = []
            if correct:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(5):
                    start.record(); run(); end.record(); end.synchronize()
                    times.append(start.elapsed_time(end))
            smoke.append({"candidate": name, "matrix_size": 256, "correct": correct, "event_ms": times})
        import triton
        write_json(output / "gpu_environment.json", {"device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "triton_version": triton.__version__})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    write_rows(output / "correctness.jsonl", checks)
    write_rows(output / "launch_topology.jsonl", topology)
    write_rows(output / "compiler_resources.jsonl", resources)
    write_rows(output / "runtime_buffer_audit.jsonl", audits)
    write_json(output / "performance_smoke.json", {"status": "SMOKE_ONLY_NOT_RANKING_EVIDENCE",
                                                "candidates": smoke})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--refresh-derived", action="store_true",
                        help="Rebuild CPU joins from intact retained GPU evidence")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.refresh_derived:
        parser.error("output directory already exists")
    if args.refresh_derived:
        if args.skip_gpu or not output.is_dir():
            parser.error("--refresh-derived requires an existing GPU study directory")
        verify_artifact_hashes(output)
    records = build_records()
    if args.refresh_derived:
        expected_hashes = {new.lowering_plan_hash for _, _, _, _, new in records
                           if isinstance(new, HBMBackedPlan)}
        for filename in ("correctness.jsonl", "launch_topology.jsonl", "runtime_buffer_audit.jsonl"):
            if any(row["lowering_plan_hash"] not in expected_hashes for row in read_rows(output / filename)):
                raise ValueError(f"retained {filename} does not match current HBM plans")
        provenance = read_json(output / "provenance.json")
        provenance["derived_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["parent_commit"] = provenance.get("parent_commit", provenance["source_commit"])
        provenance["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        write_json(output / "provenance.json", provenance)
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", {"source_corpus": str(SOURCE.relative_to(REPO)),
            "previous_evidence": str(PREVIOUS.relative_to(REPO)), "matrix_sizes": SIZES,
            "correctness_seeds": SEEDS, "representative_extra_seeds": EXTRA_SEEDS,
            "rtol": 0.005, "atol": 0.05, "gpu_enabled": not args.skip_gpu})
        write_json(output / "provenance.json", {"source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_hash_manifest_sha256": sha(SOURCE / "hash_manifest.json"),
            "previous_hash_manifest_sha256": sha(PREVIOUS / "hash_manifest.json"),
            "ha6_hash_manifest_sha256": sha(HA6 / "hash_manifest.json")})
    write_json(output / "pre_coverage.json", read_json(PREVIOUS / "coverage_summary.json"))
    write_json(output / "hbm_lifetime_patterns.json", lifetime_patterns(records))
    write_json(output / "lowering_capabilities.json", hbm_capability_manifest())
    write_rows(output / "lowering_plans.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], "algorithm": row["algorithm"],
        "lowering_plan_hash": new.lowering_plan_hash, "schedule_plan": asdict(_base(new)),
        "hbm_lifetime_count": len(new.hbm_lifetimes) if isinstance(new, HBMBackedPlan) else 0,
        "contribution_order": new.contribution_order if isinstance(new, HBMBackedPlan) else ()}
        for row, _, _, _, new in records])
    write_rows(output / "hbm_lifetimes.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], **asdict(lifetime)}
        for row, _, _, _, new in records if isinstance(new, HBMBackedPlan)
        for lifetime in new.hbm_lifetimes])
    write_rows(output / "buffer_plans.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], "lowering_plan_hash": new.lowering_plan_hash, **asdict(buffer)}
        for row, _, _, _, new in records if isinstance(new, HBMBackedPlan)
        for buffer in new.buffer_plan])
    write_rows(output / "reference_transition_checks.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], "lowering_plan_hash": new.lowering_plan_hash, **check}
        for row, dag, schedule, _, new in records if isinstance(new, HBMBackedPlan)
        for check in reference_transition_checks(dag, schedule, new)])
    if args.refresh_derived:
        pass  # Seed checks, profiler events, resources, and smoke remain intact.
    elif args.skip_gpu:
        for name in ("correctness.jsonl", "launch_topology.jsonl", "compiler_resources.jsonl",
                     "runtime_buffer_audit.jsonl"):
            write_rows(output / name, [])
        write_json(output / "performance_smoke.json", {"status": "SKIPPED_NO_GPU"})
        write_json(output / "gpu_environment.json", {"status": "SKIPPED_NO_GPU"})
    else:
        gpu_validation(output, records)
    checks = defaultdict(list)
    for item in read_rows(output / "correctness.jsonl"):
        checks[item["schedule_hash"]].append(item)
    topology = {item["schedule_hash"]: item for item in read_rows(output / "launch_topology.jsonl")}
    audits = {item["schedule_hash"]: item for item in read_rows(output / "runtime_buffer_audit.jsonl")}
    previous_status = {item["schedule_hash"]: item["validation_status"] for item in
                       read_rows(PREVIOUS / "classification_after.jsonl")}
    classification = []
    for row, _, _, old, new in records:
        base = _base(new)
        if isinstance(new, HBMBackedPlan):
            tested = checks[row["schedule_hash"]]
            status = validated_new_status(tested, topology.get(row["schedule_hash"], {}),
                                          audits.get(row["schedule_hash"], {}))
        else:
            status = previous_status[row["schedule_hash"]]
        classification.append({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
            "matrix_size": row["matrix_size"], "policy": row["policy"],
            "previous_mapping_status": old.mapping_status, "previous_validation_status": previous_status[row["schedule_hash"]],
            "family_id": base.family_id, "enabling_capability": FAMILY_HBM if isinstance(new, HBMBackedPlan) else None,
            "mapping_status": base.mapping_status, "validation_status": status,
            "lowering_plan_hash": new.lowering_plan_hash, "reasons": base.unsupported_features})
    write_rows(output / "classification_after.jsonl", classification)
    known_plan_hashes = {row["schedule_hash"]: row["lowering_plan_hash"] for row in classification}
    inherited_topology = []
    for name, directory in (("HA6", HA6), ("HA7", PREVIOUS)):
        for item in read_rows(directory / "launch_topology.jsonl"):
            if known_plan_hashes[item["schedule_hash"]] != item["lowering_plan_hash"]:
                raise ValueError(f"{name} topology plan hash differs from retained schedule")
            inherited_topology.append({**item, "source_evidence": name})
    all_topology = [*inherited_topology,
                    *({**item, "source_evidence": "HA8"} for item in topology.values())]
    if len({item["schedule_hash"] for item in all_topology}) != len(all_topology):
        raise ValueError("duplicate mapped topology hash")
    write_rows(output / "all_mapped_launch_topology.jsonl", all_topology)
    coverage = {}
    for size in SIZES:
        subset = [row for row in classification if row["matrix_size"] == size]
        def count(items):
            return {"legal": len(items), "structural_exact": sum(x["mapping_status"] == "EXACT_MAPPING" for x in items),
                    "exact_passing": sum(x["mapping_status"] == "EXACT_MAPPING" and x["validation_status"] == "PASS" for x in items),
                    "approximate_passing": sum(x["mapping_status"] == "APPROXIMATE_MAPPING" and x["validation_status"] == "PASS" for x in items),
                    "unsupported": sum(x["mapping_status"] == "UNSUPPORTED" for x in items),
                    "lowering_incorrect": sum(x["validation_status"] == "LOWERING_INCORRECT" for x in items)}
        coverage[str(size)] = {**count(subset), "by_algorithm": {algorithm: count([x for x in subset if x["algorithm"] == algorithm])
                              for algorithm in ("conventional", "strassen", "project-rank8")}}
    write_json(output / "coverage_summary.json", coverage)
    reasons = Counter(reason for row in classification if row["mapping_status"] == "UNSUPPORTED"
                      for reason in row["reasons"])
    write_json(output / "remaining_unsupported_reasons.json", dict(reasons))
    selection = selected_schedule_coverage(classification)
    write_json(output / "selected_schedule_coverage.json", selection)
    architecture = backend_architecture_decision(classification)
    write_json(output / "backend_architecture_decision.json", architecture)
    new_rows = [row for row in classification if row["family_id"] == FAMILY_HBM]
    exact_passing = sum(item["exact_passing"] for item in coverage.values())
    new_passing = sum(row["validation_status"] == "PASS" for row in new_rows)
    decision = ("HBM_LIFETIME_COVERAGE_UNLOCKED" if new_passing >= 30 and exact_passing >= 78 else
                "NUMERICAL_LIMITED" if sum(row["validation_status"] == "LOWERING_INCORRECT" for row in new_rows) > len(new_rows) // 2 else
                "LOWERING_BACKEND_STILL_FRAGMENTED" if architecture["decision"] == "GENERAL_LOWERING_IR_NEEDED" else
                "INCONCLUSIVE")
    blockers = []
    if architecture["decision"] == "GENERAL_LOWERING_IR_NEEDED":
        blockers.append("remaining groups have interacting nonlocal work and require a general lowering IR")
    if selection["totals"]["eager_old_mapped_top1"]["exact_passing"] == 0:
        blockers.append("HA-5 calibrated top-1 choices were approximate legacy mappings; new paths lack calibration")
    if any(row["validation_status"] == "LOWERING_INCORRECT" for row in classification):
        blockers.append("Strassen exact mappings fail fixed tolerance at larger sizes")
    if args.skip_gpu:
        blockers.append("new HBM mappings were not GPU-validated")
    readiness = {"status": "NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE", "primary_decision": decision,
                 "backend_architecture_decision": architecture["decision"],
                 "validated_exact_schedule_records": exact_passing,
                 "criteria": {"candidate_generation": True, "legality": True, "static_metrics": True,
                              "mode_specific_calibration": True, "pre_gpu_lowerability_query": True,
                              "deterministic_plans": True, "correctness_gate": not args.skip_gpu,
                              "failure_retention": True, "benchmark_provenance": True,
                              "high_ranked_exact_coverage": False,
                              "no_manual_rewrite_for_most_selected": False},
                 "blockers": blockers}
    write_json(output / "closed_loop_readiness.json", readiness)
    write_json(output / "aggregate_summary.json", {"decision": decision, "backend_architecture": architecture["decision"],
        "coverage": coverage, "new_structural_exact": len(new_rows), "new_exact_passing": new_passing,
        "new_incorrect": sum(row["validation_status"] == "LOWERING_INCORRECT" for row in new_rows),
        "new_approximate": sum(old.mapping_status == "UNSUPPORTED" and _base(new).mapping_status == "APPROXIMATE_MAPPING"
                               for _, _, _, old, new in records),
        "correctness_checks": sum(len(items) for items in checks.values()),
        "new_topology_records": len(topology), "all_mapped_topology_records": len(all_topology),
        "dominant_remaining_reason": reasons.most_common(1),
        "selected_coverage": selection["totals"], "readiness": readiness})
    (output / "README.md").write_text("# HA-Schedule-8 retained evidence\n\nThe 156 HA-5 schedule hashes are unchanged. New exact HBM-backed mappings require CPU lifetime/transition validation, seeded full-operation correctness, runtime buffer identity, and profiler launch topology. HA-7 statuses are retained by hash. `launch_topology.jsonl` has the 44 fresh HA-8 captures; `all_mapped_launch_topology.jsonl` joins those with 72 hash-verified HA-6/7 captures and records each source. CUDA-event samples are execution smoke, not schedule ranking evidence. See docs/hardware_aware_hbm_lifetime_lowering.md.\n")
    write_json(output / "hash_manifest.json", {str(path.relative_to(output)): sha(path)
                                               for path in sorted(output.rglob("*"))
                                               if path.is_file() and path.name != "hash_manifest.json"})
    print(json.dumps({"output": str(output), "decision": decision,
                      "new_exact": len(new_rows), "new_passing": new_passing,
                      "exact_passing_total": exact_passing}, sort_keys=True))


if __name__ == "__main__":
    main()
