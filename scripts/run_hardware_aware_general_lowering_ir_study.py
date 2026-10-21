#!/usr/bin/env python3
"""Retain HA-Schedule-9 IR translation, primitive lowering, and CUDA evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game_schedule_gpu_study import verify_artifact_hashes
from matmul_inspector.tensor_game_schedule_hbm_lowering import HBMBackedPlan
from matmul_inspector.tensor_game_schedule_ir import ir_schema, translate_schedule, validate_ir
from matmul_inspector.tensor_game_schedule_ir_planner import (
    compare_legacy_plan, plan_ir, primitive_registry,
)
from scripts.run_hardware_aware_hbm_lifetime_lowering_study import (
    PREVIOUS as HA7, SOURCE, build_records, read_json, read_rows, sha,
    write_json, write_rows,
)
from scripts.run_hardware_aware_schedule_lowering_study import REPO

HA8 = REPO / "artifacts/tensor-game/hardware-aware-hbm-lifetime-lowering-v1"
SIZES = (128, 256, 512, 1024)
SEEDS = (11, 23, 37)
EXTRA_SEEDS = (41, 53)


def _base(plan):
    return plan.schedule_plan if isinstance(plan, HBMBackedPlan) else plan


def records():
    verify_artifact_hashes(HA8)
    output = []
    for row, dag, schedule, _, legacy in build_records():
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        ir = translate_schedule(dag, schedule)
        validation = validate_ir(dag, schedule, ir)
        plan = plan_ir(dag, schedule, ir, **kwargs)
        control = plan_ir(dag, schedule, ir, prefer_legacy=True, **kwargs)
        if control.legacy_plan_hash != legacy.lowering_plan_hash and _base(legacy).mapping_status != "UNSUPPORTED":
            raise ValueError("HA-8 plan identity changed")
        output.append((row, dag, schedule, legacy, ir, validation, plan, control))
    if len(output) != 156 or len({item[0]["schedule_hash"] for item in output}) != 156:
        raise ValueError("retained corpus identity changed")
    return output


def unsupported_pre_analysis(items):
    rows = []
    reasons = Counter()
    for row, dag, schedule, legacy, ir, _, plan, _ in items:
        base = _base(legacy)
        if base.mapping_status != "UNSUPPORTED":
            continue
        nodes = {node.node_id: node for node in dag.nodes}
        reasons.update(base.unsupported_features)
        rows.append({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                     "matrix_size": row["matrix_size"], "ha8_reasons": base.unsupported_features,
                     "group_signatures": [tuple((nodes[op.node_id].operation, op.location)
                                                 for op in group.operations) for group in schedule.groups],
                     "ir_hash": ir.ir_hash, "ir_primitive_ids": plan.primitive_ids})
    return {"unsupported_ha8_records": len(rows), "by_size": dict(Counter(row["matrix_size"] for row in rows)),
            "by_algorithm_per_size": {algorithm: sum(row["algorithm"] == algorithm and row["matrix_size"] == 128
                                                  for row in rows)
                                      for algorithm in ("conventional", "strassen", "project-rank8")},
            "overlapping_reasons": dict(reasons), "records": rows,
            "outside_ir_semantics": [], "solver_bound": "at most one GEMM per group",
            "max_retained_group_operations": max(len(group.operations) for _, _, schedule, *_ in items
                                                 for group in schedule.groups)}


def high_ranked_coverage(classification):
    """Join frozen HA-5 ranks; eager/graph remain old mapped-subset rankings."""
    from scripts.run_hardware_aware_hbm_lifetime_lowering_study import selected_schedule_coverage
    old_selection = selected_schedule_coverage(classification)
    by_hash = {row["schedule_hash"]: row for row in classification}
    entries = []
    for row in old_selection["entries"]:
        selected = [by_hash[value] for value in row["selected_hashes"]]
        entries.append({**row,
            "ir_valid_count": sum(item["ir_status"] == "IR_VALID" for item in selected),
            "exact_lowerable_count": sum(item["mapping_status"] == "EXACT_MAPPING" for item in selected),
            "correctness_passing_count": sum(item["validation_status"] == "PASS" and
                                             item["mapping_status"] == "EXACT_MAPPING" for item in selected)})
    totals = {}
    for name in sorted({entry["selection"] for entry in entries}):
        subset = [entry for entry in entries if entry["selection"] == name]
        totals[name] = {"selected": sum(item["selected_count"] for item in subset),
                        "ir_valid": sum(item["ir_valid_count"] for item in subset),
                        "exact_lowerable": sum(item["exact_lowerable_count"] for item in subset),
                        "correctness_passing": sum(item["correctness_passing_count"] for item in subset)}
    return {"entries": entries, "totals": totals,
            "caveat": "Eager/graph rankings are HA-5's old mapped subset. The new IR paths were not recalibrated."}


def validated_status(checks, topology, audit):
    if not checks:
        return "NOT_RUN"
    return ("PASS" if len({row["seed"] for row in checks}) >= 3 and all(row["passed"] for row in checks)
            and topology.get("topology_agrees") is True and audit.get("status") == "PASS"
            else "LOWERING_INCORRECT")


def aggregate_ir_reasons(classification):
    reasons = Counter(reason for row in classification for reason in row["unsupported_reasons"])
    return {"ir_or_primitive_reasons": dict(reasons),
        "numerical_failure_by_algorithm_size": {f"{algorithm}:{size}": sum(
            row["algorithm"] == algorithm and row["matrix_size"] == size and
            row["validation_status"] == "LOWERING_INCORRECT" for row in classification)
            for algorithm in ("conventional", "strassen", "project-rank8") for size in SIZES}}


def ir_architecture_decision(classification):
    valid = sum(row["ir_status"] == "IR_VALID" for row in classification)
    structural = sum(row["mapping_status"] == "EXACT_MAPPING" for row in classification)
    return {"decision": "IR_ARCHITECTURE_VIABLE" if valid == 156 and structural >= 140 and
            len(primitive_registry()) <= 6 else "IR_ARCHITECTURE_INSUFFICIENT",
            "ir_valid": valid, "structural_exact": structural,
            "primitive_count": len(primitive_registry()),
            "criterion": "all 156 translate; >=140 exact primitive plans; <=6 reusable primitives"}


def _representative(row, legacy):
    if row["matrix_size"] != 256:
        return False
    return row["schedule_hash"][:8] in {
        "846be065", "566c98e3", "140404eb",  # Three newly covered patterns.
        "3e5b6637", "5ebc4315", "47181a39", "1406976b", "3174902e",
        "61e6aef2",  # Legacy anchors, including a materialized identical-kernel control.
    }


def gpu_validation(output, items):
    import torch
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_ir_runtime import (
        allocate_backend, audit_ir_workspace, execute_backend,
    )
    from scripts.run_hardware_aware_schedule_gpu_study import _compiler_capture

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --skip-gpu")
    correctness, topology, audits, resources, smoke, legacy_gpu = [], [], [], [], [], []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, dag, schedule, legacy, ir, _, plan, control in items:
            if plan.mapping_status != "EXACT_MAPPING":
                continue
            workspace = allocate_backend(dag, schedule, ir, plan, lowering_name=row["lowering_name"])
            audit = audit_ir_workspace(dag, schedule, ir, plan, workspace)
            audits.append({"schedule_hash": row["schedule_hash"], "ir_hash": ir.ir_hash,
                           "lowering_plan_hash": plan.lowering_plan_hash, **audit})
            def run(left, right, capture=None):
                return execute_backend(dag, schedule, ir, plan, left, right, workspace,
                                       lowering_name=row["lowering_name"], capture=capture)
            representative = _representative(row, legacy)
            for seed in (*SEEDS, *(EXTRA_SEEDS if representative else ())):
                torch.manual_seed(seed)
                size = row["matrix_size"]
                left = torch.randn((size, size), device="cuda", dtype=torch.float16)
                right = torch.randn_like(left)
                actual = run(left, right)
                reference = left @ right
                torch.cuda.synchronize()
                error = (actual - reference).abs().float()
                correctness.append({"schedule_hash": row["schedule_hash"], "ir_hash": ir.ir_hash,
                    "lowering_plan_hash": plan.lowering_plan_hash, "algorithm": row["algorithm"],
                    "matrix_size": size, "seed": seed, "rtol": 0.005, "atol": 0.05,
                    "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
                    "passed": bool(torch.allclose(actual, reference, rtol=0.005, atol=0.05))})
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
                run(left, right)
                torch.cuda.synchronize()
            actual_names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
            expected = [launch["kernel"] for launch in plan.launch_sequence]
            agrees = len(expected) == len(actual_names) and all(
                name.startswith(kernel) for name, kernel in zip(actual_names, expected))
            captured = []
            run(left, right, captured)
            hashes = [{"group_id": item["group_id"], "primitive": item["role"],
                       "cubin_sha256": hashlib.sha256(bytes(item["compiled"].asm["cubin"])).hexdigest()}
                      for item in captured]
            topology.append({"schedule_hash": row["schedule_hash"], "ir_hash": ir.ir_hash,
                "lowering_plan_hash": plan.lowering_plan_hash, "matrix_size": row["matrix_size"],
                "expected_launch_count": len(expected), "actual_launch_count": len(actual_names),
                "expected_roles": [launch["role"] for launch in plan.launch_sequence],
                "expected_kernel_names": expected, "actual_kernel_names": actual_names,
                "kernel_hashes": hashes, "topology_agrees": agrees})
            if representative:
                first = next((item for item in captured if item["role"] == "ORDERED_ONE_GEMM"), captured[0])
                resource = _compiler_capture(first["compiled"],
                    f"ir_{row['algorithm']}_{row['schedule_hash'][:8]}_256", output / "compiler")
                resource.update({"schedule_hash": row["schedule_hash"], "ir_hash": ir.ir_hash,
                                 "algorithm": row["algorithm"], "matrix_size": 256,
                                 "kernel_count": len(captured), "primitive": first["role"]})
                resources.append(resource)
                if all(item["passed"] for item in correctness if item["schedule_hash"] == row["schedule_hash"]):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    samples = []
                    for _ in range(5):
                        start.record(); run(left, right); end.record(); end.synchronize()
                        samples.append(start.elapsed_time(end))
                    smoke.append({"candidate": f"ir_{row['algorithm']}_{row['schedule_hash'][:8]}",
                                  "schedule_hash": row["schedule_hash"], "event_ms": samples})
            if representative and _base(legacy).mapping_status != "UNSUPPORTED":
                old_workspace = allocate_backend(dag, schedule, ir, control,
                    lowering_name=row["lowering_name"], legacy_plan=legacy)
                old_capture = [] if _base(legacy).mapping_status == "EXACT_MAPPING" else None
                old_output = execute_backend(dag, schedule, ir, control, left, right, old_workspace,
                    lowering_name=row["lowering_name"], legacy_plan=legacy, capture=old_capture)
                ir_output = run(left, right)
                same_named_hashes = []
                if old_capture is not None:
                    for current_launch, old_launch in zip(captured, old_capture):
                        if current_launch["compiled"].name == old_launch["compiled"].name:
                            same_named_hashes.append({"kernel": current_launch["compiled"].name,
                                "ir_cubin_sha256": hashlib.sha256(bytes(
                                    current_launch["compiled"].asm["cubin"])).hexdigest(),
                                "legacy_cubin_sha256": hashlib.sha256(bytes(
                                    old_launch["compiled"].asm["cubin"])).hexdigest()})
                legacy_gpu.append({"schedule_hash": row["schedule_hash"], "ir_hash": ir.ir_hash,
                    "legacy_plan_hash": legacy.lowering_plan_hash,
                    "ir_plan_hash": plan.lowering_plan_hash,
                    "outputs_allclose": bool(torch.allclose(ir_output, old_output, rtol=0.005, atol=0.05)),
                    "max_ir_legacy_absolute_error": float((ir_output - old_output).abs().float().max()),
                    "same_named_kernel_hashes": same_named_hashes,
                    "same_named_kernel_hashes_agree": all(item["ir_cubin_sha256"] == item[
                        "legacy_cubin_sha256"] for item in same_named_hashes)})
        import triton
        write_json(output / "gpu_environment.json", {"device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "triton_version": triton.__version__})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    write_rows(output / "correctness.jsonl", correctness)
    write_rows(output / "launch_topology.jsonl", topology)
    write_rows(output / "buffer_audit.jsonl", audits)
    write_rows(output / "compiler_resources.jsonl", resources)
    write_rows(output / "legacy_gpu_comparison.jsonl", legacy_gpu)
    write_json(output / "execution_smoke.json", {"status": "SMOKE_ONLY_NOT_RANKING_EVIDENCE",
                                                  "candidates": smoke})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--refresh-derived", action="store_true",
                        help="Rejoin CPU evidence to intact retained GPU results")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.refresh_derived:
        parser.error("output directory already exists")
    if args.refresh_derived and (args.skip_gpu or not output.is_dir()):
        parser.error("--refresh-derived requires an existing GPU study")
    if args.refresh_derived:
        verify_artifact_hashes(output)
    items = records()
    if args.refresh_derived:
        valid = {plan.lowering_plan_hash for _, _, _, _, _, _, plan, _ in items}
        for filename in ("correctness.jsonl", "launch_topology.jsonl", "buffer_audit.jsonl"):
            if any(row["lowering_plan_hash"] not in valid for row in read_rows(output / filename)):
                raise ValueError(f"{filename} has stale IR plans")
        provenance = read_json(output / "provenance.json")
        provenance["derived_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        write_json(output / "provenance.json", provenance)
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", {"source_corpus": str(SOURCE.relative_to(REPO)),
            "previous_evidence": str(HA8.relative_to(REPO)), "matrix_sizes": SIZES,
            "seeds": SEEDS, "representative_extra_seeds": EXTRA_SEEDS,
            "rtol": 0.005, "atol": 0.05, "gpu_enabled": not args.skip_gpu,
            "default_backend": "IR_PRIMITIVES", "legacy_control": "prefer_legacy=True"})
        write_json(output / "provenance.json", {"source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "ha8_hash_manifest_sha256": sha(HA8 / "hash_manifest.json"),
            "ha5_hash_manifest_sha256": sha(SOURCE / "hash_manifest.json")})
    write_json(output / "unsupported_pre_analysis.json", unsupported_pre_analysis(items))
    write_json(output / "ir_schema.json", ir_schema())
    write_json(output / "primitive_registry.json", {"primitives": primitive_registry()})
    write_rows(output / "schedule_ir.jsonl", [{"matrix_size": row["matrix_size"],
        "schedule_hash": row["schedule_hash"], **asdict(ir)} for row, _, _, _, ir, _, _, _ in items])
    write_rows(output / "ir_validation.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], "ir_hash": ir.ir_hash, **validation}
        for row, _, _, _, ir, validation, _, _ in items])
    write_rows(output / "ir_lowering_plans.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], **asdict(plan)} for row, _, _, _, _, _, plan, _ in items])
    write_rows(output / "legacy_equivalence.jsonl", [{"schedule_hash": row["schedule_hash"],
        "matrix_size": row["matrix_size"], "algorithm": row["algorithm"],
        **compare_legacy_plan(legacy, plan)} for row, _, _, legacy, _, _, plan, _ in items])
    if args.refresh_derived:
        pass
    elif args.skip_gpu:
        for name in ("correctness.jsonl", "launch_topology.jsonl", "buffer_audit.jsonl",
                     "compiler_resources.jsonl", "legacy_gpu_comparison.jsonl"):
            write_rows(output / name, [])
        write_json(output / "execution_smoke.json", {"status": "SKIPPED_NO_GPU"})
        write_json(output / "gpu_environment.json", {"status": "SKIPPED_NO_GPU"})
    else:
        gpu_validation(output, items)
    checks = defaultdict(list)
    for row in read_rows(output / "correctness.jsonl"):
        checks[row["schedule_hash"]].append(row)
    topology = {row["schedule_hash"]: row for row in read_rows(output / "launch_topology.jsonl")}
    audits = {row["schedule_hash"]: row for row in read_rows(output / "buffer_audit.jsonl")}
    previous_status = {row["schedule_hash"]: row for row in read_rows(HA8 / "classification_after.jsonl")}
    classification = []
    for row, _, _, legacy, ir, validation, plan, _ in items:
        status = validated_status(checks[row["schedule_hash"]], topology.get(row["schedule_hash"], {}),
                                  audits.get(row["schedule_hash"], {})) if plan.mapping_status == "EXACT_MAPPING" else "NOT_RUN"
        classification.append({"schedule_hash": row["schedule_hash"], "matrix_size": row["matrix_size"],
            "algorithm": row["algorithm"], "policy": row["policy"], "ir_hash": ir.ir_hash,
            "ir_status": validation["status"], "lowering_plan_hash": plan.lowering_plan_hash,
            "backend": plan.backend, "mapping_status": plan.mapping_status,
            "validation_status": status, "previous_mapping_status": _base(legacy).mapping_status,
            "previous_validation_status": previous_status[row["schedule_hash"]]["validation_status"],
            "unsupported_reasons": plan.unsupported_reasons})
    write_rows(output / "classification_after.jsonl", classification)
    coverage = {}
    for size in SIZES:
        subset = [row for row in classification if row["matrix_size"] == size]
        def count(rows):
            return {"legal": len(rows), "ir_valid": sum(row["ir_status"] == "IR_VALID" for row in rows),
                "ir_unsupported": sum(row["ir_status"] == "IR_UNSUPPORTED" for row in rows),
                "ir_invalid": sum(row["ir_status"] == "IR_INVALID" for row in rows),
                "structural_exact": sum(row["mapping_status"] == "EXACT_MAPPING" for row in rows),
                "exact_passing": sum(row["mapping_status"] == "EXACT_MAPPING" and
                                     row["validation_status"] == "PASS" for row in rows),
                "approximate": sum(row["mapping_status"] == "APPROXIMATE_MAPPING" for row in rows),
                "lowering_unsupported": sum(row["mapping_status"] == "LOWERING_UNSUPPORTED" for row in rows),
                "lowering_incorrect": sum(row["validation_status"] == "LOWERING_INCORRECT" for row in rows)}
        coverage[str(size)] = {**count(subset), "by_algorithm": {algorithm: count([
            row for row in subset if row["algorithm"] == algorithm])
            for algorithm in ("conventional", "strassen", "project-rank8")}}
    write_json(output / "coverage_summary.json", coverage)
    selection = high_ranked_coverage(classification)
    write_json(output / "high_ranked_coverage.json", selection)
    write_json(output / "remaining_unsupported_reasons.json", aggregate_ir_reasons(classification))
    passing = sum(item["exact_passing"] for item in coverage.values())
    valid = sum(item["ir_valid"] for item in coverage.values())
    structural = sum(item["structural_exact"] for item in coverage.values())
    architecture = ir_architecture_decision(classification)
    write_json(output / "ir_architecture_decision.json", architecture)
    decision = ("GENERAL_IR_COVERAGE_UNLOCKED" if passing >= 117 and architecture["decision"] ==
                "IR_ARCHITECTURE_VIABLE" else "NUMERICAL_LIMITED" if structural >= 140 and passing < 117 and
                not args.skip_gpu else "PRIMITIVE_BACKEND_LIMITED" if structural < 117 else "INCONCLUSIVE")
    blockers = []
    if args.skip_gpu:
        blockers.append("GPU correctness and topology not run")
    if any(row["validation_status"] == "LOWERING_INCORRECT" for row in classification):
        blockers.append("large-size Strassen exact plans fail the fixed numerical tolerance")
    blockers.append("newly exact candidates lack fresh eager/graph calibration and general result-driven reselection")
    readiness = {"status": "NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE", "primary_decision": decision,
                 "ir_architecture_decision": architecture["decision"], "validated_exact_schedule_records": passing,
                 "threshold_75_percent_met": passing >= 117,
                 "blockers": blockers,
                 "criteria": {"candidate_generation": True, "legality": True, "static_metrics": True,
                    "existing_mode_calibration": True, "cpu_lowerability_query": True,
                    "automatic_ir_translation": True, "automatic_ir_validation": True,
                    "automatic_primitive_planning": True, "correctness_gate": not args.skip_gpu,
                    "failure_retention": True, "benchmark_provenance": True,
                    "new_paths_mode_calibrated": False, "general_result_driven_reselection": False}}
    write_json(output / "closed_loop_readiness.json", readiness)
    write_json(output / "aggregate_summary.json", {"decision": decision,
        "ir_architecture_decision": architecture["decision"], "closed_loop_readiness": readiness["status"],
        "ir_valid": valid, "ir_unsupported": 156 - valid, "structural_exact": structural,
        "exact_passing": passing, "coverage": coverage,
        "seed_checks": sum(len(rows) for rows in checks.values()),
        "topology_agreements": sum(row["topology_agrees"] for row in topology.values()),
        "buffer_audits_passing": sum(row["status"] == "PASS" for row in audits.values()),
        "legacy_equivalence": dict(Counter(row["status"] for row in
                                        read_rows(output / "legacy_equivalence.jsonl"))),
        "high_ranked": selection["totals"]})
    (output / "README.md").write_text(
        "# HA-Schedule-9 retained evidence\n\nThe 156 HA-5 schedule hashes and HA-8 evidence are unchanged. "
        "The IR backend is used for every retained group. Legacy plans are retained as compatibility controls. "
        "`IR_VALID` and structural exactness do not imply numerical correctness. Fixed-tolerance seed checks, "
        "logical HBM buffer audits, and profiler launch order gate passing coverage. CUDA-event samples are "
        "execution smoke, not ranking evidence. See docs/hardware_aware_general_lowering_ir.md.\n")
    write_json(output / "hash_manifest.json", {str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "hash_manifest.json"})
    print(json.dumps({"output": str(output), "decision": decision, "ir_valid": valid,
                      "structural_exact": structural, "exact_passing": passing}, sort_keys=True))


if __name__ == "__main__":
    main()
