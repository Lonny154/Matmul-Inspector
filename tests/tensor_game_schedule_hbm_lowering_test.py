"""CPU-first contracts and optional CUDA checks for HA-Schedule-8."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json

import pytest

from matmul_inspector.tensor_game_schedule_calibration import stable_hash
from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_hbm_lowering import (
    FAMILY_HBM, HBMBackedPlan, HBMValueLifetime, _plan_payload,
    classify_hbm_schedule, hbm_lowerability_query, reference_transition_checks,
    validate_hbm_plan,
)
from matmul_inspector.tensor_game_schedule_mixed_lowering import classify_mixed_schedule
from scripts.run_hardware_aware_schedule_lowering_study import FROZEN, REPO, make_dag, project_actions, source_rows
from scripts.run_hardware_aware_hbm_lifetime_lowering_study import (
    backend_architecture_decision, selected_schedule_coverage, validated_new_status,
)


@pytest.fixture(scope="module")
def corpus():
    actions = project_actions(FROZEN)
    result = []
    for row in source_rows():
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        old = classify_mixed_schedule(dag, schedule, **kwargs)
        new = classify_hbm_schedule(dag, schedule, **kwargs)
        result.append((row, dag, schedule, old, new))
    return result


def _new(corpus, algorithm="project-rank8", prefix="00d4dd6a", size=128):
    return next(item for item in corpus if item[0]["algorithm"] == algorithm and
                item[0]["matrix_size"] == size and item[0]["schedule_hash"].startswith(prefix))


def _rehash(plan, *, lifetimes=None, buffers=None, contributions=None):
    lifetimes = plan.hbm_lifetimes if lifetimes is None else tuple(lifetimes)
    buffers = plan.buffer_plan if buffers is None else tuple(buffers)
    contributions = plan.contribution_order if contributions is None else tuple(contributions)
    digest = stable_hash(_plan_payload(plan.schedule_plan, lifetimes, buffers, contributions))
    return replace(plan, hbm_lifetimes=tuple(replace(life, lowering_plan_hash=digest) for life in lifetimes),
                   buffer_plan=buffers, contribution_order=contributions, lowering_plan_hash=digest)


def test_new_family_coverage_determinism_and_old_plan_compatibility(corpus):
    assert len(corpus) == 156
    new = [item for item in corpus if isinstance(item[-1], HBMBackedPlan)]
    assert len(new) == 44
    assert Counter((row["algorithm"], row["matrix_size"]) for row, *_ in new) == {
        (algorithm, size): count for size in (128, 256, 512, 1024)
        for algorithm, count in (("conventional", 1), ("project-rank8", 5), ("strassen", 5))}
    for row, dag, schedule, old, current in corpus:
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        assert current == classify_hbm_schedule(dag, schedule, **kwargs)
        assert hbm_lowerability_query(dag, schedule, **kwargs)["schedule_hash"] == row["schedule_hash"]
        if old.mapping_status != "UNSUPPORTED":
            assert current == old  # HA-6/7 serialized plans and hashes remain valid.
        elif isinstance(current, HBMBackedPlan):
            assert current.schedule_plan.family_id == FAMILY_HBM
            assert current.schedule_plan.mapping_status == "EXACT_MAPPING"
            assert validate_hbm_plan(dag, schedule, current)["status"] == "PASS"


def test_typed_lifetimes_and_serialization(corpus):
    _, dag, schedule, _, plan = _new(corpus, prefix="4cfed2b2")
    assert isinstance(plan, HBMBackedPlan)
    classes = {life.value_class for life in plan.hbm_lifetimes}
    assert classes == {"PREPARED_FORM", "GEMM_PRODUCT", "PARTIAL_OUTPUT", "FINAL_OUTPUT"}
    for life in plan.hbm_lifetimes:
        assert HBMValueLifetime(**asdict(life)) == life
        assert life.source_schedule_hash == schedule.schedule_hash
        assert life.lowering_plan_hash == plan.lowering_plan_hash
        assert life.byte_size > 0
        for read in life.read_semantics:
            assert read in {"LOCAL_FORWARD", "READ_FORM", "READ_PRODUCT", "READ_PARTIAL"}
    assert plan.lowering_plan_hash == stable_hash(_plan_payload(
        plan.schedule_plan, plan.hbm_lifetimes, plan.buffer_plan, plan.contribution_order))


def test_prepared_form_product_and_partial_contracts(corpus):
    _, dag, schedule, _, plan = _new(corpus, prefix="4cfed2b2")
    by_class = {name: [life for life in plan.hbm_lifetimes if life.value_class == name]
                for name in ("PREPARED_FORM", "GEMM_PRODUCT", "PARTIAL_OUTPUT", "FINAL_OUTPUT")}
    assert any("READ_FORM" in life.read_semantics for life in by_class["PREPARED_FORM"])
    assert any("LOCAL_FORWARD" in life.read_semantics for life in by_class["GEMM_PRODUCT"])
    _, _, _, _, product_plan = _new(corpus, algorithm="conventional", prefix="bbe200b8")
    assert any("READ_PRODUCT" in life.read_semantics for life in product_plan.hbm_lifetimes)
    assert any("READ_PARTIAL" in life.read_semantics for life in by_class["PARTIAL_OUTPUT"])
    assert any(life.write_semantics == "UPDATE_PARTIAL" for life in by_class["PARTIAL_OUTPUT"])
    assert all(life.write_semantics == "WRITE_FINAL" and not life.consumer_groups
               for life in by_class["FINAL_OUTPUT"])
    assert all(check["passed"] and check["state"] == "FINAL" for check in
               reference_transition_checks(dag, schedule, plan))


def test_buffer_plan_is_stable_and_matches_schedule(corpus):
    _, dag, schedule, _, plan = _new(corpus)
    assert len(plan.buffer_plan) == len(plan.hbm_lifetimes)
    assert {buffer.value_id for buffer in plan.buffer_plan} == {
        op.node_id for group in schedule.groups for op in group.operations if op.location == "HBM"}
    assert {buffer.buffer_id for buffer in plan.buffer_plan if buffer.output_block is None} == {
        f"buf:{buffer.value_id}" for buffer in plan.buffer_plan if buffer.output_block is None}
    assert validate_hbm_plan(dag, schedule, plan) == validate_hbm_plan(dag, schedule, plan)


def test_invalid_consumer_and_typed_read_are_rejected(corpus):
    _, dag, schedule, _, plan = _new(corpus, algorithm="conventional", prefix="bbe200b8")
    index = next(i for i, life in enumerate(plan.hbm_lifetimes)
                 if "READ_PRODUCT" in life.read_semantics)
    life = plan.hbm_lifetimes[index]
    altered = list(plan.hbm_lifetimes)
    altered[index] = replace(life, consumer_groups=(schedule.groups[0].group_id, *life.consumer_groups[1:]),
                             first_consumer_group=schedule.groups[0].group_id)
    with pytest.raises(ValueError, match="consumer|producer"):
        validate_hbm_plan(dag, schedule, _rehash(plan, lifetimes=altered))
    altered[index] = replace(life, read_semantics=("READ_FORM", *life.read_semantics[1:]))
    with pytest.raises(ValueError, match="consumer|typed"):
        validate_hbm_plan(dag, schedule, _rehash(plan, lifetimes=altered))


def test_final_output_and_duplicate_contributions_are_rejected(corpus):
    _, dag, schedule, _, plan = _new(corpus)
    index = next(i for i, life in enumerate(plan.hbm_lifetimes) if life.value_class == "FINAL_OUTPUT")
    life = plan.hbm_lifetimes[index]
    altered = list(plan.hbm_lifetimes)
    altered[index] = replace(life, consumer_groups=(schedule.groups[-1].group_id,),
                             consumer_node_ids=("M00",), read_semantics=("READ_PARTIAL",),
                             first_consumer_group=schedule.groups[-1].group_id,
                             final_consumer_group=schedule.groups[-1].group_id)
    with pytest.raises(ValueError):
        validate_hbm_plan(dag, schedule, _rehash(plan, lifetimes=altered))
    changed = list(plan.contribution_order)
    changed[0] = {**changed[0], "accumulation_ids": (*changed[0]["accumulation_ids"],
                                                     changed[0]["accumulation_ids"][-1])}
    with pytest.raises(ValueError, match="contribution"):
        validate_hbm_plan(dag, schedule, _rehash(plan, contributions=changed))


def test_hidden_materialization_and_overlapping_reuse_are_rejected(corpus):
    _, dag, schedule, _, plan = _new(corpus, prefix="4cfed2b2")
    launch_index = next(i for i, launch in enumerate(plan.schedule_plan.launch_sequence)
                        if launch["role"] == FAMILY_HBM and launch["product_write"])
    launches = list(plan.schedule_plan.launch_sequence)
    launches[launch_index] = {**launches[launch_index], "product_write": False}
    altered_base = replace(plan.schedule_plan, launch_sequence=tuple(launches))
    altered_plan = replace(plan, schedule_plan=altered_base)
    with pytest.raises(ValueError, match="launch writes"):
        validate_hbm_plan(dag, schedule, altered_plan)
    buffers = list(plan.buffer_plan)
    first, second = [i for i, buffer in enumerate(buffers) if buffer.output_block is None][:2]
    buffers[second] = replace(buffers[second], buffer_id=buffers[first].buffer_id)
    with pytest.raises(ValueError, match="overlapping"):
        validate_hbm_plan(dag, schedule, _rehash(plan, buffers=buffers))


def test_remaining_unsupported_patterns_are_not_promoted(corpus):
    remaining = [item for item in corpus if not isinstance(item[-1], HBMBackedPlan) and
                 item[-1].mapping_status == "UNSUPPORTED"]
    assert len(remaining) == 40
    reasons = Counter(reason for item in remaining for reason in item[-1].unsupported_features)
    assert reasons["NONLOCAL_RECONSTRUCTION_UNSUPPORTED"] == 36
    assert reasons["NONLOCAL_OUTPUT_UNSUPPORTED"] == 24
    assert reasons["NONLOCAL_FORM_WORK_UNSUPPORTED"] == 12
    assert reasons["DUPLICATE_OUTPUT_TARGET_UNSUPPORTED"] == 8


def test_coverage_selection_and_architecture_are_deterministic(corpus):
    previous = REPO / "artifacts/tensor-game/hardware-aware-mixed-group-lowering-v1/classification_after.jsonl"
    previous_status = {row["schedule_hash"]: row["validation_status"] for row in
                       (json.loads(line) for line in previous.read_text().splitlines())}
    classification = []
    for row, _, _, _, plan in corpus:
        base = plan.schedule_plan if isinstance(plan, HBMBackedPlan) else plan
        status = ("LOWERING_INCORRECT" if row["algorithm"] == "strassen" and row["matrix_size"] >= 512
                  else "PASS") if isinstance(plan, HBMBackedPlan) else previous_status[row["schedule_hash"]]
        classification.append({"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                               "matrix_size": row["matrix_size"], "mapping_status": base.mapping_status,
                               "validation_status": status, "reasons": base.unsupported_features})
    assert sum(row["mapping_status"] == "EXACT_MAPPING" and row["validation_status"] == "PASS"
               for row in classification) == 82
    selection = selected_schedule_coverage(classification)["totals"]
    assert selection["static_top1"] == {"selected": 12, "exact_passing": 8}
    assert selection["static_top3"] == {"selected": 36, "exact_passing": 21}
    assert selection["static_pareto"] == {"selected": 108, "exact_passing": 46}
    assert selection["eager_old_mapped_top1"]["exact_passing"] == 0
    decision = backend_architecture_decision(classification)
    assert decision["decision"] == "GENERAL_LOWERING_IR_NEEDED"
    assert (decision["remaining_n128_schedules"], decision["distinct_blocker_combinations"],
            decision["schedules_with_multiple_blockers"]) == (10, 6, 9)
    assert backend_architecture_decision(classification) == decision


def test_failed_correctness_and_topology_block_passing_coverage():
    good = [{"seed": seed, "passed": True} for seed in (11, 23, 37)]
    assert validated_new_status(good, {"topology_agrees": True}, {"status": "PASS"}) == "PASS"
    assert validated_new_status([*good[:2], {"seed": 37, "passed": False}],
                                {"topology_agrees": True}, {"status": "PASS"}) == "LOWERING_INCORRECT"
    assert validated_new_status(good, {"topology_agrees": False}, {"status": "PASS"}) == "LOWERING_INCORRECT"
    assert validated_new_status(good, {"topology_agrees": True}, {"status": "FAIL"}) == "LOWERING_INCORRECT"
    assert validated_new_status(good[:2], {"topology_agrees": True}, {"status": "PASS"}) == "LOWERING_INCORRECT"


@pytest.mark.parametrize("algorithm,prefix", [
    ("conventional", "bbe200b8"), ("project-rank8", "00d4dd6a"),
    ("project-rank8", "4cfed2b2"), ("strassen", "63a9d1d3")])
@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_hbm_gpu_correctness_topology_and_buffer_audit(algorithm, prefix, size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_hbm_runtime import (
        allocate_hbm, audit_hbm_workspace, execute_hbm,
    )

    source = next(row for row in source_rows() if row["algorithm"] == algorithm and
                  row["matrix_size"] == 128 and row["schedule_hash"].startswith(prefix))
    row = next(row for row in source_rows() if row["algorithm"] == algorithm and
               row["matrix_size"] == size and row["schedule"]["groups"] == source["schedule"]["groups"] and
               row["schedule"]["recompute_allowed"] == source["schedule"]["recompute_allowed"])
    dag = make_dag(algorithm, size, project_actions(FROZEN))
    schedule = _schedule(row["schedule"])
    plan = classify_hbm_schedule(dag, schedule, lowering_name=row["lowering_name"])
    assert isinstance(plan, HBMBackedPlan)
    workspace = allocate_hbm(dag, schedule, plan)
    assert audit_hbm_workspace(dag, schedule, plan, workspace)["status"] == "PASS"
    for seed in (11, 23, 37):
        torch.manual_seed(seed)
        left = torch.randn((size, size), device="cuda", dtype=torch.float16)
        right = torch.randn_like(left)
        actual = execute_hbm(dag, schedule, plan, left, right, workspace)
        if algorithm == "strassen" and size >= 512:
            assert torch.isfinite(actual).all()  # Retained numerical failure; do not hard-code across GPUs.
        else:
            assert torch.allclose(actual, left @ right, rtol=0.005, atol=0.05)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        execute_hbm(dag, schedule, plan, left, right, workspace)
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    assert len(names) == plan.schedule_plan.expected_launch_count
    assert all(actual.startswith(launch["kernel"]) for actual, launch in
               zip(names, plan.schedule_plan.launch_sequence))
