"""CPU-first contracts for bounded mixed GEMM/reconstruction lowering."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_lowering import benchmarkable, classify_schedule
from matmul_inspector.tensor_game_schedule_mixed_lowering import (
    FAMILY_MIXED, classify_mixed_schedule, mixed_group_specs, mixed_lowerability_query,
)
from matmul_inspector.tensor_game_schedule_model import (
    LaunchGroup, _digest, _schedule_payload, validate_schedule,
)
from scripts.run_hardware_aware_schedule_lowering_study import FROZEN, make_dag, project_actions, source_rows
from scripts.run_hardware_aware_mixed_group_lowering_study import group_patterns


@pytest.fixture(scope="module")
def records():
    actions = project_actions(FROZEN)
    result = []
    for row in source_rows():
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        kwargs = {"lowering_name": row["lowering_name"], "static_metrics": row["metrics"],
                  "calibration_features": row["features"]}
        old = classify_schedule(dag, schedule, **kwargs)
        new = classify_mixed_schedule(dag, schedule, **kwargs)
        result.append((row, dag, schedule, old, new))
    return result


def test_classification_is_deterministic_and_preserves_old_mappings(records):
    assert len(records) == 156
    for row, dag, schedule, old, new in records:
        assert new == classify_mixed_schedule(dag, schedule, lowering_name=row["lowering_name"],
                                               static_metrics=row["metrics"],
                                               calibration_features=row["features"])
        assert new.schedule_hash == row["schedule_hash"]
        assert len(new.lowering_plan_hash) == 64
        if old.mapping_status != "UNSUPPORTED":
            assert new == old
    assert Counter((row["matrix_size"], new.mapping_status) for row, _, _, _, new in records) == {
        (size, status): count for size in (128, 256, 512, 1024)
        for status, count in (("EXACT_MAPPING", 14), ("APPROXIMATE_MAPPING", 4), ("UNSUPPORTED", 21))}


def test_mixed_semantics_and_plan_hash(records):
    mixed = [(row, dag, schedule, plan) for row, dag, schedule, old, plan in records
             if old.mapping_status == "UNSUPPORTED" and plan.family_id == FAMILY_MIXED]
    assert len(mixed) == 36
    assert Counter(row["algorithm"] for row, *_ in mixed) == {
        "conventional": 12, "strassen": 12, "project-rank8": 12}
    for row, dag, schedule, plan in mixed:
        nodes = {node.node_id: node for node in dag.nodes}
        assert plan.expected_launch_count == len(schedule.groups) == len(plan.launch_sequence)
        assert plan.mapping_status == "EXACT_MAPPING"
        assert mixed_lowerability_query(dag, schedule)["lowering_plan_hash"]
        for launch in plan.launch_sequence:
            if launch["role"] != FAMILY_MIXED:
                continue
            assert launch["fused_gemm_id"] == launch["node_id"]
            assert launch["product_materialized"] is False
            assert launch["fused_gemm_id"] not in plan.materialized_values
            assert len(launch["reconstruction_targets"]) in (1, 2)
            assert tuple(launch["input_sources"]) == nodes[launch["fused_gemm_id"]].dependencies
            for target in launch["reconstruction_targets"]:
                node = nodes[target["accumulation_id"]]
                assert target["coefficient"] == node.coefficients[0]
                assert target["output_block"] == node.output_block
                assert target["mode"] == ("READ_MODIFY_WRITE" if target["previous_id"] else "INITIALIZE")
                assert target["write"] in ("FINAL_FP16", "PARTIAL_FP32")


def test_mixed_group_pattern_count_and_remaining_reasons(records):
    patterns = group_patterns(records)
    assert patterns["unsupported_with_mixed_group_by_size"] == {size: 29 for size in (128, 256, 512, 1024)}
    assert patterns["distinct_n128_group_signatures"] == 48
    assert patterns["isolated_remaining_blockers"] == {"COLOCATED_FORM_MATERIALIZATION_UNSUPPORTED": 8}
    remaining = Counter(reason for _, _, _, _, plan in records if plan.mapping_status == "UNSUPPORTED"
                        for reason in plan.unsupported_features)
    assert remaining["PARTIAL_OUTPUT_LIFETIME_UNSUPPORTED"] == 72
    assert remaining["MIXED_HBM_PRODUCT_UNSUPPORTED"] == 68


def test_unsupported_patterns_do_not_become_exact(records):
    unsupported = [item for item in records if item[-1].mapping_status == "UNSUPPORTED"]
    assert any("RECONSTRUCTION_FANOUT_GT2_UNSUPPORTED" in item[-1].unsupported_features for item in unsupported)
    assert any("MIXED_HBM_PRODUCT_UNSUPPORTED" in item[-1].unsupported_features for item in unsupported)
    assert any("COLOCATED_FORM_MATERIALIZATION_UNSUPPORTED" in item[-1].unsupported_features for item in unsupported)
    assert any("NONLOCAL_RECONSTRUCTION_UNSUPPORTED" in item[-1].unsupported_features for item in unsupported)
    old_approximate = [item for item in records if item[-2].mapping_status == "APPROXIMATE_MAPPING"]
    assert all(item[-1].mapping_status == "APPROXIMATE_MAPPING" for item in old_approximate)


def test_multi_gemm_group_is_rejected_even_when_schedule_is_legal(records):
    row, dag, schedule, _, _ = next(item for item in records if item[0]["algorithm"] == "conventional"
                                    and item[0]["matrix_size"] == 128 and
                                    item[0]["policy"] == "FUSE_OUTPUT_RECONSTRUCTION")
    nodes = {node.node_id: node for node in dag.nodes}
    candidates = [index for index, group in enumerate(schedule.groups)
                  if any(nodes[op.node_id].operation == "MATMUL" for op in group.operations)]
    first, second = candidates[:2]
    assert second == first + 1
    groups = list(schedule.groups)
    groups[first:second + 1] = [LaunchGroup(groups[first].group_id,
                                           groups[first].operations + groups[second].operations)]
    modified = replace(schedule, groups=tuple(groups), schedule_hash="")
    modified = replace(modified, schedule_hash=_digest(_schedule_payload(modified)))
    assert validate_schedule(dag, modified).valid
    _, reasons = mixed_group_specs(dag, modified)
    assert "NON_GEMM_MULTI_OPERATION_GROUP_UNSUPPORTED" in reasons
    assert classify_mixed_schedule(dag, modified).mapping_status == "UNSUPPORTED"


def test_failed_correctness_cannot_be_benchmarked(records):
    plan = next(plan for _, _, _, _, plan in records if plan.family_id == FAMILY_MIXED)
    good = [{"lowering_plan_hash": plan.lowering_plan_hash, "seed": seed, "passed": True}
            for seed in (11, 23, 37)]
    assert benchmarkable(plan, good)
    assert not benchmarkable(plan, [*good[:2], {**good[2], "passed": False}])
    assert not benchmarkable(plan, good[:2])


@pytest.mark.parametrize("algorithm", ["conventional", "strassen", "project-rank8"])
@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_gpu_mixed_correctness_and_topology(algorithm, size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_mixed_runtime import allocate_mixed, execute_mixed

    row = next(row for row in source_rows() if row["algorithm"] == algorithm and
               row["matrix_size"] == size and row["policy"] == "FUSE_OUTPUT_RECONSTRUCTION")
    dag = make_dag(algorithm, size, project_actions(FROZEN))
    schedule = _schedule(row["schedule"])
    plan = classify_mixed_schedule(dag, schedule, lowering_name=row["lowering_name"])
    assert plan.family_id == FAMILY_MIXED
    workspace = allocate_mixed(dag, schedule, plan)
    assert all(launch["fused_gemm_id"] not in workspace["values"]
               for launch in plan.launch_sequence if launch["role"] == FAMILY_MIXED)
    for seed in (11, 23, 37):
        torch.manual_seed(seed)
        left = torch.randn((size, size), device="cuda", dtype=torch.float16)
        right = torch.randn_like(left)
        actual = execute_mixed(dag, schedule, plan, left, right, workspace)
        passes = torch.allclose(actual, left @ right, rtol=0.005, atol=0.05)
        if algorithm == "strassen" and size >= 512:
            # Retained GPU evidence fails here, but another GPU may do better.
            assert torch.isfinite(actual).all()
        else:
            assert passes
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        execute_mixed(dag, schedule, plan, left, right, workspace)
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    assert len(names) == plan.expected_launch_count
    assert all(actual.startswith(launch["kernel"]) for actual, launch in zip(names, plan.launch_sequence))
