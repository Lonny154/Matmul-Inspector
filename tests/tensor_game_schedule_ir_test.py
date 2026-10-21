"""CPU-first schedule IR contracts and optional CUDA primitive checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json

import pytest

from matmul_inspector.tensor_game_schedule_calibration import stable_hash
from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_hbm_lowering import HBMBackedPlan, classify_hbm_schedule
from matmul_inspector.tensor_game_schedule_ir import _payload, translate_schedule, validate_ir
from matmul_inspector.tensor_game_schedule_ir_planner import (
    compare_legacy_plan, ir_lowerability_query, match_group, plan_ir, primitive_registry,
)
from scripts.run_hardware_aware_schedule_lowering_study import FROZEN, make_dag, project_actions, source_rows
from scripts.run_hardware_aware_general_lowering_ir_study import (
    aggregate_ir_reasons, high_ranked_coverage, ir_architecture_decision, validated_status,
)


@pytest.fixture(scope="module")
def corpus():
    actions = project_actions(FROZEN)
    result = []
    for row in source_rows():
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        ir = translate_schedule(dag, schedule)
        plan = plan_ir(dag, schedule, ir, lowering_name=row["lowering_name"])
        legacy = classify_hbm_schedule(dag, schedule, lowering_name=row["lowering_name"])
        result.append((row, dag, schedule, ir, plan, legacy))
    return result


def _record(corpus, algorithm="conventional", prefix="8467ca32", size=128):
    return next(item for item in corpus if item[0]["algorithm"] == algorithm and
                item[0]["matrix_size"] == size and item[0]["schedule_hash"].startswith(prefix))


def _rehash(ir, **changes):
    modified = replace(ir, **changes)
    return replace(modified, ir_hash=stable_hash(_payload(modified)))


def test_all_retained_schedules_translate_validate_and_hash(corpus):
    assert len(corpus) == 156
    assert Counter(item[3].version for item in corpus) == {"ha-schedule-9-ir-v1": 156}
    assert Counter(item[4].mapping_status for item in corpus) == {"EXACT_MAPPING": 156}
    for row, dag, schedule, ir, plan, _ in corpus:
        assert ir == translate_schedule(dag, schedule)
        serialized = json.dumps(asdict(ir), sort_keys=True, separators=(",", ":"))
        assert serialized == json.dumps(json.loads(serialized), sort_keys=True, separators=(",", ":"))
        assert ir.ir_hash == stable_hash(_payload(ir))
        assert validate_ir(dag, schedule, ir)["status"] == "IR_VALID"
        assert plan.source_schedule_hash == row["schedule_hash"]
        assert plan.ir_hash == ir.ir_hash
        assert plan.expected_launch_count == len(schedule.groups)
        assert ir_lowerability_query(dag, schedule, lowering_name=row["lowering_name"])["ir_status"] == "IR_VALID"


def test_algorithm_families_and_recomputation(corpus):
    assert {item[0]["algorithm"] for item in corpus} == {"conventional", "strassen", "project-rank8"}
    stored = _record(corpus, algorithm="project-rank8", prefix="fb511c78")
    recomputed = _record(corpus, algorithm="project-rank8", prefix="ed149344")
    assert not stored[3].recomputation_records
    assert recomputed[3].recomputation_records
    assert all(op.kind == "RECOMPUTE_FORM" for op in recomputed[3].operations
               if op.op_id in {item["op_id"] for item in recomputed[3].recomputation_records})
    legacy_families = Counter((item[5].schedule_plan if isinstance(item[5], HBMBackedPlan)
                               else item[5]).family_id for item in corpus)
    assert legacy_families["MATERIALIZED_DAG"] and legacy_families["FUSED_GEMM_RECONSTRUCTION"]
    assert legacy_families["HBM_BACKED_MIXED_DAG"] and legacy_families["LEGACY_PRODUCT_RECONSTRUCTION"]


def test_value_and_buffer_lifetimes(corpus):
    _, dag, schedule, ir, _, _ = _record(corpus)
    assert {value.value_class for value in ir.values} == {
        "INPUT", "GEMM_PRODUCT", "PARTIAL_OUTPUT", "FINAL_OUTPUT"}
    assert {buffer.value_id for buffer in ir.buffer_plan} == {
        value.value_id for value in ir.values if value.location == "HBM"}
    assert {buffer.buffer_id for buffer in ir.buffer_plan if buffer.output_block is not None} == {"output"}
    assert validate_ir(dag, schedule, ir)["hbm_buffer_count"] == len(ir.buffer_plan)


def test_invalid_dependency_missing_producer_hbm_and_cross_group_rejected(corpus):
    _, dag, schedule, ir, _, _ = _record(corpus)
    index = next(i for i, op in enumerate(ir.operations) if op.kind == "RECONSTRUCT")
    ops = list(ir.operations)
    ops[index] = replace(ops[index], inputs=("missing-value",))
    with pytest.raises(ValueError, match="dependency|operation"):
        validate_ir(dag, schedule, _rehash(ir, operations=tuple(ops)))
    value_index = next(i for i, value in enumerate(ir.values) if value.value_class == "GEMM_PRODUCT"
                       and value.location == "HBM")
    values = list(ir.values)
    values[value_index] = replace(values[value_index], producer_op="op:999")
    with pytest.raises(ValueError):
        validate_ir(dag, schedule, _rehash(ir, values=tuple(values)))
    values[value_index] = replace(ir.values[value_index], buffer_id=None)
    with pytest.raises(ValueError, match="HBM|buffer"):
        validate_ir(dag, schedule, _rehash(ir, values=tuple(values)))
    values[value_index] = replace(ir.values[value_index], location="ON_CHIP", buffer_id=None)
    with pytest.raises(ValueError):
        validate_ir(dag, schedule, _rehash(ir, values=tuple(values)))


def test_output_state_duplicates_finalization_and_recompute_inputs_rejected(corpus):
    _, dag, schedule, ir, _, _ = _record(corpus)
    transitions = list(ir.output_transitions)
    transitions[0] = {**transitions[0], "contribution_nodes": (
        *transitions[0]["contribution_nodes"], transitions[0]["contribution_nodes"][-1])}
    with pytest.raises(ValueError, match="transition"):
        validate_ir(dag, schedule, _rehash(ir, output_transitions=tuple(transitions)))
    ops = list(ir.operations)
    first_final = next(i for i, op in enumerate(ops) if op.kind == "FINALIZE_OUTPUT")
    ops[first_final] = replace(ops[first_final], kind="RECONSTRUCT")
    with pytest.raises(ValueError):
        validate_ir(dag, schedule, _rehash(ir, operations=tuple(ops)))
    _, dag, schedule, ir, _, _ = _record(corpus, algorithm="project-rank8", prefix="ed149344")
    recompute = list(ir.recomputation_records)
    recompute[0] = {**recompute[0], "input_values": ("missing-value",)}
    with pytest.raises(ValueError, match="recomputation"):
        validate_ir(dag, schedule, _rehash(ir, recomputation_records=tuple(recompute)))


def test_group_operation_order_and_buffer_alias_rejected(corpus):
    _, dag, schedule, ir, _, _ = _record(corpus)
    index = next(i for i, group in enumerate(ir.groups) if len(group.op_ids) > 1)
    groups = list(ir.groups)
    groups[index] = replace(groups[index], op_ids=tuple(reversed(groups[index].op_ids)))
    with pytest.raises(ValueError, match="group boundary"):
        validate_ir(dag, schedule, _rehash(ir, groups=tuple(groups)))
    buffers = list(ir.buffer_plan)
    first, second = [i for i, buffer in enumerate(buffers) if buffer.output_block is None][:2]
    buffers[second] = replace(buffers[second], buffer_id=buffers[first].buffer_id)
    with pytest.raises(ValueError, match="buffer"):
        validate_ir(dag, schedule, _rehash(ir, buffer_plan=tuple(buffers)))


def test_registry_determinism_and_explicit_unsupported(corpus):
    assert primitive_registry() == primitive_registry()
    assert len(primitive_registry()) == 6
    _, dag, schedule, ir, plan, _ = _record(corpus, algorithm="strassen", prefix="0c4cdc76")
    assert "ORDERED_ONE_GEMM" in plan.primitive_ids
    first = ir.groups[0]
    groups = (replace(first, intended_strategy="UNSUPPORTED"), *ir.groups[1:])
    changed = _rehash(ir, groups=groups)
    # An explicit unsupported group is retained as a lowerability result.
    assert match_group(changed, first.group_id) == (None, "GROUP_STRATEGY_UNSUPPORTED")
    rejected = plan_ir(dag, schedule, changed)
    assert rejected.mapping_status == "LOWERING_UNSUPPORTED"
    assert rejected.unsupported_reasons == (f"{first.group_id}:GROUP_STRATEGY_UNSUPPORTED",)


def test_legacy_comparison_and_hash_compatibility(corpus):
    statuses = Counter(compare_legacy_plan(legacy, plan)["status"] for _, _, _, _, plan, legacy in corpus)
    assert statuses["NOT_APPLICABLE"] == 40
    assert statuses["DIFFERENT"] == 16  # Old approximate physical paths are not exact IR executions.
    assert statuses["EQUIVALENT"] + statuses["SEMANTICALLY_EQUIVALENT_DIFFERENT_PLAN"] == 100
    for row, dag, schedule, ir, _, legacy in corpus:
        base = legacy.schedule_plan if isinstance(legacy, HBMBackedPlan) else legacy
        control = plan_ir(dag, schedule, ir, lowering_name=row["lowering_name"], prefer_legacy=True)
        if base.mapping_status != "UNSUPPORTED":
            assert control.legacy_plan_hash == legacy.lowering_plan_hash
            assert control.launch_sequence == base.launch_sequence
            assert control.mapping_status == base.mapping_status


def test_coverage_selection_failure_gate_and_architecture(corpus):
    classification = [{"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                       "matrix_size": row["matrix_size"], "ir_status": "IR_VALID",
                       "mapping_status": "EXACT_MAPPING", "unsupported_reasons": (),
                       "validation_status": "LOWERING_INCORRECT" if row["algorithm"] == "strassen" and
                       row["matrix_size"] >= 512 else "PASS"}
                      for row, *_ in corpus]
    assert sum(row["validation_status"] == "PASS" for row in classification) == 132
    high = high_ranked_coverage(classification)["totals"]
    assert high["static_top1"] == {"selected": 12, "ir_valid": 12,
                                   "exact_lowerable": 12, "correctness_passing": 10}
    assert high["static_top3"]["correctness_passing"] == 30
    assert high["static_pareto"]["correctness_passing"] == 90
    assert high["eager_old_mapped_top1"]["correctness_passing"] == 11
    assert high["graph_old_mapped_top1"]["correctness_passing"] == 12
    reasons = aggregate_ir_reasons(classification)
    assert reasons["ir_or_primitive_reasons"] == {}
    assert reasons["numerical_failure_by_algorithm_size"]["strassen:512"] == 12
    assert ir_architecture_decision(classification)["decision"] == "IR_ARCHITECTURE_VIABLE"
    assert ir_architecture_decision(classification) == ir_architecture_decision(classification)
    good = [{"seed": seed, "passed": True} for seed in (11, 23, 37)]
    assert validated_status(good, {"topology_agrees": True}, {"status": "PASS"}) == "PASS"
    assert validated_status(good[:2], {"topology_agrees": True}, {"status": "PASS"}) == "LOWERING_INCORRECT"
    assert validated_status(good, {"topology_agrees": False}, {"status": "PASS"}) == "LOWERING_INCORRECT"
    assert validated_status(good, {"topology_agrees": True}, {"status": "FAIL"}) == "LOWERING_INCORRECT"


@pytest.mark.parametrize("algorithm,prefix", [
    ("conventional", "8467ca32"), ("project-rank8", "30158351"),
    ("strassen", "0c4cdc76"), ("project-rank8", "fb511c78"),
    ("project-rank8", "ed149344")])
@pytest.mark.parametrize("size", [128, 256, 512, 1024])
def test_ir_gpu_correctness_and_topology(algorithm, prefix, size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torch.profiler import DeviceType, ProfilerActivity, profile
    from matmul_inspector.tensor_game_schedule_ir_runtime import allocate_ir, audit_ir_workspace, execute_ir

    source = next(row for row in source_rows() if row["algorithm"] == algorithm and
                  row["matrix_size"] == 128 and row["schedule_hash"].startswith(prefix))
    row = next(row for row in source_rows() if row["algorithm"] == algorithm and
               row["matrix_size"] == size and row["schedule"]["groups"] == source["schedule"]["groups"] and
               row["schedule"]["recompute_allowed"] == source["schedule"]["recompute_allowed"])
    dag = make_dag(algorithm, size, project_actions(FROZEN))
    schedule = _schedule(row["schedule"])
    ir = translate_schedule(dag, schedule)
    plan = plan_ir(dag, schedule, ir, lowering_name=row["lowering_name"])
    workspace = allocate_ir(dag, ir, plan)
    assert audit_ir_workspace(dag, schedule, ir, plan, workspace)["status"] == "PASS"
    for seed in (11, 23, 37):
        torch.manual_seed(seed)
        left = torch.randn((size, size), device="cuda", dtype=torch.float16)
        right = torch.randn_like(left)
        actual = execute_ir(dag, schedule, ir, plan, left, right, workspace)
        if algorithm == "strassen" and size >= 512:
            assert torch.isfinite(actual).all()  # Known fixed-tolerance failure is retained in study evidence.
        else:
            assert torch.allclose(actual, left @ right, rtol=0.005, atol=0.05)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        execute_ir(dag, schedule, ir, plan, left, right, workspace)
        torch.cuda.synchronize()
    names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
    assert len(names) == plan.expected_launch_count
    assert all(actual.startswith(launch["kernel"]) for actual, launch in zip(names, plan.launch_sequence))


@pytest.mark.parametrize("algorithm,prefix", [
    ("conventional", "5ebc4315"), ("project-rank8", "1406976b"),
    ("project-rank8", "3174902e"), ("project-rank8", "61e6aef2"),
    ("strassen", "d3013486")])
def test_ir_and_legacy_backend_output_agreement(algorithm, prefix):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from matmul_inspector.tensor_game_schedule_ir_runtime import (
        allocate_backend, execute_backend,
    )

    row = next(row for row in source_rows() if row["algorithm"] == algorithm and
               row["matrix_size"] == 256 and row["schedule_hash"].startswith(prefix))
    dag = make_dag(algorithm, 256, project_actions(FROZEN))
    schedule = _schedule(row["schedule"])
    ir = translate_schedule(dag, schedule)
    current = plan_ir(dag, schedule, ir, lowering_name=row["lowering_name"])
    control = plan_ir(dag, schedule, ir, lowering_name=row["lowering_name"], prefer_legacy=True)
    legacy = classify_hbm_schedule(dag, schedule, lowering_name=row["lowering_name"])
    current_workspace = allocate_backend(dag, schedule, ir, current, lowering_name=row["lowering_name"])
    control_workspace = allocate_backend(dag, schedule, ir, control, lowering_name=row["lowering_name"],
                                         legacy_plan=legacy)
    torch.manual_seed(11)
    left = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    actual = execute_backend(dag, schedule, ir, current, left, right, current_workspace,
                             lowering_name=row["lowering_name"])
    expected = execute_backend(dag, schedule, ir, control, left, right, control_workspace,
                               lowering_name=row["lowering_name"], legacy_plan=legacy)
    assert torch.allclose(actual, expected, rtol=0.005, atol=0.05)
