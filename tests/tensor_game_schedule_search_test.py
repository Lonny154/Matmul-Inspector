"""CPU-only CP-SAT schedule-search controls and independent accounting gates."""

from pathlib import Path
import json

import pytest

pytest.importorskip("ortools")

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_schedule_model import (
    BlockConfig, build_execution_dag, evaluate_schedule, make_schedule, reference_dag,
    validate_schedule,
)
from matmul_inspector.tensor_game_schedule_search import (
    OBJECTIVES, SearchConfig, eligible_recompute_forms, fixed_operation_order,
    rebind_schedule, solve_schedule,
)


PROJECT_RANK8 = Path(__file__).resolve().parents[1] / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"
CONFIG = BlockConfig(128, "fp16", "fp32", "fp16")
UNIT = CONFIG.block_bytes("fp16")


def dag(name, size=128):
    config = BlockConfig(size, "fp16", "fp32", "fp16")
    if name == "project-rank8":
        record = json.loads(PROJECT_RANK8.read_text().splitlines()[0])
        return build_execution_dag(name, tuple(RankOneAction(**action) for action in record["actions"]), config)
    return reference_dag(name, config)


def solve(name, units=4, budget_blocks=0, objective="LEX_TEMP_RECOMPUTE_LAUNCH", **kwargs):
    return solve_schedule(dag(name), SearchConfig(units * UNIT,
                                                  recomputation_budget_operations=budget_blocks * CONFIG.block_elements,
                                                  objective=objective, timeout_seconds=5, **kwargs))


@pytest.mark.parametrize("name", ["conventional", "strassen", "project-rank8"])
def test_recovered_reference_algorithms_are_legal_and_agree(name):
    outcome = solve(name)
    assert outcome.status == "OPTIMAL"
    assert outcome.agreement == "PASS"
    assert outcome.schedule is not None and outcome.metrics is not None
    assert validate_schedule(dag(name), outcome.schedule).valid
    assert outcome.metrics["validation_status"] == "PASS"
    for metric, predicted in outcome.predicted.items():
        assert outcome.metrics[metric] == predicted
    assert outcome.metrics["peak_abstract_local_bytes"] <= 4 * UNIT


def test_fixed_order_is_topological_and_group_count_is_canonical():
    source = dag("strassen")
    order = fixed_operation_order(source)
    assert len(order) == len([node for node in source.nodes if node.operation != "INPUT"])
    outcome = solve("strassen")
    assert tuple(group.group_id for group in outcome.schedule.groups) == tuple(
        f"G{index:03d}" for index in range(len(outcome.schedule.groups)))
    assert len(outcome.schedule.groups) == outcome.predicted["logical_launch_groups"]


def test_zero_capacity_forces_hbm_intermediates_and_boundaries():
    source = dag("conventional")
    outcome = solve_schedule(source, SearchConfig(0, timeout_seconds=5))
    assert outcome.agreement == "PASS"
    assert outcome.predicted["peak_abstract_local_bytes"] == 0
    assert all(op.location == "HBM" for group in outcome.schedule.groups for op in group.operations)
    assert outcome.metrics["temporary_hbm_traffic_bytes"] == evaluate_schedule(
        source, make_schedule(source, "MATERIALIZE_ALL"))["temporary_hbm_traffic_bytes"]
    assert outcome.predicted["logical_launch_groups"] >= len(source.terms)


@pytest.mark.parametrize("name", ["conventional", "strassen", "project-rank8"])
def test_capacity_sweep_is_bounded_and_improves_traffic(name):
    tight = solve(name, units=1)
    roomy = solve(name, units=4)
    assert tight.agreement == roomy.agreement == "PASS"
    assert tight.metrics["peak_abstract_local_bytes"] <= UNIT
    assert roomy.metrics["peak_abstract_local_bytes"] <= 4 * UNIT
    assert roomy.metrics["temporary_hbm_traffic_bytes"] < tight.metrics["temporary_hbm_traffic_bytes"]


def test_cross_group_uses_are_hbm_or_explicit_recompute():
    source = dag("project-rank8")
    outcome = solve("project-rank8", budget_blocks=1)
    group_of = {op.node_id: index for index, group in enumerate(outcome.schedule.groups)
                for op in group.operations if op.location != "EPHEMERAL"}
    hbm = {op.node_id for group in outcome.schedule.groups for op in group.operations if op.location == "HBM"}
    for node in source.nodes:
        if node.node_id not in group_of:
            continue
        for dep in node.dependencies:
            if dep in group_of and group_of[dep] < group_of[node.node_id]:
                assert dep in hbm or dep in outcome.schedule.recompute_allowed
    assert validate_schedule(source, outcome.schedule).valid


def test_recompute_positive_control_counts_work_and_removes_hbm_form():
    source = dag("project-rank8")
    no_recompute = solve("project-rank8", budget_blocks=0)
    enabled = solve("project-rank8", budget_blocks=1)
    assert set(enabled.schedule.recompute_allowed) <= set(eligible_recompute_forms(source))
    assert len(enabled.schedule.recompute_allowed) == 1
    assert enabled.metrics["number_of_recomputed_intermediates"] == 1
    assert enabled.metrics["recomputation_arithmetic_operations"] == CONFIG.block_elements
    assert enabled.metrics["extra_source_read_bytes"] == 2 * UNIT
    assert enabled.metrics["temporary_hbm_traffic_bytes"] == 368640
    assert no_recompute.metrics["temporary_hbm_traffic_bytes"] == 393216
    assert enabled.metrics["temporary_hbm_bytes_written"] < no_recompute.metrics["temporary_hbm_bytes_written"]
    assert enabled.metrics["logical_launch_groups"] == 8


def test_manual_fusion_point_recovered_when_capacity_permits():
    for name in ("conventional", "strassen", "project-rank8"):
        source = dag(name)
        manual = evaluate_schedule(source, make_schedule(source, "FUSE_INPUT_AND_OUTPUT"))
        outcome = solve(name, units=4)
        assert outcome.metrics["temporary_hbm_traffic_bytes"] <= manual["temporary_hbm_traffic_bytes"]


def test_lexicographic_stages_freeze_optima_and_min_recompute_cap():
    result = solve("project-rank8", budget_blocks=1)
    assert [stage["status"] for stage in result.stages] == ["OPTIMAL"] * 3
    assert [stage["objective"] for stage in result.stages] == [
        "temporary_hbm_traffic_bytes", "recomputation_arithmetic_operations", "logical_launch_groups"]
    assert [stage["value"] for stage in result.stages] == [
        result.metrics["temporary_hbm_traffic_bytes"], result.metrics["recomputation_arithmetic_operations"],
        result.metrics["logical_launch_groups"]]
    constrained = solve("project-rank8", budget_blocks=1, objective="MIN_RECOMPUTE_OPS",
                        temporary_traffic_cap_bytes=368640)
    assert constrained.agreement == "PASS"
    assert constrained.metrics["recomputation_arithmetic_operations"] == CONFIG.block_elements


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_supported_objectives_have_independent_agreement(objective):
    kwargs = {"temporary_traffic_cap_bytes": 868352} if objective == "MIN_RECOMPUTE_OPS" else {}
    outcome = solve("strassen", units=4, objective=objective, **kwargs)
    assert outcome.status == "OPTIMAL"
    assert outcome.agreement == "PASS"


def test_normalized_reuse_scales_metrics_and_rehashes():
    source = dag("strassen", 128)
    target = dag("strassen", 256)
    outcome = solve("strassen")
    rebound = rebind_schedule(source, target, outcome.schedule)
    larger = evaluate_schedule(target, rebound)
    assert rebound.schedule_hash != outcome.schedule.schedule_hash
    assert larger["temporary_hbm_traffic_bytes"] == 4 * outcome.metrics["temporary_hbm_traffic_bytes"]
    assert larger["peak_abstract_local_bytes"] == 4 * outcome.metrics["peak_abstract_local_bytes"]
    assert larger["logical_launch_groups"] == outcome.metrics["logical_launch_groups"]


def test_deterministic_hashes_and_solver_counts():
    first = solve("conventional")
    second = solve("conventional")
    assert first.schedule.schedule_hash == second.schedule.schedule_hash
    assert first.solver_summary["variables"] > 0
    assert first.solver_summary["constraints"] > 0
    assert first.solver_summary["workers"] == 1
    assert first.solver_summary["version"]


def test_invalid_validator_and_disagreement_are_rejected(monkeypatch):
    import matmul_inspector.tensor_game_schedule_search as search
    original = search.validate_schedule
    monkeypatch.setattr(search, "validate_schedule", lambda *_: type("Bad", (), {"valid": False, "issues": ()})())
    invalid = solve("conventional")
    assert invalid.status == "MODEL_INVALID" and invalid.schedule is None
    monkeypatch.setattr(search, "validate_schedule", original)
    evaluator = search.evaluate_schedule

    def inconsistent(*args):
        result = evaluator(*args)
        result["temporary_hbm_traffic_bytes"] += 1
        return result

    monkeypatch.setattr(search, "evaluate_schedule", inconsistent)
    mismatch = solve("conventional")
    assert mismatch.status == "MODEL_INVALID" and mismatch.agreement == "FAIL"
    assert mismatch.schedule is None


def test_unknown_and_timeout_never_report_infeasible(monkeypatch):
    from ortools.sat.python import cp_model
    real_solve = cp_model.CpSolver.Solve

    def unknown(self, model):
        real_solve(self, model)  # Populate response statistics for the adapter.
        return cp_model.UNKNOWN

    monkeypatch.setattr(cp_model.CpSolver, "Solve", unknown)
    outcome = solve("conventional")
    assert outcome.status == "UNKNOWN"
    assert outcome.schedule is None


def test_recompute_objective_requires_a_traffic_constraint():
    with pytest.raises(ValueError, match="traffic cap"):
        SearchConfig(UNIT, objective="MIN_RECOMPUTE_OPS")


def test_pareto_deduplicates_equal_static_vectors(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    command = importlib.import_module("run_hardware_aware_schedule_search")
    base = {"temporary_hbm_traffic_bytes": 100, "recomputation_arithmetic_operations": 0,
            "peak_abstract_local_bytes": 8, "logical_launch_groups": 2}
    candidates = [{**base, "source": "manual", "policy": "A", "schedule_hash": "a"},
                  {**base, "source": "solver", "run_id": "R1", "schedule_hash": "b"},
                  {**base, "temporary_hbm_traffic_bytes": 120, "source": "solver",
                   "run_id": "R2", "schedule_hash": "c"}]
    first = command.pareto_frontier(candidates)
    assert first == command.pareto_frontier(list(reversed(candidates)))
    assert len(first) == 1
    assert {row["schedule_hash"] for row in first[0]["equivalent_schedules"]} == {"a", "b"}
