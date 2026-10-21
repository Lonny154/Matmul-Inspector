"""CPU-only legality and exact integer accounting for schedule candidates."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_schedule_model import (
    BlockConfig, LaunchGroup, POLICIES, ScheduledOp, ValidationIssue, ValidationResult,
    _digest, _schedule_payload,
    build_execution_dag, evaluate_schedule, make_schedule, reference_dag,
    validate_dag, validate_schedule,
)


PROJECT_RANK8 = Path(__file__).resolve().parents[1] / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"


def revised(schedule, groups):
    pending = replace(schedule, groups=tuple(groups), schedule_hash="")
    return replace(pending, schedule_hash=_digest(_schedule_payload(pending)))


@pytest.mark.parametrize("algorithm,rank", [("conventional", 8), ("strassen", 7)])
def test_reference_dag_exact_topological_outputs_and_hash(algorithm, rank):
    dag = reference_dag(algorithm, BlockConfig(128, "fp16", "fp32", "fp16"))
    assert validate_dag(dag).valid
    assert dag.dag_hash == reference_dag(algorithm, dag.config).dag_hash
    assert len(dag.terms) == rank
    assert dag.output_ids == ("O0", "O1", "O2", "O3")
    assert {node.operation for node in dag.nodes} == {"INPUT", "LINEAR_FORM", "MATMUL", "ACCUMULATE", "OUTPUT"} if algorithm == "strassen" else {"INPUT", "MATMUL", "ACCUMULATE", "OUTPUT"}


@pytest.mark.parametrize("algorithm", ["conventional", "strassen"])
@pytest.mark.parametrize("policy", POLICIES)
def test_reference_policies_legal_and_deterministic(algorithm, policy):
    dag = reference_dag(algorithm, BlockConfig(128, "fp16", "fp32", "fp16"))
    schedule = make_schedule(dag, policy)
    assert validate_schedule(dag, schedule).valid
    assert schedule.schedule_hash == make_schedule(dag, policy).schedule_hash
    assert evaluate_schedule(dag, schedule) == evaluate_schedule(dag, schedule)
    assert evaluate_schedule(dag, schedule)["validation_status"] == "PASS"


def test_cross_group_on_chip_lifetime_rejected():
    dag = reference_dag("conventional", BlockConfig(2))
    baseline = make_schedule(dag, "MATERIALIZE_ALL")
    groups = list(baseline.groups)
    groups[0] = replace(groups[0], operations=(ScheduledOp("M00", "ON_CHIP"),))
    invalid = revised(baseline, groups)
    assert "UNAVAILABLE_VALUE" in {issue.code for issue in validate_schedule(dag, invalid).issues}
    with pytest.raises(ValueError, match="invalid schedule"):
        evaluate_schedule(dag, invalid)


def test_missing_producer_and_unknown_node_rejected():
    dag = reference_dag("conventional", BlockConfig(2))
    baseline = make_schedule(dag, "MATERIALIZE_ALL")
    missing = revised(baseline, baseline.groups[1:])
    codes = {issue.code for issue in validate_schedule(dag, missing).issues}
    assert {"MISSING_REQUIRED_NODE", "UNAVAILABLE_VALUE"} <= codes
    unknown = revised(baseline, (LaunchGroup("bad", (ScheduledOp("UNKNOWN", "HBM"),)),) + baseline.groups)
    assert "UNKNOWN_NODE" in {issue.code for issue in validate_schedule(dag, unknown).issues}


def test_recompute_executes_again_without_hbm_materialization():
    first = json.loads(PROJECT_RANK8.read_text().splitlines()[0])
    dag = build_execution_dag("project-rank8", tuple(RankOneAction(**a) for a in first["actions"]), BlockConfig(2))
    baseline = evaluate_schedule(dag, make_schedule(dag, "MATERIALIZE_ALL"))
    fused = evaluate_schedule(dag, make_schedule(dag, "FUSE_INPUT_AND_OUTPUT"))
    recompute = evaluate_schedule(dag, make_schedule(dag, "RECOMPUTE_CHEAP_FORMS"))
    assert recompute["number_of_recomputed_intermediates"] >= 1
    assert recompute["recomputation_arithmetic_operations"] >= 1
    assert recompute["linear_form_arithmetic_operations"] > fused["linear_form_arithmetic_operations"]
    assert recompute["temporary_hbm_bytes_written"] < fused["temporary_hbm_bytes_written"]
    assert recompute["recomputed_hbm_write_bytes_avoided"] > 0
    assert recompute["recomputed_hbm_reload_bytes_avoided"] > 0
    assert recompute["extra_source_read_bytes"] > 0
    assert baseline["hbm_write_bytes"] > recompute["hbm_write_bytes"]


def test_hand_calculated_conventional_hbm_and_lifetimes():
    dag = reference_dag("conventional", BlockConfig(2))
    result = evaluate_schedule(dag, make_schedule(dag, "MATERIALIZE_ALL"))
    # One 1x1 FP32 block is 4 bytes. Eight products precede reconstruction.
    assert result["hbm_read_bytes"] == 128
    assert result["hbm_write_bytes"] == 80
    assert result["temporary_hbm_bytes_written"] == 64
    assert result["temporary_hbm_bytes_read"] == 64
    assert result["final_output_bytes"] == 16
    assert result["peak_temp_hbm_bytes"] == 36  # Eight products + first accumulator at birth.
    assert result["peak_simultaneously_live_intermediates"] == 9
    assert result["number_of_materialized_intermediates"] == 16


def test_scaling_and_dtype_bytes():
    small = reference_dag("strassen", BlockConfig(128, "fp16", "fp32", "fp16"))
    large = reference_dag("strassen", BlockConfig(256, "fp16", "fp32", "fp16"))
    small_cost = evaluate_schedule(small, make_schedule(small, "MATERIALIZE_ALL"))
    large_cost = evaluate_schedule(large, make_schedule(large, "MATERIALIZE_ALL"))
    assert large_cost["input_block_bytes"] == 4 * small_cost["input_block_bytes"]
    assert large_cost["temporary_hbm_traffic_bytes"] == 4 * small_cost["temporary_hbm_traffic_bytes"]
    assert large_cost["linear_form_arithmetic_operations"] == 4 * small_cost["linear_form_arithmetic_operations"]
    assert large_cost["reconstruction_arithmetic_operations"] == 4 * small_cost["reconstruction_arithmetic_operations"]
    fp32 = reference_dag("strassen", BlockConfig(128, "fp32", "fp32", "fp32"))
    fp32_cost = evaluate_schedule(fp32, make_schedule(fp32, "MATERIALIZE_ALL"))
    assert fp32_cost["input_block_bytes"] == 2 * small_cost["input_block_bytes"]
    assert fp32_cost["product_accumulation_block_bytes"] == small_cost["product_accumulation_block_bytes"]
    assert fp32_cost["output_block_bytes"] == 2 * small_cost["output_block_bytes"]


def test_invalid_schedule_excluded_from_aggregate(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_hardware_aware_schedule_study.py"
    spec = importlib.util.spec_from_file_location("schedule_study", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_validator = module.validate_schedule

    def reject_one(dag, schedule):
        if schedule.policy == "FUSE_INPUT_PREPARATION":
            return ValidationResult(False, (ValidationIssue("TEST_REJECTION", "deliberate"),))
        return real_validator(dag, schedule)

    monkeypatch.setattr(module, "validate_schedule", reject_one)
    study = module.collect_study(("conventional",), (2,), "fp32", "fp32", "fp32", POLICIES)
    assert study["aggregate"]["normal_result_count"] == 4
    assert len(study["aggregate"]["validation_failures"]) == 1
    assert all(row["policy"] != "FUSE_INPUT_PREPARATION" for row in study["results"])
