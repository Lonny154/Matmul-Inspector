"""CPU-first tests for bounded schedule-family lowering and evidence gates."""

from __future__ import annotations

from collections import Counter

import pytest

from matmul_inspector.tensor_game_schedule_gpu_study import _schedule
from matmul_inspector.tensor_game_schedule_lowering import (
    FAMILY_LEGACY, FAMILY_MATERIALIZED, aggregate_graph_sessions, benchmarkable,
    capability_manifest, classify_schedule, coverage_summary, lowerability_query, plan_record,
)
from matmul_inspector.tensor_game_schedule_model import BlockConfig, make_schedule, reference_dag
from scripts.run_hardware_aware_schedule_lowering_study import FROZEN, graph_sessions, make_dag, project_actions, source_rows


@pytest.fixture(scope="module")
def corpus():
    actions = project_actions(FROZEN)
    records = []
    for row in source_rows():
        dag = make_dag(row["algorithm"], row["matrix_size"], actions)
        schedule = _schedule(row["schedule"])
        plan = classify_schedule(dag, schedule, lowering_name=row["lowering_name"],
                                 static_metrics=row["metrics"], calibration_features=row["features"])
        records.append((row, dag, schedule, plan))
    return records


def test_every_retained_class_is_deterministic_and_hash_linked(corpus):
    assert len(corpus) == 156
    assert Counter((row["algorithm"], row["matrix_size"]) for row, *_ in corpus) == {
        (algorithm, size): count for algorithm, count in
        (("conventional", 10), ("strassen", 12), ("project-rank8", 17))
        for size in (128, 256, 512, 1024)}
    for row, dag, schedule, plan in corpus:
        assert plan == classify_schedule(dag, schedule, lowering_name=row["lowering_name"],
                                         static_metrics=row["metrics"], calibration_features=row["features"])
        assert plan.schedule_hash == row["schedule_hash"] == plan_record(plan)["schedule_hash"]
        assert len(plan.lowering_plan_hash) == 64
        assert plan.static_metrics_hash and plan.calibration_features_hash


def test_exact_requires_singleton_hbm_semantics(corpus):
    for row, _, schedule, plan in corpus:
        if plan.mapping_status == "EXACT_MAPPING":
            assert plan.family_id == FAMILY_MATERIALIZED
            assert not schedule.recompute_allowed
            assert all(len(group.operations) == 1 and group.operations[0].location == "HBM"
                       for group in schedule.groups)
            assert plan.expected_launch_count == len(schedule.groups)
        if row["lowering_name"]:
            assert plan.family_id == FAMILY_LEGACY
            assert plan.mapping_status == "APPROXIMATE_MAPPING"
            assert "PRODUCT_AND_RECONSTRUCTION_PLACEMENT_DIFFERS" in plan.unsupported_features
    row, dag, schedule, _ = next(item for item in corpus if item[0]["algorithm"] == "strassen"
                                 and item[0]["matrix_size"] == 128 and item[0]["lowering_name"] is None)
    with pytest.raises(ValueError, match="exact retained schedule hash"):
        classify_schedule(dag, schedule, lowering_name="strassen_reference")


def test_coverage_manifest_and_future_query(corpus):
    families = {family["family_id"] for family in capability_manifest()["families"]}
    assert families == {FAMILY_MATERIALIZED, FAMILY_LEGACY}
    for size in (128, 256, 512, 1024):
        plans = [plan for row, _, _, plan in corpus if row["matrix_size"] == size]
        coverage = coverage_summary(plans)
        assert (coverage["legal"], coverage["EXACT_MAPPING"], coverage["APPROXIMATE_MAPPING"],
                coverage["UNSUPPORTED"]) == (39, 5, 4, 30)
        assert set(coverage["by_algorithm"]) == {"conventional", "strassen", "project-rank8"}
    for row, dag, schedule, plan in corpus:
        answer = lowerability_query(dag, schedule, lowering_name=row["lowering_name"])
        assert answer["supported"] == (plan.mapping_status != "UNSUPPORTED")
        assert answer["exact"] == (plan.mapping_status == "EXACT_MAPPING")
        if not answer["supported"]:
            assert answer["reasons"]


def test_duplicate_semantics_share_family_but_not_hash(corpus):
    group = [(row, plan) for row, _, _, plan in corpus if row["algorithm"] == "conventional"
             and row["matrix_size"] == 128 and row["policy"] in ("MATERIALIZE_ALL", "FUSE_INPUT_PREPARATION")]
    assert len(group) == 2
    assert {plan.family_id for _, plan in group} == {FAMILY_MATERIALIZED}
    assert len({plan.schedule_hash for _, plan in group}) == 2


def test_unsupported_cannot_be_benchmarked_and_failed_correctness_is_excluded(corpus):
    unsupported = next(plan for _, _, _, plan in corpus if plan.mapping_status == "UNSUPPORTED")
    exact = next(plan for _, _, _, plan in corpus if plan.mapping_status == "EXACT_MAPPING")
    passing = [{"lowering_plan_hash": exact.lowering_plan_hash, "seed": seed, "passed": True}
               for seed in (11, 23, 37)]
    assert benchmarkable(exact, passing)
    assert not benchmarkable(unsupported, [{**row, "lowering_plan_hash": unsupported.lowering_plan_hash}
                                            for row in passing])
    assert not benchmarkable(exact, passing[:-1])
    assert not benchmarkable(exact, [*passing[:-1], {**passing[-1], "passed": False}])


def test_cpu_query_rejects_unimplemented_dtype():
    dag = reference_dag("conventional", BlockConfig(128, "fp32", "fp32", "fp32"))
    schedule = make_schedule(dag, "MATERIALIZE_ALL")
    query = lowerability_query(dag, schedule)
    assert not query["supported"] and "DTYPE_TEMPLATE_MISSING" in query["reasons"]


def test_graph_aggregation_preserves_sessions_and_contradictions():
    previous = graph_sessions(None)
    assert len(previous) == 20
    before = aggregate_graph_sessions(previous)
    assert [row["session_count"] for row in before] == [5] * 4
    contradiction = {"session_id": "CONTRADICTION", "matrix_size": 512,
                     "median_stored_over_recomputed_ratio": 1.1,
                     "ratio_bootstrap_95pct": [1.05, 1.15], "source": "synthetic-test"}
    after = aggregate_graph_sessions([*previous, contradiction])
    case = next(row for row in after if row["matrix_size"] == 512)
    assert case["session_count"] == 6 and case["confidence"] == "DISAGREES"
    assert any(row["session_id"] == "CONTRADICTION" for row in case["sessions"])
    assert len(previous) == 20


@pytest.mark.parametrize("algorithm", ["conventional", "strassen", "project-rank8"])
def test_materialized_template_gpu_smoke(algorithm):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from matmul_inspector.tensor_game_schedule_lowering_runtime import allocate_materialized, execute_materialized

    row = next(row for row in source_rows() if row["algorithm"] == algorithm and row["matrix_size"] == 128
               and row["policy"] == "MATERIALIZE_ALL")
    dag = make_dag(algorithm, 128, project_actions(FROZEN))
    schedule = _schedule(row["schedule"])
    plan = classify_schedule(dag, schedule)
    workspace = allocate_materialized(dag)
    for seed in (11, 23, 37):
        torch.manual_seed(seed)
        left = torch.randn((128, 128), device="cuda", dtype=torch.float16)
        right = torch.randn_like(left)
        actual = execute_materialized(dag, schedule, plan, left, right, workspace)
        assert torch.allclose(actual, left @ right, rtol=0.005, atol=0.05)
    if algorithm == "conventional":
        from torch.profiler import DeviceType, ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
            execute_materialized(dag, schedule, plan, left, right, workspace)
            torch.cuda.synchronize()
        actual_names = [event.name for event in recorded.events() if event.device_type == DeviceType.CUDA]
        assert len(actual_names) == plan.expected_launch_count
        assert all(name.startswith(item["kernel"]) for name, item in zip(actual_names, plan.launch_sequence))


@pytest.mark.parametrize("name", ["conventional_reference", "strassen_reference", "rank8_stored", "rank8_recomputed"])
def test_existing_mapping_gpu_correctness(name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from scripts.run_hardware_aware_schedule_mechanism_study import frozen_lowering

    lower = frozen_lowering() if name.startswith("rank8_") else reference_lowering(name.split("_")[0])
    mode = {"conventional_reference": "reference", "strassen_reference": "reference",
            "rank8_stored": "stored_form", "rank8_recomputed": "recomputed_form"}[name]
    workspace = allocate_workspace(lower, 128, mode)
    torch.manual_seed(11)
    left = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    actual = execute_selected(lower, left, right, workspace, mode)
    assert torch.allclose(actual, left @ right, rtol=0.005, atol=0.05)
    if name.startswith("rank8_"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute_selected(lower, left, right, workspace, mode)
        graph.replay()
        assert torch.allclose(workspace["output"], left @ right, rtol=0.005, atol=0.05)
