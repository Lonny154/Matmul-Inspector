"""Bounded GPU gates for the known HA-Schedule-5 rank-8 lowering pair."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from matmul_inspector.tensor_game_schedule_calibration import validation_assessment
from matmul_inspector.tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
from matmul_inspector.tensor_game_schedule_mechanism_study import RECOMPUTED, STORED, paired_summary, summarize_raw
from scripts.run_hardware_aware_schedule_calibration import calibration_profiles, source_data
from scripts.run_hardware_aware_schedule_mechanism_study import benchmark_graph_modes, frozen_lowering


@pytest.fixture(scope="module")
def pair():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    lowering = frozen_lowering()
    size = 128
    torch.manual_seed(771)
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn((size, size), device="cuda", dtype=torch.float16)
    workspaces = {STORED: allocate_workspace(lowering, size, "stored_form"),
                  RECOMPUTED: allocate_workspace(lowering, size, "recomputed_form")}
    eager = {STORED: lambda: execute_selected(lowering, left, right, workspaces[STORED], "stored_form"),
             RECOMPUTED: lambda: execute_selected(lowering, left, right, workspaces[RECOMPUTED], "recomputed_form")}
    return left, right, workspaces, eager


def test_fresh_eager_known_pair_correctness(pair):
    left, right, _, eager = pair
    reference = left @ right
    for function in eager.values():
        assert torch.allclose(function(), reference, rtol=5e-3, atol=5e-2)


def test_fresh_graph_known_pair_correctness(pair):
    left, right, workspaces, eager = pair
    graphs = {}
    for name, function in eager.items():
        function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graph.replay()
        graphs[name] = graph
    reference = left @ right
    assert all(torch.allclose(workspaces[name]["output"], reference, rtol=5e-3, atol=5e-2)
               for name in graphs)


def test_calibration_selected_known_candidate_benchmark_executes(pair):
    _, _, _, eager = pair
    graphs = {}
    for name, function in eager.items():
        function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        graphs[name] = graph.replay
    raw = benchmark_graph_modes(torch, session="TEST", size=128, eager=eager, graphs=graphs,
                                warmups=1, iterations=2, rounds=2, seed=17)
    summaries = summarize_raw(raw)
    assert len(raw) == 16
    assert len(paired_summary(summaries, "graph_eager")) == 1
    assert len(paired_summary(summaries, "graph_replay")) == 1


def test_failed_validation_downgrades_calibration(pair):
    profile = next(row for row in calibration_profiles(source_data())
                   if row["matrix_size"] == 128 and row["execution_mode"] == "EAGER")
    failed = validation_assessment(profile, {"ratio_bootstrap_95pct": [0.8, 0.9]})
    assert failed["status"] == "CONTRADICTED"
    assert failed["confidence_after_validation"] == "UNSTABLE"
