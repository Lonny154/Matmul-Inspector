"""CPU-only contracts for the fixed-address 3x3 CUDA Graph diagnostic."""

import copy
import shutil

import pytest

from matmul_inspector import experiment_pair
from scripts import run_mi_research_6b_graph as study
from scripts import analyze_mi_research_6b as analysis


def sample(block_ms):
    return {"cuda_event_elapsed_ms_per_block": block_ms,
            "cuda_event_elapsed_ms_per_operation": block_ms / study.OPERATIONS,
            "host_submission_ms_per_block": 0.02,
            "host_wall_ms_per_block": block_ms + 0.10,
            "cuda_event_minus_host_wall_ms_per_block": -0.10,
            "monotonic_raw_ms_per_block": block_ms + 0.10,
            "cuda_event_minus_monotonic_raw_ms_per_block": -0.10,
            "complete_operations_per_block": study.OPERATIONS,
            "host_submission_calls_per_block": 1}


def fixture_raw():
    checks = {"passed": True, "failing_elements": 0, "candidate_nonfinite_count": 0}
    clock = experiment_pair._preflight_summary([{
        "perf_counter_ms": 20.0, "monotonic_raw_ms": 20.0,
        "raw_minus_perf_ms": 0.0, "raw_over_perf": 1.0,
        "read_span_bound_ms": 0.01} for _ in range(5)])
    rounds = [{"round": i, "order": experiment_pair.pair_order(i),
               "measurements": {"A": sample(1.2 + i * 0.001),
                                "B": sample(0.8 + i * 0.001)}}
              for i in range(1, study.ROUNDS + 1)]
    aa = [{"round": i, "order": ["X", "Y"] if i % 2 else ["Y", "X"],
           "measurements": {"X": sample(1.2), "Y": sample(1.201)}}
          for i in range(1, study.ROUNDS + 1)]
    return {"schema_version": 1, "mode": "cuda_graph", "session": 1,
            "size": study.SIZE, "ranks": study.RANKS,
            "operations_per_replay": study.OPERATIONS,
            "warmups_per_graph": study.WARMUPS, "tolerance": study.TOLERANCE,
            "capture_order": ["A", "B"],
            "correctness": {"fixed_buffer_addresses_unchanged": True,
                            "cases": [{"case": name,
                                       "replays": {label: [copy.deepcopy(checks), copy.deepcopy(checks)]
                                                   for label in study.LABELS},
                                       "eager_agreement": {label: {"passed": True}
                                                           for label in study.LABELS}}
                                      for name in study.CASES]},
            "graph_profile": {label: {"kernel_events": study.OPERATIONS * (rank + 1),
                                      "product_events": study.OPERATIONS * rank,
                                      "reconstruction_events": study.OPERATIONS,
                                      "graph_launch_calls": 1}
                              for label, rank in study.RANKS.items()},
            "benchmark": {"mode": "cuda_graph_fixed_address",
                          "iterations_per_measurement": study.OPERATIONS,
                          "measurement_sequence": experiment_pair.expected_diagnostic_sequence(
                              study.ROUNDS, study.OPERATIONS, "before"),
                          "host_clock_preflight": clock,
                          "unranked_timing_controls": {"A": sample(1.2), "B": sample(0.8)},
                          "aa_kernel": "A", "aa_iterations_per_measurement": study.OPERATIONS,
                          "aa_rounds": aa, "rounds": rounds}}


def test_frozen_source_and_prior_evidence():
    assert study.artifacts.digest(study.SOURCE) == study.SOURCE_SHA256
    assert study.artifacts.digest(study.COEFFICIENTS) == study.COEFFICIENTS_SHA256
    assert study.eager.verify(study.PREVIOUS_BUNDLE)["status"] == "PASS"


@pytest.mark.parametrize("rank", [23, 27])
def test_profiler_sequence_requires_every_ordered_kernel(rank):
    names = [name for _ in range(study.OPERATIONS)
             for name in (["_block_product"] * rank + ["_reconstruct"])]
    trace = {"traceEvents": [{"cat": "cuda_runtime", "name": "cudaGraphLaunch"}]
             + [{"cat": "kernel", "name": name, "ts": index} for index, name in enumerate(names)]}
    assert study.trace_profile(trace, rank)["kernel_events"] == study.OPERATIONS * (rank + 1)
    bad = copy.deepcopy(trace)
    bad["traceEvents"][1]["name"] = "_reconstruct"
    with pytest.raises(ValueError, match="reordered"):
        study.trace_profile(bad, rank)
    bad = copy.deepcopy(trace)
    bad["traceEvents"].pop()
    with pytest.raises(ValueError, match="physical GPU kernel events"):
        study.trace_profile(bad, rank)
    bad = copy.deepcopy(trace)
    bad["traceEvents"].pop(0)
    with pytest.raises(ValueError, match="graph submission"):
        study.trace_profile(bad, rank)


def test_graph_measure_normalizes_one_replay(monkeypatch):
    calls = []
    def fake_measure(_torch, _function, iterations, *, raw_clock):
        calls.append(iterations)
        return {"cuda_event_elapsed_ms_per_operation": 1.0,
                "cuda_event_elapsed_ms_per_block": 1.0}
    monkeypatch.setattr(study.experiment.CudaTritonBackend, "_measure", staticmethod(fake_measure))
    result = study.graph_measure(None, None, study.OPERATIONS, raw_clock=None)
    assert calls == [1]
    assert result["cuda_event_elapsed_ms_per_operation"] == 1 / study.OPERATIONS
    assert result["complete_operations_per_block"] == study.OPERATIONS
    with pytest.raises(ValueError, match="operation count"):
        study.graph_measure(None, None, study.OPERATIONS - 1, raw_clock=None)


def test_existing_decision_and_strict_protocol():
    raw = fixture_raw()
    result = study.summarize(raw)
    assert result["decision"]["decision"] == "B_FASTER"
    assert result["decision"]["decision_schema_version"] == 3
    assert result["decision"]["timing_basis"].startswith("one fixed-address CUDA Graph")
    assert result["b_faster_rounds"] == study.ROUNDS
    bad = copy.deepcopy(raw)
    bad["benchmark"]["rounds"][0]["measurements"]["B"]["complete_operations_per_block"] -= 1
    with pytest.raises(ValueError, match="operation count"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["correctness"]["cases"][0]["replays"]["B"][1]["passed"] = False
    with pytest.raises(ValueError, match="numerical correctness"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["graph_profile"]["B"]["kernel_events"] -= 1
    with pytest.raises(ValueError, match="topology"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["capture_order"] = ["B", "A"]
    with pytest.raises(ValueError, match="protocol"):
        study.summarize(bad)


def test_retained_sessions_and_cross_session_replay(tmp_path):
    for session in (1, 2):
        bundle = analysis.EVIDENCE / f"graph-s{session}"
        result = study.verify(bundle)
        assert result == {"status": "PASS", "session": session,
                          "decision": "INCONCLUSIVE", "kernel_events": {"A": 560, "B": 480}}
    assert analysis.run(True) == {"status": "PASS", "classification": "C",
                                  "session_decisions": ["INCONCLUSIVE", "INCONCLUSIVE"]}
    bundle = tmp_path / "changed_bundle"
    shutil.copytree(analysis.EVIDENCE / "graph-s1", bundle)
    trace = bundle / "B_graph_trace.json"
    trace.write_bytes(trace.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(bundle)
