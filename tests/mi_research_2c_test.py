"""CPU-only capture, timing, and offline-replay checks for MI-Research-2C."""

import json

import pytest

from matmul_inspector import experiment_pair
from scripts import run_mi_research_2c_graph as graph


def measurement(total_ms):
    wall = total_ms + 0.05
    return {"cuda_event_elapsed_ms_per_block": total_ms,
            "cuda_event_elapsed_ms_per_operation": total_ms / graph.OPERATIONS,
            "host_submission_ms_per_block": 0.04,
            "host_wall_ms_per_block": wall,
            "cuda_event_minus_host_wall_ms_per_block": total_ms - wall,
            "monotonic_raw_ms_per_block": wall,
            "cuda_event_minus_monotonic_raw_ms_per_block": total_ms - wall,
            "complete_operations_per_block": graph.OPERATIONS,
            "host_submission_calls_per_block": 1}


def trace(label):
    names = (["fc1_kernel"] * graph.OPERATIONS if label == "A" else
             [name for _ in range(graph.OPERATIONS)
              for name in ("split_k_partial", "reduce_partials")])
    return {"traceEvents": [{"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": -1}] +
            [{"cat": "kernel", "name": name, "ts": index} for index, name in enumerate(names)]}


def fixture_raw():
    preflight = experiment_pair._preflight_summary([
        {"perf_counter_ms": 20.0, "monotonic_raw_ms": 20.0,
         "raw_minus_perf_ms": 0.0, "raw_over_perf": 1.0, "read_span_bound_ms": 0.001}
        for _ in range(5)])
    rounds = []
    aa_rounds = []
    for index in range(1, graph.ROUNDS + 1):
        rounds.append({"round": index, "order": experiment_pair.pair_order(index),
                       "measurements": {"A": measurement(2.0), "B": measurement(1.0)}})
        aa_rounds.append({"round": index,
                          "order": ["X", "Y"] if index % 2 else ["Y", "X"],
                          "measurements": {"X": measurement(2.0), "Y": measurement(2.02)}})
    return {"schema_version": 1, "mode": "cuda_graph", "session": 1,
            "operations_per_replay": graph.OPERATIONS, "rounds": graph.ROUNDS,
            "warmups_per_graph": graph.WARMUPS,
            "correctness": {"passed": True, "seeds": [
                {"seed": seed, "checks": {label: {"passed": True} for label in ("A", "B")}}
                for seed in range(3)]},
            "graph_profile": {label: graph.require_complete_trace(trace(label), label)
                              for label in ("A", "B")},
            "benchmark": {
                "iterations_per_measurement": graph.OPERATIONS,
                "measurement_sequence": experiment_pair.expected_diagnostic_sequence(
                    graph.ROUNDS, graph.OPERATIONS, "before"),
                "host_clock_preflight": preflight,
                "unranked_timing_controls": {label: measurement(value)
                                             for label, value in (("A", 2.0), ("B", 1.0))},
                "aa_rounds": aa_rounds, "aa_kernel": "A",
                "aa_iterations_per_measurement": graph.OPERATIONS,
                "rounds": rounds}}


def test_one_replay_normalizes_to_one_hundred_complete_operations():
    original = {"cuda_event_elapsed_ms_per_block": 2.5,
                "cuda_event_elapsed_ms_per_operation": 2.5,
                "host_submission_ms_per_block": 0.2}
    result = graph.normalize_replay_measurement(original)
    assert result["cuda_event_elapsed_ms_per_operation"] == 0.025
    assert result["cuda_event_elapsed_ms_per_block"] == 2.5
    assert result["host_submission_ms_per_block"] == 0.2
    assert result["host_submission_calls_per_block"] == 1
    with pytest.raises(ValueError, match="one measured replay"):
        graph.normalize_replay_measurement({**original,
                                            "cuda_event_elapsed_ms_per_operation": 0.025})


def test_profile_requires_both_split_kernels_in_order_for_every_operation():
    assert graph.require_complete_trace(trace("A"), "A")["kernel_nodes_observed"] == 100
    assert graph.require_complete_trace(trace("B"), "B")["kernel_nodes_observed"] == 200
    missing_reduction = trace("B")
    missing_reduction["traceEvents"].pop()
    with pytest.raises(ValueError, match="complete ordered"):
        graph.require_complete_trace(missing_reduction, "B")
    wrong_order = trace("B")
    wrong_order["traceEvents"][1]["name"] = "reduce_partials"
    with pytest.raises(ValueError, match="complete ordered"):
        graph.require_complete_trace(wrong_order, "B")


def test_graph_summary_uses_existing_strict_paired_decision_and_controls():
    raw = fixture_raw()
    decision = graph.summarize(raw)["decision"]
    assert decision["decision"] == "B_FASTER"
    assert decision["aa_control"]["rounds"] == graph.ROUNDS
    assert len(decision["paired_A_minus_B_ms"]) == graph.ROUNDS
    raw["benchmark"]["rounds"][0]["measurements"]["B"]["host_submission_calls_per_block"] = 100
    with pytest.raises(ValueError, match="operation count"):
        graph.summarize(raw)


def test_offline_manifest_trace_and_decision_replay(tmp_path):
    raw = fixture_raw()
    for name in ("source.py", "experiment.json", "diagnostic_script.py"):
        (tmp_path / name).write_text(f"{name}\n")
    for field, name in (("source_sha256", "source.py"), ("spec_sha256", "experiment.json"),
                        ("diagnostic_script_sha256", "diagnostic_script.py")):
        raw[field] = graph.digest(tmp_path / name)
    raw["compiler"] = {}
    for name in ("conventional", "partial", "reduction"):
        (tmp_path / f"{name}.ptx").write_text(f"{name}\n")
        raw["compiler"][name] = {"ptx_sha256": graph.digest(tmp_path / f"{name}.ptx")}
    raw["graph_trace_sha256"] = {}
    for label in ("A", "B"):
        graph.write_json(tmp_path / f"{label}_trace.json", trace(label))
        raw["graph_trace_sha256"][label] = graph.digest(tmp_path / f"{label}_trace.json")
    graph.write_json(tmp_path / "raw.json", raw)
    graph.write_json(tmp_path / "summary.json", graph.summarize(raw))
    graph.manifest(tmp_path)
    assert graph.verify(tmp_path)["decision"] == "B_FASTER"
    trace_data = json.loads((tmp_path / "B_trace.json").read_text())
    trace_data["traceEvents"].pop()
    graph.write_json(tmp_path / "B_trace.json", trace_data)
    with pytest.raises(ValueError, match="manifest"):
        graph.verify(tmp_path)
    graph.manifest(tmp_path)
    with pytest.raises(ValueError, match="complete ordered"):
        graph.verify(tmp_path)
