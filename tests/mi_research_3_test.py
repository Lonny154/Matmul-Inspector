"""CPU-only shape, graph completeness, paired decisions, and offline replay."""

import copy
import json

import pytest

from matmul_inspector import experiment_pair
from scripts import run_mi_research_3_graph as study


def measurement(total_ms):
    wall = total_ms + 0.05
    return {"cuda_event_elapsed_ms_per_block": total_ms,
            "cuda_event_elapsed_ms_per_operation": total_ms / study.OPERATIONS,
            "host_submission_ms_per_block": 0.04,
            "host_wall_ms_per_block": wall,
            "cuda_event_minus_host_wall_ms_per_block": total_ms - wall,
            "monotonic_raw_ms_per_block": wall,
            "cuda_event_minus_monotonic_raw_ms_per_block": total_ms - wall,
            "complete_operations_per_block": study.OPERATIONS,
            "host_submission_calls_per_block": 1}


def trace(label):
    names = {"A": ["fc1_kernel"], "B": ["split_k_partial", "reduce_partials"],
             "C": ["gemm_kernel"]}[label] * study.OPERATIONS
    return {"traceEvents": [{"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": -1}] +
            [{"cat": "kernel", "name": name, "ts": index} for index, name in enumerate(names)]}


def fixture_raw():
    preflight = experiment_pair._preflight_summary([
        {"perf_counter_ms": 20.0, "monotonic_raw_ms": 20.0,
         "raw_minus_perf_ms": 0.0, "raw_over_perf": 1.0, "read_span_bound_ms": 0.001}
        for _ in range(5)])
    pairs = {}
    for key, physical in study.PAIR_NAMES.items():
        aa, rounds = [], []
        for index in range(1, study.ROUNDS + 1):
            rounds.append({"round": index, "order": experiment_pair.pair_order(index),
                           "measurements": {"A": measurement(2.0), "B": measurement(1.0)}})
            aa.append({"round": index, "order": ["X", "Y"] if index % 2 else ["Y", "X"],
                       "measurements": {"X": measurement(2.0), "Y": measurement(2.02)}})
        pairs[key] = {"physical_labels": list(physical),
                      "measurement_sequence": experiment_pair.expected_diagnostic_sequence(
                          study.ROUNDS, study.OPERATIONS, "before"),
                      "unranked_timing_controls": {"A": measurement(2.0), "B": measurement(1.0)},
                      "aa_rounds": aa, "rounds": rounds}
    return {"schema_version": 1, "mode": "cuda_graph", "m": 32,
            "k": study.K, "n": study.N, "session": 1,
            "geometry": study.shape_geometry(32), "phase_order": study.pair_phase_order(1),
            "operations_per_replay": study.OPERATIONS, "rounds": study.ROUNDS,
            "warmups_per_graph": study.WARMUPS,
            "correctness": {"passed": True, "seeds": [
                {"seed": seed, "checks": {label: {"passed": True} for label in ("A", "B", "C")}}
                for seed in range(3)]},
            "host_clock_preflight": preflight, "pairs": pairs,
            "graph_setup_ms": {label: 1.0 for label in ("A", "B", "C")},
            "graph_profile": {label: study.graph_profile(trace(label), label) for label in ("A", "B", "C")}}


def test_frozen_shapes_and_counterbalanced_phases():
    assert [study.shape_geometry(m)["conventional_ctas"] for m in study.SHAPES] == [16, 32, 64, 128]
    assert study.shape_geometry(512)["scratch_bytes"] == 8 * 1024 * 1024
    assert study.pair_phase_order(1) == list(reversed(study.pair_phase_order(2)))
    with pytest.raises(ValueError):
        study.shape_geometry(64)


def test_graph_trace_rejects_partial_split_and_irregular_cublas():
    assert study.graph_profile(trace("B"), "B")["kernel_nodes_observed"] == 200
    assert study.graph_profile(trace("C"), "C")["complete_operations"] == 100
    broken = trace("B")
    broken["traceEvents"].pop()
    with pytest.raises(ValueError):
        study.graph_profile(broken, "B")
    broken = trace("C")
    broken["traceEvents"][20]["name"] = "unexpected"
    with pytest.raises(ValueError):
        study.graph_profile(broken, "C")


def test_three_decisions_keep_physical_identities_and_correctness_gate():
    raw = fixture_raw()
    result = study.summarize(raw)
    assert all(result["pairs"][key]["decision"]["decision"] == "B_FASTER" for key in study.PAIR_NAMES)
    assert result["pairs"]["BC"]["physical_labels"] == ["B", "C"]
    bad = copy.deepcopy(raw)
    bad["correctness"]["seeds"][1]["checks"]["B"]["passed"] = False
    with pytest.raises(ValueError, match="numerical"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["pairs"]["AB"]["rounds"][0]["measurements"]["B"]["complete_operations_per_block"] = 99
    with pytest.raises(ValueError, match="operation count"):
        study.summarize(bad)


def test_offline_manifest_and_decision_replay(tmp_path):
    raw = fixture_raw()
    for field, name in (("source_sha256", "source.py"), ("base_spec_sha256", "base_spec.json"),
                        ("diagnostic_script_sha256", "diagnostic_script.py"),
                        ("prior_script_sha256", "prior_script.py")):
        (tmp_path / name).write_bytes(study.SOURCE.read_bytes() if name == "source.py" else name.encode())
        raw[field] = study.prior.digest(tmp_path / name)
    raw["compiler"] = {}
    for name in ("conventional", "partial", "reduction"):
        (tmp_path / f"{name}.ptx").write_text(name)
        raw["compiler"][name] = {"ptx_sha256": study.prior.digest(tmp_path / f"{name}.ptx")}
    raw["graph_trace_sha256"] = {}
    for label in ("A", "B", "C"):
        study.prior.write_json(tmp_path / f"{label}_trace.json", trace(label))
        raw["graph_trace_sha256"][label] = study.prior.digest(tmp_path / f"{label}_trace.json")
    study.prior.write_json(tmp_path / "raw.json", raw)
    study.prior.write_json(tmp_path / "summary.json", study.summarize(raw))
    study.prior.manifest(tmp_path)
    assert study.verify(tmp_path)["decisions"]["AB"] == "B_FASTER"
    broken = json.loads((tmp_path / "C_trace.json").read_text())
    broken["traceEvents"].pop()
    study.prior.write_json(tmp_path / "C_trace.json", broken)
    with pytest.raises(ValueError, match="manifest"):
        study.verify(tmp_path)
