"""CPU-only fixtures for version 3 clock-aware paired evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from matmul_inspector import experiment, experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "examples/mi_timing_2_fc1_pair.json").read_text())


def sample(event_ms: float, *, perf_ms: float | None = None, raw_ms: float | None = None):
    perf_ms = event_ms - 0.5 if perf_ms is None else perf_ms
    raw_ms = event_ms + 0.5 if raw_ms is None else raw_ms
    return {"cuda_event_elapsed_ms_per_operation": event_ms / 100,
            "cuda_event_elapsed_ms_per_block": event_ms,
            "host_submission_ms_per_block": perf_ms - 0.1,
            "host_wall_ms_per_block": perf_ms,
            "cuda_event_minus_host_wall_ms_per_block": event_ms - perf_ms,
            "monotonic_raw_ms_per_block": raw_ms,
            "cuda_event_minus_monotonic_raw_ms_per_block": event_ms - raw_ms}


def fixture_result(spec=SPEC, *, a_ms=10.0, b_ms=20.0, aa_gap=0.01):
    rounds = []
    aa_rounds = []
    for index in range(1, 9):
        order = experiment_pair.pair_order(index)
        rounds.append({"round": index, "order": order,
                       "measurements": {label: sample(a_ms if label == "A" else b_ms)
                                        for label in order}})
        aa_order = ["X", "Y"] if index % 2 else ["Y", "X"]
        aa_rounds.append({"round": index, "order": aa_order,
                          "measurements": {slot: sample(15.0 + (aa_gap if slot == "Y" else 0))
                                           for slot in aa_order}})
    return {"spec": spec, "status": "success",
            "correctness": {"passed": True, "checks": {"A": {"passed": True}, "B": {"passed": True}}},
            "environment": {"device": {"name": "mock"}, "versions": {}}, "git": {"commit": "mock"},
            "benchmark": {"mode": "preallocated", "warmups_per_kernel": 30,
                          "iterations_per_measurement": 100,
                          "unranked_timing_controls": {"A": sample(a_ms), "B": sample(b_ms)},
                          "host_clock_preflight": experiment_pair._preflight_summary([
                              {"perf_counter_ms": 20.0, "monotonic_raw_ms": 21.0,
                               "raw_minus_perf_ms": 1.0, "raw_over_perf": 1.05,
                               "read_span_bound_ms": 0.002} for _ in range(5)]),
                          "aa_kernel": "A", "aa_iterations_per_measurement": 100,
                          "aa_rounds": aa_rounds, "rounds": rounds}}


def test_v3_spec_reuses_bounded_preallocated_contract():
    assert experiment_pair.validate_spec(SPEC) == SPEC
    schema = json.loads((ROOT / "schemas/mi_timing_2_pair_v3.schema.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == 3
    for change in (lambda s: s["benchmark"].update(mode="allocating"),
                   lambda s: s["benchmark"].update(rounds=21),
                   lambda s: s["kernels"].update(B=copy.deepcopy(s["kernels"]["A"]))):
        bad = copy.deepcopy(SPEC)
        change(bad)
        with pytest.raises(ValueError):
            experiment_pair.validate_spec(bad)


def test_clock_preflight_detects_resolved_rate_difference(monkeypatch):
    ticks = []
    raw = []
    for index in range(5):
        base = index * 30_000_000
        ticks.extend([base, base + 1000, base + 20_000_000, base + 20_001_000])
        raw.extend([base, base + 21_000_000])
    monkeypatch.setattr(experiment_pair.time, "perf_counter_ns", lambda: ticks.pop(0))
    result = experiment_pair.host_clock_preflight(reader=lambda: raw.pop(0), sleep=lambda _: None)
    assert result["status"] == "RATE_DIFFERENCE_OBSERVED"
    assert result["median_raw_over_perf"] > 1.04
    assert len(result["samples"]) == 5


def test_matched_measurement_window_has_exact_operation_count(monkeypatch):
    order = []
    perf = iter([0, 21_000_000, 22_000_000])
    raw = iter([1000, 21_999_000])
    monkeypatch.setattr(experiment.time, "perf_counter_ns", lambda: (order.append("perf"), next(perf))[1])

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            order.append("event")

        def synchronize(self):
            order.append("sync")

        def elapsed_time(self, other):
            return 20.0

    class Cuda:
        pass

    Cuda.Event = Event

    class Torch:
        cuda = Cuda()

    result = experiment.CudaTritonBackend._measure(
        Torch(), lambda: order.append("call"), 3,
        raw_clock=lambda: (order.append("raw"), next(raw))[1])
    assert order == ["perf", "raw", "event", "call", "call", "call", "event",
                     "perf", "sync", "raw", "perf"]
    assert result["cuda_event_elapsed_ms_per_block"] == 20
    assert result["host_wall_ms_per_block"] == 22
    assert result["monotonic_raw_ms_per_block"] == pytest.approx(21.998)


def test_clock_aware_decision_and_aa_variability():
    raw = fixture_result()
    result = experiment_pair.summarize_clock_aware(raw["correctness"],
                                                    experiment_pair.timing_rows(raw), None, raw["benchmark"])
    assert result["decision"] == "A_FASTER"
    assert result["timing_boundary_discrepancy"] is True
    assert result["raw_boundary_discrepancy"] is False
    assert "HOST_CLOCK_RATE_DIFFERENCE" in result["warnings"]
    assert result["aa_control"]["rounds"] == 8
    noisy = fixture_result(a_ms=10, b_ms=10.02, aa_gap=0.05)
    result = experiment_pair.summarize_clock_aware(noisy["correctness"],
                                                    experiment_pair.timing_rows(noisy), None,
                                                    noisy["benchmark"])
    assert result["decision"] == "INCONCLUSIVE"
    assert "AB_WITHIN_AA_VARIABILITY" in result["warnings"]


def test_missing_raw_or_bad_aa_and_correctness_gate():
    raw = fixture_result()
    raw["benchmark"]["rounds"][0]["measurements"]["A"].pop("monotonic_raw_ms_per_block")
    result = experiment_pair.summarize_clock_aware(raw["correctness"],
                                                    experiment_pair.timing_rows(raw), None, raw["benchmark"])
    assert result["decision"] == "INCONCLUSIVE"
    assert "INVALID_DIAGNOSTICS" in result["warnings"]
    raw = fixture_result()
    raw["benchmark"]["aa_rounds"].pop()
    result = experiment_pair.summarize_clock_aware(raw["correctness"],
                                                    experiment_pair.timing_rows(raw), None, raw["benchmark"])
    assert result["decision"] == "INCONCLUSIVE"
    raw = fixture_result()
    raw["correctness"]["checks"]["A"]["passed"] = False
    result = experiment_pair.summarize_clock_aware(raw["correctness"], [],
                                                    {"stage": "correctness"}, None)
    assert result["decision"] == "REJECTED_INCORRECT"


def test_unavailable_raw_and_unexplained_boundary_are_inconclusive():
    unavailable = fixture_result()
    unavailable["benchmark"]["host_clock_preflight"] = {
        "status": "UNAVAILABLE", "reason": "CLOCK_MONOTONIC_RAW is unavailable", "samples": []}
    for row in unavailable["benchmark"]["unranked_timing_controls"].values():
        row.pop("monotonic_raw_ms_per_block")
        row.pop("cuda_event_minus_monotonic_raw_ms_per_block")
    for group in ("rounds", "aa_rounds"):
        for round_row in unavailable["benchmark"][group]:
            for row in round_row["measurements"].values():
                row.pop("monotonic_raw_ms_per_block")
                row.pop("cuda_event_minus_monotonic_raw_ms_per_block")
    result = experiment_pair.summarize_clock_aware(
        unavailable["correctness"], experiment_pair.timing_rows(unavailable),
        None, unavailable["benchmark"])
    assert result["decision"] == "INCONCLUSIVE"
    assert "MONOTONIC_RAW_UNAVAILABLE" in result["warnings"]

    unexplained = fixture_result()
    samples = [{"perf_counter_ms": 20.0, "monotonic_raw_ms": 20.001,
                "raw_minus_perf_ms": 0.001, "raw_over_perf": 20.001 / 20.0,
                "read_span_bound_ms": 0.002} for _ in range(5)]
    unexplained["benchmark"]["host_clock_preflight"] = experiment_pair._preflight_summary(samples)
    result = experiment_pair.summarize_clock_aware(
        unexplained["correctness"], experiment_pair.timing_rows(unexplained),
        None, unexplained["benchmark"])
    assert result["decision"] == "INCONCLUSIVE"
    assert "without a resolved clock-rate explanation" in result["reason"]


def test_v3_report_manifest_and_offline_replay(tmp_path):
    output = tmp_path / "pair"

    def compiler(spec, output):
        return {"A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "ptx_comparison": {"status": "UNAVAILABLE"}}

    result = experiment_pair.run_pair(SPEC, output, runner=fixture_result,
                                      compiler_capture=compiler, command=["example", "--output", str(output)])
    assert result["decision"] == "A_FASTER"
    assert "Clock-aware diagnostics" in (output / "report.md").read_text()
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    saved = json.loads((output / "raw_result.json").read_text())
    saved["benchmark"]["aa_rounds"][0]["measurements"]["X"]["cuda_event_elapsed_ms_per_block"] = 999
    experiment_pair._write_json(output / "raw_result.json", saved)
    experiment_pair._manifest(output)
    with pytest.raises(ValueError, match="reproduce"):
        experiment_pair.verify_bundle(output)


def test_successful_v3_bundle_rejects_malformed_controls_even_if_rehashed(tmp_path):
    output = tmp_path / "pair"

    def compiler(spec, output):
        return {"A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "ptx_comparison": {"status": "UNAVAILABLE"}}

    experiment_pair.run_pair(SPEC, output, runner=fixture_result, compiler_capture=compiler)
    raw = json.loads((output / "raw_result.json").read_text())
    raw["benchmark"]["aa_rounds"].pop()
    experiment_pair._write_json(output / "raw_result.json", raw)
    decision = experiment_pair.summarize_clock_aware(
        raw["correctness"], experiment_pair.timing_rows(raw), None, raw["benchmark"])
    experiment_pair._write_json(output / "comparison.json", decision)
    report = experiment_pair.render_report(
        json.loads((output / "experiment.json").read_text()), raw["correctness"], decision,
        json.loads((output / "compiler.json").read_text()),
        json.loads((output / "provenance.json").read_text()))
    (output / "report.md").write_text(report)
    experiment_pair._manifest(output)
    with pytest.raises(ValueError, match="invalid timing diagnostics"):
        experiment_pair.verify_bundle(output)
