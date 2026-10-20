"""CPU-only tests for the bounded A/A placement diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from matmul_inspector import experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "examples/mi_timing_2_fc1_pair.json").read_text())


@pytest.mark.parametrize("placement,first,second", [
    ("before", ["A", "A", "A", "B"], ["A", "A", "B", "A"]),
    ("after", ["A", "B", "A", "A"], ["B", "A", "A", "A"]),
])
def test_phase_order_and_operation_counts(placement, first, second):
    calls = []

    def measure(_torch, function, iterations, **_options):
        for _ in range(iterations):
            calls.append(function())
        return {"calls": iterations}

    aa, ab, sequence = experiment_pair.measure_paired_rounds(
        measure, None, {"A": lambda: "A", "B": lambda: "B"}, 3, 2, {},
        with_aa=True, aa_placement=placement)
    assert [calls[index] for index in range(0, len(calls), 3)] == first + second
    assert len(calls) == 24
    assert [row["order"] for row in ab] == [["A", "B"], ["B", "A"]]
    assert [row["order"] for row in aa] == [["X", "Y"], ["Y", "X"]]
    assert sequence == experiment_pair.expected_diagnostic_sequence(2, 3, placement)
    assert {row["iterations_per_block"] for row in sequence} == {3}


def test_default_legacy_rounds_keep_no_aa():
    calls = []
    aa, ab, sequence = experiment_pair.measure_paired_rounds(
        lambda _torch, function, iterations, **_kw: calls.append(function()),
        None, {"A": lambda: "A", "B": lambda: "B"}, 100, 2, {}, with_aa=False)
    assert aa == []
    assert calls == ["A", "B", "B", "A"]
    assert [row["phase"] for row in sequence] == ["ab", "ab"]


def test_diagnostic_option_rejects_invalid_or_legacy_spec(tmp_path):
    legacy = json.loads((ROOT / "examples/mi_workflow_2_fc1_pair.json").read_text())
    for placement, spec in (("invalid", SPEC), ("after", legacy)):
        with pytest.raises(ValueError, match="diagnostic A/A placement"):
            experiment_pair.run_pair(spec, tmp_path / placement,
                                     diagnostic_aa_placement=placement)
        assert not (tmp_path / placement).exists()


def test_diagnostic_bundle_replays_phase_order(tmp_path):
    source = json.loads((ROOT / "artifacts/mi-timing-2/fc1-pair-v1/raw_result.json").read_text())

    def runner(spec, *, runtime_spec):
        placement = runtime_spec["_diagnostic_aa_placement"]
        raw = copy.deepcopy(source)
        raw["spec"] = spec
        raw["benchmark"]["diagnostic_aa_placement"] = placement
        raw["benchmark"]["measurement_sequence"] = experiment_pair.expected_diagnostic_sequence(
            spec["benchmark"]["rounds"], spec["benchmark"]["iterations"], placement)
        return raw

    def compiler(spec, output):
        return {"A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
                "ptx_comparison": {"status": "UNAVAILABLE"}}

    for placement in ("before", "after"):
        output = tmp_path / placement
        result = experiment_pair.run_pair(
            SPEC, output, runner=runner, compiler_capture=compiler,
            diagnostic_aa_placement=placement)
        assert result["decision_schema_version"] == 3
        assert f"Diagnostic A/A placement: {placement}" in (output / "report.md").read_text()
        assert experiment_pair.verify_bundle(output)["status"] == "PASS"
        raw = json.loads((output / "raw_result.json").read_text())
        raw["benchmark"]["measurement_sequence"].reverse()
        experiment_pair._write_json(output / "raw_result.json", raw)
        experiment_pair._manifest(output)
        with pytest.raises(ValueError, match="phase order"):
            experiment_pair.verify_bundle(output)
