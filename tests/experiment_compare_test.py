"""CPU-only MI-Workflow-1 spec, decision, bundle, and CLI checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from matmul_inspector import cli, experiment_compare


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples/mi_workflow_1_fc1.json").read_text())


def test_spec_is_registered_bounded_and_finite():
    assert experiment_compare.validate_spec(EXAMPLE) == EXAMPLE
    for change, message in [
        (lambda d: d["candidate"].update(implementation="triton_source"), "packaged"),
        (lambda d: d["benchmark"].update(mode="preallocated"), "eager"),
        (lambda d: d["benchmark"].update(rounds=4), "measurement limits"),
        (lambda d: d["correctness"].update(atol=float("nan")), "finite"),
        (lambda d: d["workload"].update(m=10000), "size limit"),
        (lambda d: d.update(extra=True), "exactly"),
    ]:
        item = copy.deepcopy(EXAMPLE)
        change(item)
        with pytest.raises(ValueError, match=message):
            experiment_compare.validate_spec(item)


def test_versioned_schema_fields_match_validator_and_report():
    spec_schema = json.loads((ROOT / "schemas/mi_workflow_1_experiment_v1.schema.json").read_text())
    comparison_schema = json.loads((ROOT / "schemas/mi_workflow_1_comparison_v1.schema.json").read_text())
    assert set(spec_schema["required"]) == set(spec_schema["properties"]) == set(EXAMPLE)
    assert spec_schema["properties"]["candidate"]["properties"]["implementation"]["const"] == "triton_fc1"
    assert set(comparison_schema["properties"]) == set(
        experiment_compare.summarize({"passed": True}, rows([0.2] * 5, [0.1] * 5)))
    assert set(comparison_schema["properties"]["decision"]["enum"]) == {
        "INCONCLUSIVE", "FAILED", "REJECTED_INCORRECT", "CANDIDATE_FASTER", "BASELINE_FASTER"}


def rows(base, candidate):
    values = []
    for index, (a, b) in enumerate(zip(base, candidate), 1):
        for implementation, value in (("baseline", a), ("candidate", b)):
            values.append({"round": index, "order": ["baseline", "candidate"],
                           "implementation": implementation,
                           "cuda_event_elapsed_ms_per_operation": value})
    return values


def test_decision_requires_correctness_and_separated_rounds():
    measured = rows([0.2] * 5, [0.1] * 5)
    assert experiment_compare.summarize({"passed": True}, measured)["decision"] == "CANDIDATE_FASTER"
    failed = experiment_compare.summarize({"passed": False}, measured,
                                          {"stage": "correctness"})
    assert failed["decision"] == "REJECTED_INCORRECT"
    assert failed["speedup_baseline_over_candidate"] is None
    overlapping = experiment_compare.summarize({"passed": True}, rows(
        [0.10, 0.12, 0.11, 0.09, 0.11], [0.11, 0.10, 0.12, 0.10, 0.09]))
    assert overlapping["decision"] == "INCONCLUSIVE"
    assert overlapping["speedup_baseline_over_candidate"] is not None
    too_few = experiment_compare.summarize({"passed": True}, rows([0.2] * 4, [0.1] * 4))
    assert too_few["speedup_baseline_over_candidate"] is None
    failed_execution = experiment_compare.summarize({"passed": None}, [],
                                                     {"stage": "prerequisite"})
    assert failed_execution["decision"] == "FAILED"
    boundary = {"measurements": {"baseline": {
        "cuda_event_not_greater_than_synchronized_host_wall": False}}}
    discrepancy = experiment_compare.summarize({"passed": True}, measured,
                                                boundary_check=boundary)
    assert discrepancy["decision"] == "INCONCLUSIVE"
    assert discrepancy["speedup_baseline_over_candidate"] is None


def fake_run_factory(*, correct=True):
    def runner(config, *, output_root):
        root = Path(output_root) / "mock-run"
        root.mkdir(parents=True)
        measurements = []
        if correct:
            for index in range(1, 9):
                order = ["baseline", "candidate"] if index % 2 else ["candidate", "baseline"]
                row = {"round": index, "order": order}
                for name, value in (("baseline", 0.2), ("candidate", 0.1)):
                    for field in experiment_compare.MEASUREMENT_FIELDS:
                        row[f"{name}_{field}"] = value
                measurements.append(row)
        result = {"run_directory": str(root.resolve()), "config": config,
                  "status": "success" if correct else "failed",
                  "correctness": {"passed": correct},
                  "environment": {"device": {"name": "mock"}, "versions": {}},
                  "git": {"commit": "mock", "dirty": True}}
        if correct:
            result["benchmark"] = {"rounds": measurements, "iterations_per_measurement": 100}
        else:
            result["failure"] = {"stage": "correctness", "message": "wrong result"}
        (root / "result.json").write_text(json.dumps(result))
        return result
    return runner


def test_bundle_replays_and_detects_tampering(tmp_path):
    target = tmp_path / "bundle"
    result = experiment_compare.run_compare(EXAMPLE, target, runner=fake_run_factory(),
        compiler_capture=lambda *_: {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE",
                                     "sass": "UNAVAILABLE", "baseline_compiler": "UNAVAILABLE"})
    assert result["decision"] == "CANDIDATE_FASTER"
    assert experiment_compare.verify_bundle(target)["rounds"] == 8
    assert len((target / "timings.jsonl").read_text().splitlines()) == 16
    assert (target / "candidate_source.py").is_file()
    with pytest.raises(ValueError, match="already exists"):
        experiment_compare.run_compare(EXAMPLE, target, runner=fake_run_factory())
    (target / "timings.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest"):
        experiment_compare.verify_bundle(target)


def test_hash_consistent_false_report_is_rejected(tmp_path):
    target = tmp_path / "false-report"
    experiment_compare.run_compare(EXAMPLE, target, runner=fake_run_factory(),
        compiler_capture=lambda *_: {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE",
                                     "sass": "UNAVAILABLE", "baseline_compiler": "UNAVAILABLE"})
    comparison = json.loads((target / "comparison.json").read_text())
    comparison["decision"] = "INCONCLUSIVE"
    experiment_compare._write_json(target / "comparison.json", comparison)
    experiment_compare._manifest(target)
    with pytest.raises(ValueError, match="replay"):
        experiment_compare.verify_bundle(target)


def test_incorrect_bundle_has_no_timing_or_speedup(tmp_path):
    target = tmp_path / "wrong"
    result = experiment_compare.run_compare(EXAMPLE, target, runner=fake_run_factory(correct=False))
    assert result["decision"] == "REJECTED_INCORRECT"
    assert result["speedup_baseline_over_candidate"] is None
    assert (target / "timings.jsonl").read_text() == ""
    assert experiment_compare.verify_bundle(target)["status"] == "PASS"


def test_installed_dispatch_supports_offline_verify(tmp_path, capsys):
    target = tmp_path / "verify"
    experiment_compare.run_compare(EXAMPLE, target, runner=fake_run_factory(),
        compiler_capture=lambda *_: {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE",
                                     "sass": "UNAVAILABLE", "baseline_compiler": "UNAVAILABLE"})
    assert cli.main(["experiment", "compare", "--verify", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
