"""CPU-only validation of paired FC1 specs, decisions, artifacts, and replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from matmul_inspector import cli, experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "examples/mi_workflow_2_fc1_pair.json").read_text())


def measurements(a=None, b=None, *, discrepancy=False):
    a = a or [0.20] * 8
    b = b or [0.10] * 8
    rows = []
    for index, pair in enumerate(zip(a, b), 1):
        for label in experiment_pair.pair_order(index):
            value = pair[0 if label == "A" else 1]
            elapsed = value * 100
            wall = elapsed - 1 if discrepancy else elapsed + 1
            rows.append({"round": index, "order": experiment_pair.pair_order(index),
                         "kernel": label, "iterations": 100,
                         "cuda_event_elapsed_ms_per_operation": value,
                         "cuda_event_elapsed_ms_per_block": elapsed,
                         "host_submission_ms_per_block": elapsed - 2,
                         "host_wall_ms_per_block": wall,
                         "cuda_event_minus_host_wall_ms_per_block": elapsed - wall})
    return rows


def checks(a=True, b=True):
    return {"passed": a and b, "checks": {"A": {"passed": a}, "B": {"passed": b}}}


def controls(discrepancy=False):
    return {label: {"cuda_event_minus_host_wall_ms_per_block": 1 if discrepancy else -1}
            for label in experiment_pair.LABELS}


def test_spec_exact_bounded_registered_distinct():
    assert experiment_pair.validate_spec(SPEC) == SPEC
    for edit in [
        lambda s: s["kernels"].pop("B"),
        lambda s: s["kernels"]["B"].update(implementation="triton_source"),
        lambda s: s["kernels"]["B"]["parameters"].update(block_k=7),
        lambda s: s["kernels"].update(B=copy.deepcopy(s["kernels"]["A"])),
        lambda s: s["benchmark"].update(rounds=4),
        lambda s: s["benchmark"].pop("mode"),
        lambda s: s["correctness"].update(atol=float("nan")),
        lambda s: s.update(extra=1),
    ]:
        item = copy.deepcopy(SPEC)
        edit(item)
        with pytest.raises(ValueError):
            experiment_pair.validate_spec(item)
    schema = json.loads((ROOT / "schemas/mi_workflow_2_pair_v1.schema.json").read_text())
    assert set(schema["required"]) == set(SPEC)


def test_order_is_balanced_and_validated():
    assert [experiment_pair.pair_order(i) for i in range(1, 9)] == [
        ["A", "B"], ["B", "A"]] * 4
    with pytest.raises(ValueError):
        experiment_pair.pair_order(0)


def test_correctness_gates_ranking():
    result = experiment_pair.summarize(checks(False, True), measurements(),
                                       {"stage": "correctness", "message": "A failed"}, controls())
    assert result["decision"] == "REJECTED_INCORRECT"
    assert result["speedup_A_over_B"] is None
    assert experiment_pair.summarize(checks(False, True), measurements(), controls=controls())[
        "decision"] == "REJECTED_INCORRECT"


def test_pair_aggregation_and_uncertainty():
    result = experiment_pair.summarize(checks(), measurements(), controls=controls())
    assert result["decision"] == "B_FASTER"
    assert result["rounds"] == 8
    assert result["paired_A_minus_B_ms"] == pytest.approx([0.1] * 8)
    assert result["paired_signs"]["B_faster"] == 8
    assert result["speedup_A_over_B"] == pytest.approx(0.5)
    assert result["paired_difference_ms"]["median_absolute_deviation_ms"] == 0


def test_overlap_or_timer_conflict_is_inconclusive():
    overlap = experiment_pair.summarize(checks(), measurements(
        [0.10, 0.13] * 4, [0.12, 0.11] * 4), controls=controls())
    assert overlap["decision"] == "INCONCLUSIVE"
    assert overlap["paired_signs"] == {"A_faster": 4, "B_faster": 4, "ties": 0}
    discrepancy = experiment_pair.summarize(checks(), measurements(discrepancy=True),
                                            controls=controls())
    assert discrepancy["decision"] == "INCONCLUSIVE"
    assert discrepancy["timing_boundary_discrepancy"] is True
    assert discrepancy["speedup_A_over_B"] is None
    control_discrepancy = experiment_pair.summarize(checks(), measurements(),
                                                    controls=controls(True))
    assert control_discrepancy["decision"] == "INCONCLUSIVE"
    no_controls = experiment_pair.summarize(checks(), measurements())
    assert no_controls["decision"] == "INCONCLUSIVE"
    # Device intervals favor A, while host-wall pairs favor B. Each host
    # interval still encloses its corresponding CUDA-event interval.
    conflict = measurements([0.10] * 8, [0.20] * 8)
    for row in conflict:
        if row["kernel"] == "A":
            row["host_wall_ms_per_block"] = 30.0
            row["cuda_event_minus_host_wall_ms_per_block"] = -20.0
    contradictory = experiment_pair.summarize(checks(), conflict, controls=controls())
    assert contradictory["decision"] == "INCONCLUSIVE"
    assert "host-wall" in contradictory["reason"]


def fake_runner(spec):
    rows = measurements()
    rounds = []
    for index in range(1, 9):
        ordered = [row for row in rows if row["round"] == index]
        rounds.append({"round": index, "order": experiment_pair.pair_order(index),
                       "measurements": {row["kernel"]: {field: row[field]
                                        for field in experiment_pair.FIELDS} for row in ordered}})
    return {"spec": spec, "status": "success", "correctness": checks(),
            "environment": {"device": {"name": "mock"}, "versions": {}},
            "git": {"commit": "mock"}, "benchmark": {
                "mode": "allocating", "warmups_per_kernel": 30,
                "iterations_per_measurement": 100, "unranked_timing_controls": controls(),
                "rounds": rounds}}


def mock_compiler(spec, output):
    return {"A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
            "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
            "ptx_comparison": {"status": "UNAVAILABLE"}}


def test_bundle_report_manifest_and_offline_replay(tmp_path, capsys):
    output = tmp_path / "pair"
    result = experiment_pair.run_pair(SPEC, output, runner=fake_runner,
                                      compiler_capture=mock_compiler,
                                      command=["example", "experiment", "pair"])
    assert result["decision"] == "B_FASTER"
    assert "Paired A−B" in (output / "report.md").read_text()
    assert "| Round | CUDA-event" in (output / "report.md").read_text()
    assert "GPU: mock" in (output / "report.md").read_text()
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    assert cli.main(["experiment", "pair", "--verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "B_FASTER"
    with pytest.raises(ValueError, match="already exists"):
        experiment_pair.run_pair(SPEC, output, runner=fake_runner)
    (output / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        experiment_pair.verify_bundle(output)


def test_hash_consistent_false_decision_is_rejected(tmp_path):
    output = tmp_path / "pair"
    experiment_pair.run_pair(SPEC, output, runner=fake_runner, compiler_capture=mock_compiler)
    changed = json.loads((output / "comparison.json").read_text())
    changed["decision"] = "INCONCLUSIVE"
    experiment_pair._write_json(output / "comparison.json", changed)
    experiment_pair._manifest(output)
    with pytest.raises(ValueError, match="reproduce"):
        experiment_pair.verify_bundle(output)


def test_incorrect_result_keeps_failure_and_no_timing(tmp_path):
    def wrong(spec):
        return {"spec": spec, "status": "failed", "correctness": checks(False, True),
                "failure": {"stage": "correctness", "message": "A failed"}}
    output = tmp_path / "wrong"
    result = experiment_pair.run_pair(SPEC, output, runner=wrong)
    assert result["decision"] == "REJECTED_INCORRECT"
    assert (output / "timings.jsonl").read_text() == ""
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"


def test_ptx_difference_uses_existing_comparator(tmp_path, monkeypatch):
    before = ROOT / "artifacts/mi-dogfood-1/baseline-v1/candidate.ptx"
    after = ROOT / "artifacts/mi-dogfood-1/stages2-v1/candidate.ptx"
    assert before.is_file() and after.is_file()
    output = tmp_path / "compiler"
    output.mkdir()
    from matmul_inspector import experiment_compare

    def capture(spec, destination):
        path = before if spec["candidate"]["parameters"]["num_stages"] == 3 else after
        (destination / "candidate.ptx").write_bytes(path.read_bytes())
        return {"status": "CAPTURED", "ptx": "CAPTURED", "sass": "UNAVAILABLE",
                "metadata": {"shared": 24576 if path == before else 12288}}

    monkeypatch.setattr(experiment_compare, "_compiler_capture", capture)
    result = experiment_pair.capture_compiler(SPEC, output)
    assert result["ptx_comparison"]["status"] == "CAPTURED"
    assert result["ptx_comparison"]["normalized_equal"] is False
    assert result["ptx_comparison"]["normalized_changed_lines"] > 0
    assert result["ptx_comparison"]["A_ptx_sha256"] == experiment_pair._digest(output / "A.ptx")
    assert result["ptx_comparison"]["B_ptx_sha256"] == experiment_pair._digest(output / "B.ptx")
    assert (output / "ptx_comparison/ptx/normalized.diff").is_file()


def test_bundle_replays_ptx_comparison_not_just_hashes(tmp_path, monkeypatch):
    before = ROOT / "artifacts/mi-dogfood-1/baseline-v1/candidate.ptx"
    after = ROOT / "artifacts/mi-dogfood-1/stages2-v1/candidate.ptx"
    from matmul_inspector import experiment_compare

    def capture(spec, destination):
        path = before if spec["candidate"]["parameters"]["num_stages"] == 3 else after
        (destination / "candidate.ptx").write_bytes(path.read_bytes())
        return {"status": "CAPTURED", "ptx": "CAPTURED", "sass": "UNAVAILABLE"}

    monkeypatch.setattr(experiment_compare, "_compiler_capture", capture)
    output = tmp_path / "pair-ptx"
    experiment_pair.run_pair(SPEC, output, runner=fake_runner)
    assert experiment_pair.verify_bundle(output)["status"] == "PASS"
    (output / "ptx_comparison/ptx/normalized.diff").write_text("false diff\n")
    experiment_pair._manifest(output)
    with pytest.raises(ValueError, match="PTX comparison"):
        experiment_pair.verify_bundle(output)
