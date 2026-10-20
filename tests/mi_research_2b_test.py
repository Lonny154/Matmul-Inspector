"""CPU-only checks for the narrow Split-2 phase diagnostic."""

import copy
import json

import pytest

from scripts import run_mi_research_2b_diagnostic as diagnostic


def sample(value):
    return {"cuda_event_elapsed_ms_per_block": value,
            "host_wall_ms_per_block": value + 0.1,
            "monotonic_raw_ms_per_block": value + 0.09,
            "host_submission_ms_per_block": value / 2}


def fixture_raw(session):
    return {"schema_version": 1, "session": session,
            "iterations_per_block": diagnostic.ITERATIONS,
            "correctness": {"passed": True}, "graph_phase_probe": {},
            "rounds": [{"round": index, "group_order": diagnostic.group_order(session),
                        "complete_order": diagnostic.complete_order(index, session),
                        "phase_order": diagnostic.phase_order(index),
                        "measurements": {key: sample(index + offset)
                                         for offset, key in enumerate((
                                             "conventional", "alloc", "reuse", "partial", "reduction"), 1)}}
                       for index in range(1, diagnostic.ROUNDS + 1)],
            "instrumented_probe": [
                {"block": index, "per_call": [
                    {"scratch_allocation_ns": 1, "partial_launch_submission_ns": 2,
                     "reduction_launch_submission_ns": 3}
                    for _ in range(diagnostic.ITERATIONS)]}
                for index in range(1, diagnostic.PROBE_BLOCKS + 1)]}


def test_balanced_orders_and_reversed_outer_groups():
    assert diagnostic.group_order(1) == list(reversed(diagnostic.group_order(2)))
    for session in (1, 2):
        complete = [diagnostic.complete_order(i, session) for i in range(1, 9)]
        assert complete.count(["alloc", "reuse"]) == 4
        assert complete.count(["reuse", "alloc"]) == 4
        assert diagnostic.summarize(fixture_raw(session))["rounds"] == 8
    assert diagnostic.complete_order(1, 1) != diagnostic.complete_order(1, 2)


def test_summary_rejects_missing_or_changed_round_order():
    raw = fixture_raw(1)
    raw["rounds"][0]["complete_order"].reverse()
    with pytest.raises(ValueError, match="order"):
        diagnostic.summarize(raw)
    raw = fixture_raw(1)
    raw["rounds"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        diagnostic.summarize(raw)


def test_offline_manifest_and_summary_replay(tmp_path):
    raw = fixture_raw(2)
    (tmp_path / "source.py").write_text("# source snapshot\n")
    (tmp_path / "experiment.json").write_text("{}\n")
    (tmp_path / "diagnostic_script.py").write_text("# diagnostic snapshot\n")
    (tmp_path / "partial.ptx").write_text("partial\n")
    (tmp_path / "reduction.ptx").write_text("reduction\n")
    raw["source_sha256"] = diagnostic.digest(tmp_path / "source.py")
    raw["spec_sha256"] = diagnostic.digest(tmp_path / "experiment.json")
    raw["diagnostic_script_sha256"] = diagnostic.digest(tmp_path / "diagnostic_script.py")
    raw["compiler"] = {name: {"ptx_sha256": diagnostic.digest(tmp_path / f"{name}.ptx")}
                       for name in ("partial", "reduction")}
    diagnostic.write_json(tmp_path / "raw.json", raw)
    diagnostic.write_json(tmp_path / "summary.json", diagnostic.summarize(raw))
    diagnostic.manifest(tmp_path)
    assert diagnostic.verify(tmp_path)["status"] == "PASS"

    changed = copy.deepcopy(raw)
    changed["rounds"][0]["measurements"]["alloc"]["cuda_event_elapsed_ms_per_block"] += 1
    diagnostic.write_json(tmp_path / "raw.json", changed)
    with pytest.raises(ValueError, match="manifest"):
        diagnostic.verify(tmp_path)
    diagnostic.manifest(tmp_path)
    with pytest.raises(ValueError, match="summary"):
        diagnostic.verify(tmp_path)
