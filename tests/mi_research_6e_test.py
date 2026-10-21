"""CPU-only contracts and retained evidence for the staged graph controls."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import analyze_mi_research_6e as analysis
from scripts import run_mi_research_6e_controls as study


def test_frozen_protocol_and_independent_null_gates():
    protocol = json.loads(study.PROTOCOL.read_text())
    study.check_protocol(protocol)
    changed = copy.deepcopy(protocol)
    changed["rounds_per_session"] = 7
    with pytest.raises(ValueError, match="protocol"):
        study.check_protocol(changed)
    for condition in ("staged_null", "fused_null"):
        for session in (1, 2):
            bundle = analysis.EVIDENCE / f"{condition}-s{session}"
            assert study.verify(bundle)["status"] == "PASS"
            raw = json.loads((bundle / "raw.json").read_text())
            assert raw["correctness"]["distinct_input_and_output_addresses"]
            assert all(row["cross_instance_bitwise_equal"] for row in raw["correctness"]["cases"])
            assert study.summarize(raw)["positive_rounds"] in range(9)


def test_cross_rule_replays_without_relabeling_history():
    for session in (1, 2):
        bundle = analysis.EVIDENCE / f"cross-s{session}"
        assert study.verify(bundle)["existing_rule"] == "B_FASTER"
        raw = json.loads((bundle / "raw.json").read_text())
        assert study.summarize(raw)["existing_rule"]["decision"] == "B_FASTER"
        assert len(raw["benchmark"]["same_instance_aa_rounds"]) == study.ROUNDS
    recovered = json.loads((analysis.EVIDENCE / "cross-s1/recovery.json").read_text())
    assert recovered["measurement_repeated"] is False
    assert (analysis.EVIDENCE / "cross-s1/failure.json").exists()
    assert analysis.run(verify=True) == {"status": "PASS", "classification": "C",
                                         "cross_decisions": ["B_FASTER", "B_FASTER"]}
    result = analysis.analyze()
    assert [row["decision"] for row in result["historical_6d_unchanged"]] == [
        "INCONCLUSIVE", "INCONCLUSIVE"]
    assert all(row["positive_cross_pairs"] == 8 for row in result["sensitivity"])


def test_corrupt_topology_timing_and_manifest_are_rejected(tmp_path):
    raw = json.loads((analysis.EVIDENCE / "staged_null-s1/raw.json").read_text())
    bad = copy.deepcopy(raw)
    bad["graph_profile"]["Y"]["kernels_per_operation"] -= 1
    with pytest.raises(ValueError, match="topology"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["benchmark"]["rounds"][0]["measurements"]["X"]["complete_operations_per_block"] -= 1
    with pytest.raises(ValueError, match="normalization"):
        study.summarize(bad)
    copied = tmp_path / "changed"
    shutil.copytree(analysis.EVIDENCE / "staged_null-s1", copied)
    trace = copied / "X_graph_trace.json"
    trace.write_bytes(trace.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(copied)
