"""CPU-only protocol and retained-evidence checks for MI-Research-7."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import analyze_mi_research_7 as analysis
from scripts import run_mi_research_7 as study


def test_frozen_protocol_and_four_session_decisions():
    assert study.check_protocol()["rounds_per_pair"] == 8
    expected = {"eager": ["INCONCLUSIVE", "INCONCLUSIVE"],
                "cuda_graph": ["B_FASTER", "B_FASTER"]}
    for mode, decisions in expected.items():
        for session in (1, 2):
            bundle = analysis.BASE / f"{mode}-s{session}"
            assert study.verify(bundle)["status"] == "PASS"
            raw = json.loads((bundle / "raw.json").read_text())
            assert raw["precision_policy"]["restored"]
            assert raw["correctness"]["fixed_addresses_unchanged"]
            assert all(row["physical_kernel_events"] == (1 if mode == "eager" else 20)
                       for row in raw["topology"].values())
            summary = study.summarize(raw)
            assert [summary["pairs"][other]["decision"]["decision"]
                    for other in ("T", "C")] == [decisions[session - 1]] * 2


def test_cross_session_analysis_replays():
    result = analysis.verify()
    assert result["status"] == "PASS"
    assert result["classification"] == "B — Valid comparison without a supported advantage"
    assert result["decisions"] == {
        "eager": {"T": ["INCONCLUSIVE"] * 2, "C": ["INCONCLUSIVE"] * 2},
        "cuda_graph": {"T": ["B_FASTER"] * 2, "C": ["B_FASTER"] * 2}}


def test_changed_operation_count_and_source_are_rejected(tmp_path):
    bundle = analysis.BASE / "cuda_graph-s1"
    raw = json.loads((bundle / "raw.json").read_text())
    bad = copy.deepcopy(raw)
    bad["pairs"]["T"]["rounds"][0]["measurements"]["A"][
        "complete_operations_per_block"] = 19
    with pytest.raises(ValueError, match="operation count"):
        study.summarize(bad)
    copied = tmp_path / "changed"
    shutil.copytree(bundle, copied)
    source = copied / "perminov_source.py"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(copied)
