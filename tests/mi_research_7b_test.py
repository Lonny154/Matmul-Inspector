"""CPU-only checks for the bounded MI-Research-7B study."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import analyze_mi_research_7b as analysis
from scripts import generate_mi_research_7b_kernel as generator
from scripts import run_mi_research_7b as study


def test_generated_mapping_and_frozen_protocol():
    assert generator.TARGET.read_text() == generator.render()
    assert study.check_protocol()["source_sha256"]["B"] == generator.digest(generator.TARGET)


def test_retained_sessions_and_cross_session_decision():
    assert analysis.verify()["status"] == "PASS"
    for name in analysis.SESSIONS:
        assert study.verify(analysis.BASE / name)["status"] == "PASS"
    result = analysis.analyze()
    assert [result["sessions"][f"primary-s{i}"]["decision"] for i in (1, 2)] == ["B_FASTER"] * 2
    assert result["sessions"]["cublas-s1"]["decision"] == "B_FASTER"


def test_timing_corruption_and_source_corruption_rejected(tmp_path):
    bundle = analysis.BASE / "primary-s1"
    raw = json.loads((bundle / "raw.json").read_text())
    changed = copy.deepcopy(raw)
    changed["pair"]["rounds"][0]["order"] = ["B", "A"]
    with pytest.raises(ValueError, match="pair order"):
        study.summarize(changed)
    copied = tmp_path / "changed"
    shutil.copytree(bundle, copied)
    source = copied / "B_source.py"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(copied)
