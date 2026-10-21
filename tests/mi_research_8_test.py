"""CPU-only exact and retained-evidence checks for MI-Research-8."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import analyze_mi_research_8 as analysis
from scripts import generate_mi_research_8_kernel as generator
from scripts import run_mi_research_8 as study


def test_matched_generator_and_exact_factor_map():
    generated, stats = generator.generate()
    assert generated == generator.TARGET.read_text()
    assert generator.sha256(generator.BASELINE) == generator.BASELINE_SHA256
    assert stats["dots_by_block"] == [8, 6, 6, 11, 3, 10, 5, 3, 6]
    assert study.check_protocol()["source_sha256"]["B_58"] == generator.sha256(generator.TARGET)


def test_retained_sessions_and_analysis():
    assert analysis.verify()["status"] == "PASS"
    result = analysis.analyze()
    assert result["identical_rank_23_factor_maps"]
    assert result["tensor"] == {"checked": 729, "mismatches": 0, "unit": 27, "zero": 702}
    assert result["decisions"] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert result["sessions"]["graph-s1"]["compiler"]["B"]["metadata"]["n_spills"] == 100


def test_corrupt_pair_and_source_are_rejected(tmp_path):
    bundle = analysis.BASE / "graph-s1"
    raw = json.loads((bundle / "raw.json").read_text())
    changed = copy.deepcopy(raw)
    changed["pair"]["rounds"][0]["order"] = ["B", "A"]
    with pytest.raises(ValueError, match="paired graph order changed"):
        study.summarize(changed)
    copied = tmp_path / "changed"
    shutil.copytree(bundle, copied)
    source = copied / "B_source.py"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(copied)
