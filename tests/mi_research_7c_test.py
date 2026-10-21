"""CPU-only verification of the bounded MI-Research-7C counter study."""

from __future__ import annotations

import json
import shutil

import pytest

from scripts import analyze_mi_research_7c as study


def test_retained_profile_and_historical_evidence():
    assert study.verify()["status"] == "PASS"
    result = study.analyze()
    assert result["measured_active_sm_coverage"] == {"A": 4, "B": 34}
    assert result["classification"] == "B — Multiple contributors identified"
    a = result["sessions"]["A"]["targeted_capture"]["metrics"]
    b = result["sessions"]["B"]["targeted_capture"]["metrics"]
    assert b["smsp__sass_inst_executed_op_local_ld.sum"] < a[
        "smsp__sass_inst_executed_op_local_ld.sum"]
    assert b["dram__bytes_read.sum"] > a["dram__bytes_read.sum"]


def test_wrong_kernel_and_counter_corruption_rejected(tmp_path, monkeypatch):
    copied = tmp_path / "mi-research-7c"
    shutil.copytree(study.BASE, copied)
    monkeypatch.setattr(study, "BASE", copied)
    text = (copied / "A_ncu_targeted_console.txt").read_text()
    (copied / "A_ncu_targeted_console.txt").write_text(text.replace('"_fused"', '"wrong"', 1))
    with pytest.raises(ValueError, match="wrong profiled kernel"):
        study.analyze()
    (copied / "A_ncu_targeted_console.txt").write_text(text)
    record = json.loads((copied / "A_preflight.json").read_text())
    record["correctness"][0]["graph"]["passed"] = False
    (copied / "A_preflight.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="numerical gate changed"):
        study.analyze()
