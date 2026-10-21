"""CPU-only checks for the bounded staged CUDA Graph attribution record."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import analyze_mi_research_6f as analysis
from scripts import run_mi_research_6f_trace as study


def test_frozen_bounds_and_correctness_records():
    protocol = json.loads(study.PROTOCOL.read_text())
    study.check_protocol(protocol)
    changed = copy.deepcopy(protocol)
    changed["sessions"][1]["measured_blocks"] = 3
    with pytest.raises(ValueError, match="protocol"):
        study.check_protocol(changed)
    for session in (1, 2):
        bundle = analysis.BASE / f"session{session}"
        assert study.verify(bundle)["status"] == "PASS"
        raw = json.loads((bundle / "raw.json").read_text())
        assert raw["correctness"]["fixed_addresses_unchanged"]
        assert len(raw["measurements"]) == (12 if session == 1 else 2)


def test_node_replay_correlation_and_intervals():
    assert analysis.verify() == {"status": "PASS", "decision": "C — Still unexplained",
                                 "blocks": 2, "kernel_rows": 960}
    result, timeline = analysis.analyze()
    blocks = result["session2"]["blocks"]
    assert [block["kernel_count"] for block in blocks] == [480, 480]
    assert blocks[0]["graph_launch_correlation_id"] != blocks[1]["graph_launch_correlation_id"]
    assert abs(blocks[0]["kernel_span_us"] - blocks[1]["kernel_span_us"]) < 1
    assert blocks[0]["graph_launch_api_us"] > blocks[1]["graph_launch_api_us"]
    for replay_id in (block["replay_id"] for block in blocks):
        nodes = [row for row in timeline if row["replay_id"] == replay_id]
        assert len(nodes) == 480
        assert [row["name"] for row in nodes[:24]] == ["_block_product"] * 23 + [
            "_reconstruct"]


def test_bundle_manifest_rejects_tampering(tmp_path):
    copied = tmp_path / "session"
    shutil.copytree(analysis.BASE / "session1", copied)
    source = copied / "staged_source.py"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(copied)
