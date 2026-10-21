"""CPU-only checks for the frozen graph comparison and retained evidence."""

from __future__ import annotations

import copy
import json

import pytest

from scripts import analyze_mi_research_6d as analysis
from scripts import run_mi_research_6d_graph as study


def synthetic_trace(label: str) -> dict:
    names = (["_block_product"] * 23 + ["_reconstruct"]) if label == "A" else ["_fused"]
    return {"traceEvents": [{"cat": "cuda_runtime", "name": "cudaGraphLaunch"}] + [
        {"cat": "kernel", "name": name, "ts": index,
         "args": {"grid": [2, 2, 1], "block": [128, 1, 1]}}
        for index, name in enumerate(names * study.OPERATIONS)]}


@pytest.mark.parametrize("label,count", [("A", 480), ("B", 20)])
def test_exact_physical_topology(label, count):
    trace = synthetic_trace(label)
    assert study.trace_profile(trace, label)["kernel_events"] == count
    missing = copy.deepcopy(trace)
    missing["traceEvents"].pop()
    with pytest.raises(ValueError, match="expected"):
        study.trace_profile(missing, label)
    reordered = copy.deepcopy(trace)
    reordered["traceEvents"][1]["name"] = "_fused" if label == "A" else "_reconstruct"
    with pytest.raises(ValueError, match="incomplete or reordered"):
        study.trace_profile(reordered, label)
    no_submission = copy.deepcopy(trace)
    no_submission["traceEvents"].pop(0)
    with pytest.raises(ValueError, match="graph submission"):
        study.trace_profile(no_submission, label)


def test_frozen_sources_and_cross_session_replay(tmp_path):
    assert study.artifacts.digest(study.STAGED) == study.previous.SOURCE_SHA256
    assert study.artifacts.digest(study.FUSED) == study.FUSED_SHA256
    assert study.artifacts.digest(study.COEFFICIENTS) == study.previous.COEFFICIENTS_SHA256
    assert study.circuit.verify()["status"] == "PASS"
    assert study.fused_probe.verify()["status"] == "PASS"
    for session in (1, 2):
        assert study.verify(analysis.EVIDENCE / f"graph-s{session}") == {
            "status": "PASS", "session": session, "decision": "INCONCLUSIVE"}
    assert analysis.run(verify=True) == {"status": "PASS", "classification": "C",
                                         "session_decisions": ["INCONCLUSIVE", "INCONCLUSIVE"]}
    assert all(row["decision"] == "INCONCLUSIVE" for row in analysis.analyze()["sessions"])
    bundle = tmp_path / "changed_bundle"
    import shutil
    shutil.copytree(analysis.EVIDENCE / "graph-s1", bundle)
    trace = bundle / "B_graph_trace.json"
    trace.write_bytes(trace.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest mismatch"):
        study.verify(bundle)


def test_protocol_rejects_missing_graph_gate_and_timing_corruption():
    raw = json.loads((analysis.EVIDENCE / "graph-s1/raw.json").read_text())
    assert study.summarize(raw)["decision"]["decision"] == "INCONCLUSIVE"
    bad = copy.deepcopy(raw)
    bad["correctness"]["cases"][0]["replays"]["B"][0]["passed"] = False
    with pytest.raises(ValueError, match="numerical correctness"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["graph_profile"]["B"]["kernel_events"] -= 1
    with pytest.raises(ValueError, match="topology"):
        study.summarize(bad)
    bad = copy.deepcopy(raw)
    bad["benchmark"]["rounds"][0]["measurements"]["A"]["complete_operations_per_block"] -= 1
    with pytest.raises(ValueError, match="normalization"):
        study.summarize(bad)
