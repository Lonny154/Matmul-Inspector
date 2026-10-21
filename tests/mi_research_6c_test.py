"""CPU-only exact, scheduling, artifact, and probe-source checks."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from scripts import run_mi_research_6c as study
from scripts import run_mi_research_6c_gpu_probe as probe


def data():
    return json.loads(study.SOURCE.read_text()), json.loads(study.CERT.read_text())


def test_all_three_exact_configurations_and_two_schedules():
    source, cert = data()
    result = study.analyze(source, cert)
    assert set(result["configurations"]) == {"laderman_98", "perminov_58", "perminov_55"}
    assert [result["configurations"][name]["total_additions"] for name in
            ("laderman_98", "perminov_58", "perminov_55")] == [98, 58, 55]
    assert all(row["tensor"] == {"checked": 729, "unit": 27, "zero": 702, "mismatches": 0}
               for row in result["configurations"].values())
    assert result["configurations"]["perminov_58"]["factor_coefficients"] == \
           result["configurations"]["perminov_55"]["factor_coefficients"]
    for row in result["configurations"].values():
        published, liveness = (row["schedules"][key] for key in ("published", "liveness"))
        assert published["case_output_sha256"] == liveness["case_output_sha256"]
        assert sorted(published["order"]) == sorted(liveness["order"])
        assert liveness["peak_live_intermediates"] <= published["peak_live_intermediates"]
        assert all(value["last_used_at"] >= value["created_at"]
                   for value in liveness["value_lifetimes"].values())


def test_cost_and_depth_tradeoff_are_recorded():
    rows = study.analyze(*data())["configurations"]
    assert rows["perminov_58"]["addition_counts"] == {"U": 14, "V": 15, "W": 29}
    assert rows["perminov_55"]["addition_counts"] == {"U": 13, "V": 14, "W": 28}
    assert rows["perminov_55"]["schedules"]["published"]["peak_dependency_depth_operations"] > \
           rows["perminov_58"]["schedules"]["published"]["peak_dependency_depth_operations"]
    assert rows["perminov_55"]["schedules"]["liveness"]["peak_live_intermediates"] == 17


def test_source_and_certificate_corruption_are_rejected():
    source, cert = data()
    broken = copy.deepcopy(cert)
    broken["circuits"]["U_input_9_to_23"]["gates"][0]["left_sign"] *= -1
    with pytest.raises(ValueError, match="factor mismatch"):
        study.analyze(source, broken)
    broken = copy.deepcopy(source)
    broken["u"][0][0]["value"] *= -1
    with pytest.raises(ValueError, match="factor mismatch"):
        study.analyze(broken, cert)
    broken = copy.deepcopy(cert)
    broken["circuits"]["W_output_task_23_to_9"]["gates"][0]["left"] = 100
    with pytest.raises(ValueError, match="gate operand"):
        study.analyze(source, broken)


def test_rendered_probe_matches_checked_dag_and_disjoint_output_ownership():
    source, cert = data()
    graph, _ = study.build(source, cert, optimized=True)
    order = study.order_and_metrics(graph, liveness=True)["order"]
    rendered = study.render_probe(graph, order)
    assert rendered == study.PROBE_SOURCE.read_text()
    assert rendered.count("tl.dot(") == 23
    assert rendered.count("tl.store(") == 9
    assert "_fused[(2, 2)]" in rendered
    assert "atomic" not in rendered.lower().split("@triton.jit", 1)[1]


def test_offline_manifest_and_gpu_trace_gate(tmp_path):
    assert study.verify()["status"] == "PASS"
    bad = tmp_path / "changed"
    shutil.copytree(study.OUT, bad)
    (bad / "upstream/certificate.json").write_bytes(
        (bad / "upstream/certificate.json").read_bytes() + b" ")
    original = study.OUT
    try:
        study.OUT = bad
        study.UPSTREAM = bad / "upstream"
        study.CERT = study.UPSTREAM / "certificate.json"
        study.SOURCE = study.UPSTREAM / "perminov_source.json"
        with pytest.raises(ValueError, match="upstream certificate"):
            study.verify()
    finally:
        study.OUT = original
        study.UPSTREAM = original / "upstream"
        study.CERT = study.UPSTREAM / "certificate.json"
        study.SOURCE = study.UPSTREAM / "perminov_source.json"
    trace = {"traceEvents": [
        {"cat": "kernel", "name": "_fused",
         "args": {"grid": [2, 2, 1], "block": [128, 1, 1]}},
        {"cat": "cuda_driver", "name": "cuLaunchKernel"}]}
    assert probe.trace_topology(trace)["physical_kernel_events"] == 1
    bad_grid = copy.deepcopy(trace)
    bad_grid["traceEvents"][0]["args"]["grid"] = [1, 1, 1]
    with pytest.raises(ValueError, match="grid"):
        probe.trace_topology(bad_grid)
    trace["traceEvents"].append({"cat": "kernel", "name": "_fused"})
    with pytest.raises(ValueError, match="one complete fused"):
        probe.trace_topology(trace)
