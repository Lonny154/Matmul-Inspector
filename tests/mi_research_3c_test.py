"""CPU-only tests for the opt-in numerical contract and evidence replay."""

import copy
import json

import pytest

from scripts import run_mi_research_3_graph as graph
from scripts import run_mi_research_3c_correctness as gate
from tests.mi_research_3_test import fixture_raw


def explicit_graph_raw():
    raw = fixture_raw()
    raw["schema_version"] = 2
    raw["reference_policy"] = graph.EXPLICIT_POLICY
    raw["correctness"]["reference"] = "fp32_matmul_fp16_inputs_rounded_fp16"
    raw["correctness"]["cublas_precision"] = "fp16_reduced_precision_reduction_disabled"
    raw["precision_settings"] = {
        "active_fp16_reduced_precision_reduction": False,
        "active_allow_tf32": False,
        "active_float32_matmul_precision": "highest",
    }
    return raw


def test_explicit_graph_policy_replays_and_rejects_wrong_precision():
    raw = explicit_graph_raw()
    assert graph.summarize(raw)["reference_policy"] == graph.EXPLICIT_POLICY
    for field, value in (("active_fp16_reduced_precision_reduction", True),
                         ("active_allow_tf32", True),
                         ("active_float32_matmul_precision", "medium")):
        invalid = copy.deepcopy(raw)
        invalid["precision_settings"][field] = value
        with pytest.raises(ValueError, match="explicit reference"):
            graph.summarize(invalid)
    invalid = copy.deepcopy(raw)
    invalid["correctness"]["seeds"][2]["checks"]["C"]["passed"] = False
    with pytest.raises(ValueError, match="numerical"):
        graph.summarize(invalid)


def test_policy_scope_restores_torch_settings_on_failure(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    old = (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           torch.backends.cuda.matmul.allow_tf32,
           torch.get_float32_matmul_precision())

    def fail(*args):
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
        raise RuntimeError("mocked GPU failure")

    monkeypatch.setattr(graph, "_run", fail)
    try:
        with pytest.raises(RuntimeError, match="mocked GPU failure"):
            graph.run(tmp_path / "unused", 256, 1, graph.EXPLICIT_POLICY)
        assert old == (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                       torch.backends.cuda.matmul.allow_tf32,
                       torch.get_float32_matmul_precision())
    finally:
        torch.set_float32_matmul_precision(old[2])
        torch.backends.cuda.matmul.allow_tf32 = old[1]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old[0]


def test_preflight_summary_requires_all_shapes_seeds_and_gates():
    raw = {"schema_version": 1, "reference_policy": "fp32_rounded_fp16",
           "shape": {"m_values": list(gate.MS), "k": gate.K, "n": gate.N},
           "tolerance": gate.TOLERANCE,
           "cases": [{"m": m, "seed": seed,
                      "checks": {label: {"passed": True} for label in ("A", "B", "C")}}
                     for m in gate.MS for seed in range(3)]}
    assert gate.summarize(raw)["all_passed"]
    raw["cases"][9]["checks"]["B"]["passed"] = False
    assert not gate.summarize(raw)["all_passed"]
    raw["cases"].pop()
    with pytest.raises(ValueError, match="configuration"):
        gate.summarize(raw)


def test_preflight_offline_hash_and_summary_replay(tmp_path):
    raw = {"schema_version": 1, "reference_policy": "fp32_rounded_fp16",
           "shape": {"m_values": list(gate.MS), "k": gate.K, "n": gate.N},
           "tolerance": gate.TOLERANCE,
           "cases": [{"m": m, "seed": seed,
                      "checks": {label: {"passed": True} for label in ("A", "B", "C")}}
                     for m in gate.MS for seed in range(3)]}
    for field, name in (("source_sha256", "source.py"),
                        ("conventional_sha256", "conventional_source.py"),
                        ("script_sha256", "diagnostic_script.py")):
        (tmp_path / name).write_bytes(gate.SOURCE.read_bytes() if name == "source.py" else name.encode())
        raw[field] = gate.artifacts.digest(tmp_path / name)
    gate.artifacts.write_json(tmp_path / "raw.json", raw)
    gate.artifacts.write_json(tmp_path / "summary.json", gate.summarize(raw))
    gate.artifacts.manifest(tmp_path)
    assert gate.verify(tmp_path) == {"status": "PASS", "cases": 12}
    altered = json.loads((tmp_path / "summary.json").read_text())
    altered["all_passed"] = False
    gate.artifacts.write_json(tmp_path / "summary.json", altered)
    with pytest.raises(ValueError, match="manifest"):
        gate.verify(tmp_path)
