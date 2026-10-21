"""CPU-only contract tests for the bounded algebraic exploration."""

import ast
import copy
import json

import pytest

from matmul_inspector import experiment_pair
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
from scripts import generate_mi_research_5_candidates as selection
from scripts import run_mi_research_5_feasibility as study


def fixture_raw(label="peer_corpus", size=128):
    return {"schema_version": 1, "label": label, "size": size,
            "grid": [size // 32, size // 32],
            "configured_ctas": (size // 32) ** 2,
            "cases_expected": list(study.CASES), "tolerance": study.TOLERANCE,
            "baseline_sha256": selection.BASELINE_SHA256,
            "candidate_sha256": selection.verify_source_hash(label),
            "precision": {"fp32_matmul_precision": "highest", "allow_tf32": False,
                          "allow_fp16_reduced_precision_reduction": False},
            "cases": [{"case": name, "checks": {
                key: {"passed": True, "failing_elements": 0,
                      "candidate_nonfinite_count": 0} for key in study.LABELS}}
                for name in study.CASES],
            "profiles": {key: {"kernel_events": 1, "launch_calls": 1,
                                "expected_kernel_seen": True} for key in study.LABELS},
            "compiler": {key: {"metadata": {"n_spills": 0, "global_scratch_size": 0},
                               "ptx_sha256": "hash"} for key in study.LABELS}}


def test_selection_exactness_provenance_and_liveness():
    assert selection.verify() == {"status": "PASS", "representations": 2, "sources": 3}
    records = selection.select_records()
    assert len({row["structural_signature_id"] for row in records.values()}) == 2
    assert all(row["known_equivalence"] == "exact GL(2,Z)^3 change of basis"
               for row in records.values())
    peer = selection.actions_of(records[selection.CHOICES["peer_corpus"][0]])
    assert selection.live_score(peer, tuple(range(7))) == (3, 14)
    assert selection.live_score(peer, selection.CHOICES["peer_liveness"][1]) == (2, 10)
    bad = list(peer)
    bad[0] = type(peer[0])((3, *peer[0].u[1:]), peer[0].v, peer[0].w)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        validate_decomposition(bad)


@pytest.mark.parametrize("label", tuple(selection.CHOICES))
def test_frozen_kernel_and_pair_contract(label):
    record_id, order = selection.CHOICES[label]
    source = selection.SOURCES / f"mi_research_5_{label}.py"
    tree = ast.parse(source.read_text())
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_variant_kernel")
    product_calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "_product"]
    assert len(product_calls) == 7
    assert selection.digest(source) == selection.verify_source_hash(label)
    assert source.read_text() == selection.render(selection.select_records()[record_id], order)
    spec_path = selection.ROOT / f"examples/mi_research_5_{label}_n256_pair.json"
    spec = experiment_pair.validate_spec(json.loads(spec_path.read_text()))
    assert spec["workload"] == {"m": 256, "n": 256, "k": 256, "dtype": "float16"}
    assert spec["correctness"] == study.TOLERANCE
    assert spec["benchmark"] == {"mode": "preallocated", "warmups": 10,
                                  "iterations": 20, "rounds": 8}
    assert spec["kernels"]["A"]["source"]["sha256"] == selection.BASELINE_SHA256
    assert spec["kernels"]["B"]["source"]["sha256"] == selection.digest(source)


@pytest.mark.parametrize("size", study.SIZES)
def test_numerical_topology_and_resource_gates(size):
    raw = fixture_raw(size=size)
    assert study.summarize(raw)["performance_gate_passed"]
    for field, value in (("passed", False), ("failing_elements", 1),
                         ("candidate_nonfinite_count", 1)):
        bad = copy.deepcopy(raw)
        bad["cases"][0]["checks"]["B"][field] = value
        assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["profiles"]["B"]["launch_calls"] = 2
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["compiler"]["B"]["metadata"]["n_spills"] = 1
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["grid"] = [1, 1]
    with pytest.raises(ValueError, match="protocol"):
        study.summarize(bad)


def test_offline_replay_and_tamper_detection(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for source, name in ((selection.BASELINE, "baseline_source.py"),
                         (selection.SOURCES / "mi_research_5_peer_corpus.py", "candidate_source.py"),
                         (selection.ROOT / "scripts/run_mi_research_5_feasibility.py",
                          "diagnostic_script.py")):
        (bundle / name).write_bytes(source.read_bytes())
    raw = fixture_raw()
    raw["script_sha256"] = study.artifacts.digest(bundle / "diagnostic_script.py")
    raw["trace_sha256"] = {}
    for label, kernel in (("A", "_strassen_kernel"), ("B", "_variant_kernel")):
        trace = {"traceEvents": [{"cat": "cuda_driver", "name": "cuLaunchKernelEx", "ts": 0},
                                 {"cat": "kernel", "name": kernel, "ts": 1}]}
        study.artifacts.write_json(bundle / f"{label}_trace.json", trace)
        (bundle / f"{label}.ptx").write_text(f"// {kernel}\n")
        raw["profiles"][label] = study.profile_summary(trace, label)
        raw["trace_sha256"][label] = study.artifacts.digest(bundle / f"{label}_trace.json")
        raw["compiler"][label]["ptx_sha256"] = study.artifacts.digest(bundle / f"{label}.ptx")
        raw["compiler"][label]["ir_sha256"] = {}
    study.artifacts.write_json(bundle / "raw.json", raw)
    study.artifacts.write_json(bundle / "summary.json", study.summarize(raw))
    study.artifacts.manifest(bundle)
    assert study.verify(bundle)["status"] == "PASS"
    (bundle / "A.ptx").write_text("tampered\n")
    with pytest.raises(ValueError, match="manifest"):
        study.verify(bundle)


def test_direct_schedule_pair_is_matched():
    path = selection.ROOT / "examples/mi_research_5_schedule_n256_pair.json"
    spec = experiment_pair.validate_spec(json.loads(path.read_text()))
    assert spec["kernels"]["A"]["source"]["sha256"] == selection.verify_source_hash("peer_corpus")
    assert spec["kernels"]["B"]["source"]["sha256"] == selection.verify_source_hash("peer_liveness")
    assert spec["kernels"]["A"]["parameters"] == spec["kernels"]["B"]["parameters"]
