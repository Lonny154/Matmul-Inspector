"""CPU-only checks for the bounded fused Strassen scaling study."""

import ast
import copy
import importlib.util
import json

import pytest

from matmul_inspector import experiment_pair
from scripts import run_mi_research_4c_feasibility as study


def _function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def fixture_raw(size):
    return {"schema_version": 1, "size": size, "grid": [size // 32, size // 32],
            "configured_ctas": (size // 32) ** 2, "case_names": list(study.CASES),
            "tolerance": study.TOLERANCE, "rank7_sha256": study.RANK7_SHA256,
            "rank8_sha256": study.RANK8_SHA256,
            "precision": {"fp32_matmul_precision": "highest", "allow_tf32": False,
                          "allow_fp16_reduced_precision_reduction": False},
            "cases": [{"case": name, "checks": {
                label: {"passed": True, "failing_elements": 0} for label in study.LABELS}}
                for name in study.CASES],
            "profiles": {label: {"kernel_events": 1, "launch_calls": 1,
                                 "expected_kernel_seen": True} for label in study.LABELS},
            "compiler": {label: {"metadata": {"n_spills": 0, "global_scratch_size": 0},
                                 "ptx_sha256": "hash"} for label in study.LABELS}}


def test_scaling_sources_preserve_committed_kernel_bodies():
    previous_a = study.ROOT / "examples/kernels/mi_research_4b_fused_rank8.py"
    previous_b = study.ROOT / "examples/kernels/mi_research_4_single_kernel_strassen.py"
    for source, prior, kernel in ((study.RANK8, previous_a, "_rank8_kernel"),
                                  (study.RANK7, previous_b, "_strassen_kernel")):
        assert ast.dump(_function(source, "_product")) == ast.dump(_function(prior, "_product"))
        assert ast.dump(_function(source, kernel)) == ast.dump(_function(prior, kernel))
        assert study.artifacts.digest(source) == (
            study.RANK8_SHA256 if kernel == "_rank8_kernel" else study.RANK7_SHA256)


@pytest.mark.parametrize("size", study.SIZES)
def test_pair_specs_match_frozen_sources_and_budget(size):
    spec = experiment_pair.validate_spec(json.loads(
        (study.ROOT / f"examples/mi_research_4c_n{size}_pair.json").read_text()))
    assert spec["workload"] == {"m": size, "n": size, "k": size, "dtype": "float16"}
    assert spec["correctness"] == study.TOLERANCE
    assert spec["benchmark"] == {"mode": "preallocated", "warmups": 10,
                                  "iterations": 20, "rounds": 8}
    assert all(spec["kernels"][label]["implementation"] == "triton_source"
               for label in study.LABELS)
    assert spec["kernels"]["A"]["parameters"] == spec["kernels"]["B"]["parameters"]
    assert spec["kernels"]["A"]["source"]["sha256"] == study.RANK8_SHA256
    assert spec["kernels"]["B"]["source"]["sha256"] == study.RANK7_SHA256


@pytest.mark.parametrize("size", study.SIZES)
def test_correctness_topology_resource_and_grid_gates(size):
    raw = fixture_raw(size)
    assert study.summarize(raw)["performance_gate_passed"]
    for field, value in (("passed", False), ("failing_elements", 1)):
        bad = copy.deepcopy(raw)
        bad["cases"][0]["checks"]["B"][field] = value
        assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["profiles"]["A"]["launch_calls"] = 2
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["compiler"]["B"]["metadata"]["n_spills"] = 1
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["grid"] = [1, 1]
    with pytest.raises(ValueError, match="protocol"):
        study.summarize(bad)


@pytest.mark.parametrize("source", [study.RANK8, study.RANK7])
def test_scaling_wrapper_rejects_cpu_and_wrong_config(source):
    import torch
    spec = importlib.util.spec_from_file_location(f"test_{source.stem}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = torch.zeros((256, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match="configuration"):
        module.matmul_into(a, a, a, {**module.PARAMETERS, "num_stages": 3})
    with pytest.raises(ValueError, match="CUDA FP16"):
        module.matmul_into(a, a, a, module.PARAMETERS)


def test_offline_bundle_replay_and_hash_tamper_detection(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for source, name in ((study.RANK8, "rank8_source.py"),
                         (study.RANK7, "rank7_source.py"),
                         (study.ROOT / "scripts/run_mi_research_4c_feasibility.py",
                          "diagnostic_script.py")):
        (bundle / name).write_bytes(source.read_bytes())
    raw = fixture_raw(256)
    raw["script_sha256"] = study.artifacts.digest(bundle / "diagnostic_script.py")
    raw["trace_sha256"] = {}
    for label, kernel in (("A", "_rank8_kernel"), ("B", "_strassen_kernel")):
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
