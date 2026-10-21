"""CPU-only checks for the matched fused rank-8/rank-7 experiment."""

import ast
import copy
import json

import pytest

from matmul_inspector import experiment_pair
from matmul_inspector.tensor_game import standard_2x2_decomposition
from scripts import run_mi_research_4b_feasibility as study


def _function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def fixture_raw():
    return {"schema_version": 1, "size": study.SIZE, "case_names": list(study.CASES),
            "tolerance": study.TOLERANCE, "rank7_sha256": study.RANK7_SHA256,
            "precision": {"fp32_matmul_precision": "highest", "allow_tf32": False,
                          "allow_fp16_reduced_precision_reduction": False},
            "cases": [{"case": name, "checks": {
                label: {"passed": True, "failing_elements": 0} for label in study.LABELS}}
                for name in study.CASES],
            "profiles": {label: {"kernel_events": 1, "launch_calls": 1,
                                 "expected_kernel_seen": True} for label in study.LABELS},
            "compiler": {label: {"metadata": {"n_spills": 0, "global_scratch_size": 0},
                                 "ptx_sha256": "hash"} for label in study.LABELS}}


def test_rank8_coefficients_and_output_mapping_match_standard_decomposition():
    helper = _function(study.RANK8, "_product")
    original = _function(study.RANK7, "_product")
    assert ast.dump(helper) == ast.dump(original)
    kernel = _function(study.RANK8, "_rank8_kernel")
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_product"]
    actual = [tuple(ast.literal_eval(arg) for arg in call.args[-8:]) for call in calls]
    expected = [tuple(action.u + action.v) for action in standard_2x2_decomposition()]
    assert actual == expected
    targets = [node.targets[0].id if isinstance(node, ast.Assign) else node.target.id
               for node in kernel.body if isinstance(node, (ast.Assign, ast.AugAssign))
               and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target,
                              ast.Name) and
               (node.targets[0] if isinstance(node, ast.Assign) else node.target).id
               in {"c00", "c01", "c10", "c11"}]
    assert targets == ["c00", "c01", "c00", "c01", "c10", "c11", "c10", "c11"]
    stores = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "store"]
    assert len(stores) == 4


def test_pair_spec_is_two_trusted_local_sources_with_matched_settings():
    spec = experiment_pair.validate_spec(json.loads(
        (study.ROOT / "examples/mi_research_4b_n128_pair.json").read_text()))
    assert spec["schema_version"] == 3
    assert spec["workload"] == {"m": 128, "n": 128, "k": 128, "dtype": "float16"}
    assert spec["benchmark"] == {"mode": "preallocated", "warmups": 10,
                                  "iterations": 20, "rounds": 8}
    assert all(spec["kernels"][label]["implementation"] == "triton_source"
               for label in study.LABELS)
    assert spec["kernels"]["A"]["parameters"] == spec["kernels"]["B"]["parameters"]


def test_profiler_counts_one_driver_launch_and_rejects_extra_kernel():
    trace = {"traceEvents": [{"cat": "cuda_driver", "name": "cuLaunchKernelEx", "ts": 0},
                             {"cat": "kernel", "name": "_rank8_kernel", "ts": 1}]}
    assert study.profile_summary(trace, "A")["launch_calls"] == 1
    assert study.profile_summary(trace, "A")["expected_kernel_seen"]
    trace["traceEvents"].append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2})
    assert study.profile_summary(trace, "A")["launch_calls"] == 1
    trace["traceEvents"].append({"cat": "kernel", "name": "other", "ts": 3})
    assert not study.profile_summary(trace, "A")["expected_kernel_seen"]


def test_correctness_topology_and_resource_gates():
    raw = fixture_raw()
    assert study.summarize(raw)["performance_gate_passed"]
    for field, value in (("passed", False), ("failing_elements", 1)):
        bad = copy.deepcopy(raw)
        bad["cases"][0]["checks"]["A"][field] = value
        assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["profiles"]["B"]["launch_calls"] = 2
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["compiler"]["A"]["metadata"]["global_scratch_size"] = 4
    assert not study.summarize(bad)["performance_gate_passed"]
    bad = copy.deepcopy(raw)
    bad["cases"].pop()
    with pytest.raises(ValueError, match="protocol"):
        study.summarize(bad)


def test_rank8_wrapper_rejects_unapproved_shape_on_cpu():
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("mi_research_4b_source_test", study.RANK8)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = torch.zeros((128, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="configuration"):
        module.matmul_into(a, a, a, {**module.PARAMETERS, "num_stages": 3})
    with pytest.raises(ValueError, match="CUDA FP16"):
        module.matmul_into(a, a, a, module.PARAMETERS)


def test_offline_bundle_replay_and_hash_tamper_detection(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for source, name in ((study.RANK8, "rank8_source.py"),
                         (study.RANK7, "rank7_source.py"),
                         (study.ROOT / "scripts/run_mi_research_4b_feasibility.py",
                          "diagnostic_script.py")):
        (bundle / name).write_bytes(source.read_bytes())
    raw = fixture_raw()
    raw["rank8_sha256"] = study.artifacts.digest(bundle / "rank8_source.py")
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
