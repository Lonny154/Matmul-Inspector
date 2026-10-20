"""CPU-only gates for the bounded single-kernel Strassen study."""

import ast
import copy
import json
import os
import time

import pytest

from matmul_inspector.tensor_game import strassen_2x2_decomposition
from matmul_inspector import experiment_pair
from scripts import run_mi_research_4_feasibility as study


def fixture_raw():
    return {"schema_version": 2, "sizes": list(study.SIZES),
            "case_names": list(study.CASE_NAMES), "tolerance": study.TOLERANCE,
            "precision": {"fp32_matmul_precision": "highest", "allow_tf32": False,
                          "allow_fp16_reduced_precision_reduction": False},
            "cases": [{"size": size, "case": name,
                       "checks": {label: {"passed": True}
                                  for label in ("A", "B", "cublas_no_reduced")}}
                      for size in study.SIZES for name in study.CASE_NAMES],
            "profiles": {str(size): {label: {"kernel_events": 1, "cuda_launch_calls": 1,
                                            "expected_kernel_seen": True}
                                     for label in ("A", "B")} for size in study.SIZES},
            "compiler": {str(size): {label: {"metadata": {"n_spills": 0}, "ptx_sha256": "hash"}
                                     for label in ("A", "B")} for size in study.SIZES}}


def test_source_product_coefficients_match_exact_strassen():
    tree = ast.parse(study.SOURCE.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_strassen_kernel")
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_product"]
    actual = [tuple(ast.literal_eval(arg) for arg in call.args[-8:]) for call in calls]
    expected = [tuple(action.u + action.v) for action in strassen_2x2_decomposition()]
    assert actual == expected


def test_profiler_gate_requires_one_named_launch():
    events = {"traceEvents": [{"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0},
                              {"cat": "kernel", "name": "_strassen_kernel", "ts": 1}]}
    assert study.profile_summary(events, "B") == {
        "kernel_names": ["_strassen_kernel"], "kernel_events": 1,
        "cuda_launch_calls": 1, "expected_kernel_seen": True}
    events["traceEvents"].append({"cat": "kernel", "name": "unexpected", "ts": 2})
    assert not study.profile_summary(events, "B")["expected_kernel_seen"]


def test_profiler_counts_driver_launch_without_double_counting_runtime():
    events = {"traceEvents": [{"cat": "cuda_driver", "name": "cuLaunchKernelEx", "ts": 0},
                              {"cat": "kernel", "name": "_strassen_kernel", "ts": 1}]}
    assert study.profile_summary(events, "B")["cuda_launch_calls"] == 1
    assert study.profile_summary(events, "B", schema_version=1)["cuda_launch_calls"] == 0
    events["traceEvents"].append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2})
    assert study.profile_summary(events, "B")["cuda_launch_calls"] == 1
    events["traceEvents"].append({"cat": "cuda_driver", "name": "cuLaunchKernelEx", "ts": 3})
    assert study.profile_summary(events, "B")["cuda_launch_calls"] == 2


def test_correctness_topology_and_spill_gates():
    raw = fixture_raw()
    assert study.summarize(raw)["performance_gate_passed"]
    invalid = copy.deepcopy(raw)
    invalid["cases"][0]["checks"]["B"]["passed"] = False
    assert not study.summarize(invalid)["performance_gate_passed"]
    invalid = copy.deepcopy(raw)
    invalid["profiles"]["128"]["B"]["cuda_launch_calls"] = 2
    assert not study.summarize(invalid)["performance_gate_passed"]
    invalid = copy.deepcopy(raw)
    invalid["compiler"]["64"]["B"]["metadata"]["n_spills"] = 1
    assert not study.summarize(invalid)["performance_gate_passed"]
    invalid = copy.deepcopy(raw)
    invalid["precision"]["allow_tf32"] = True
    with pytest.raises(ValueError, match="protocol"):
        study.summarize(invalid)


def test_local_wrapper_rejects_unapproved_configuration_without_gpu():
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("mi_research_4_source_test", study.SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = torch.zeros((64, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="configuration"):
        module.matmul_into(a, a, a, {**module.PARAMETERS, "num_stages": 3})
    with pytest.raises(ValueError, match="CUDA FP16"):
        module.matmul_into(a, a, a, module.PARAMETERS)


def test_pair_worker_drains_large_compiler_capture_before_join(monkeypatch):
    class LargeCaptureBackend:
        def run(self, _spec, _log):
            return {"status": "success", "local_compiler_capture": {
                "B": {"ptx_text": "x" * 200_000}}}

    monkeypatch.setattr(experiment_pair, "PairBackend", LargeCaptureBackend)
    spec = experiment_pair.validate_spec(json.loads(
        (study.ROOT / "examples/mi_research_4_n64_pair.json").read_text()))
    spec["timeout_seconds"] = 3
    raw = experiment_pair.run_worker(spec)
    assert raw["status"] == "success"
    assert len(raw["local_compiler_capture"]["B"]["ptx_text"]) == 200_000


def test_pair_worker_reports_exited_child_without_waiting_for_timeout(monkeypatch):
    from matmul_inspector import experiment

    monkeypatch.setattr(experiment, "_worker", lambda *_args: os._exit(1))
    spec = experiment_pair.validate_spec(json.loads(
        (study.ROOT / "examples/mi_research_4_n64_pair.json").read_text()))
    spec["timeout_seconds"] = 3
    start = time.monotonic()
    raw = experiment_pair.run_worker(spec)
    assert time.monotonic() - start < 2
    assert raw["failure"]["stage"] == "execution"
