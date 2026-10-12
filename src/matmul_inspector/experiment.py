"""Reproducible baseline-versus-candidate experiment orchestration.

The orchestration and result schema deliberately have no CUDA dependency.  The
CUDA/Triton implementation is imported lazily by :class:`CudaTritonBackend`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import multiprocessing
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid


SCHEMA_VERSION = 2
BENCHMARK_MODES = frozenset({"allocating", "preallocated"})
SOURCE_IMPLEMENTATION = "triton_source"
SOURCE_INTERFACE_VERSION = "matmul_into_v1"
SOURCE_ENTRY_POINT = "matmul_into"
PARAMETER_NAMES = frozenset({
    "block_m", "block_n", "block_k", "num_warps", "num_stages", "group_size_m",
})
DEFAULT_CONFIG: dict[str, Any] = {
    "workload": {"m": 512, "k": 768, "n": 3072, "dtype": "float16"},
    "seed": 0,
    "correctness": {"atol": 1e-2, "rtol": 1e-2},
    "benchmark": {"mode": "allocating", "warmups": 100, "iterations": 500, "rounds": 10},
    "timeout_seconds": 300.0,
    "baseline": {"implementation": "pytorch_matmul"},
    "candidate": {
        "implementation": "triton_fc1",
        "parameters": {
            "block_m": 64, "block_n": 128, "block_k": 32,
            "num_warps": 4, "num_stages": 3, "group_size_m": 1,
        },
    },
}


def resolved_benchmark_mode(value: Mapping[str, Any]) -> str:
    """Return a mode while interpreting legacy omitted fields as allocating."""
    benchmark = value.get("benchmark") if isinstance(value, Mapping) else None
    if isinstance(benchmark, Mapping) and isinstance(benchmark.get("mode"), str):
        return benchmark["mode"]
    config = value.get("config") if isinstance(value, Mapping) else None
    if isinstance(config, Mapping):
        return resolved_benchmark_mode(config)
    return "allocating"


class StageFailure(RuntimeError):
    """A failure whose experiment stage is known."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class ExperimentBackend(Protocol):
    def run(self, config: Mapping[str, Any], log: Callable[[str], None]) -> dict[str, Any]: ...


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized, JSON-compatible configuration."""
    if not isinstance(raw, Mapping):
        raise ValueError("configuration must be a JSON object")
    allowed = {"workload", "seed", "correctness", "benchmark", "timeout_seconds",
               "baseline", "candidate"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")
    config = json.loads(json.dumps(raw))
    for required in allowed:
        if required not in config:
            raise ValueError(f"missing configuration field: {required}")

    workload = config["workload"]
    if not isinstance(workload, dict):
        raise ValueError("workload must be an object")
    for dimension in ("m", "k", "n"):
        _positive_int(workload.get(dimension), f"workload.{dimension}")
    if workload.get("dtype") != "float16":
        raise ValueError("only workload.dtype='float16' is currently supported")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")

    correctness = config["correctness"]
    if not isinstance(correctness, dict):
        raise ValueError("correctness must be an object")
    for tolerance in ("atol", "rtol"):
        value = correctness.get(tolerance)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"correctness.{tolerance} must be a non-negative number")

    benchmark = config["benchmark"]
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be an object")
    benchmark.setdefault("mode", "allocating")
    unknown_benchmark = sorted(set(benchmark) - {"mode", "warmups", "iterations", "rounds"})
    if unknown_benchmark:
        raise ValueError(f"unknown benchmark fields: {', '.join(unknown_benchmark)}")
    if benchmark["mode"] not in BENCHMARK_MODES:
        raise ValueError("benchmark.mode must be 'allocating' or 'preallocated'")
    for field in ("warmups", "iterations", "rounds"):
        _positive_int(benchmark.get(field), f"benchmark.{field}")
    timeout = config["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive number")

    baseline, candidate = config["baseline"], config["candidate"]
    if not isinstance(baseline, dict) or baseline.get("implementation") != "pytorch_matmul":
        raise ValueError("baseline.implementation must be 'pytorch_matmul'")
    implementation = candidate.get("implementation") if isinstance(candidate, dict) else None
    if implementation not in {"triton_fc1", SOURCE_IMPLEMENTATION}:
        raise ValueError("candidate.implementation must be 'triton_fc1' or 'triton_source'")
    parameters = candidate.get("parameters")
    expected = PARAMETER_NAMES
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("candidate.parameters must contain exactly: " + ", ".join(sorted(expected)))
    for field in expected:
        _positive_int(parameters[field], f"candidate.parameters.{field}")
    if workload["k"] % parameters["block_k"]:
        raise ValueError("workload.k must be divisible by candidate.parameters.block_k")
    if implementation == SOURCE_IMPLEMENTATION:
        if benchmark["mode"] != "preallocated":
            raise ValueError("triton_source candidates support benchmark.mode='preallocated' only")
        source = candidate.get("source")
        if not isinstance(source, dict):
            raise ValueError("candidate.source must be an object for triton_source")
        allowed_source = {"path", "entry_point", "interface_version", "sha256", "original_path"}
        unknown_source = sorted(set(source) - allowed_source)
        if unknown_source:
            raise ValueError("unknown candidate.source fields: " + ", ".join(unknown_source))
        path = source.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("candidate.source.path must be a non-empty local path")
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(f"candidate source file does not exist: {resolved_path}")
        try:
            source_bytes = resolved_path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read candidate source file: {resolved_path}: {error}") from error
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        supplied_hash = source.get("sha256")
        if supplied_hash is not None and supplied_hash != source_hash:
            raise ValueError("candidate source SHA-256 does not match the referenced file")
        if source.get("entry_point") != SOURCE_ENTRY_POINT:
            raise ValueError(f"candidate.source.entry_point must be '{SOURCE_ENTRY_POINT}'")
        if source.get("interface_version") != SOURCE_INTERFACE_VERSION:
            raise ValueError(
                f"candidate.source.interface_version must be '{SOURCE_INTERFACE_VERSION}'")
        candidate["source"] = {
            "path": str(resolved_path),
            "original_path": source.get("original_path", str(resolved_path)),
            "sha256": source_hash,
            "entry_point": SOURCE_ENTRY_POINT,
            "interface_version": SOURCE_INTERFACE_VERSION,
        }
    elif set(candidate) != {"implementation", "parameters"}:
        raise ValueError("triton_fc1 candidate must contain exactly implementation and parameters")
    return config


def snapshot_candidate_source(config: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    """Copy a validated source candidate byte-for-byte and return its preserved identity."""
    normalized = validate_config(config)
    source = normalized["candidate"].get("source")
    if source is None:
        return normalized
    source_path = Path(source["path"])
    source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("candidate source changed while it was being preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_bytes)
    normalized["candidate"]["source"] = {
        **source, "path": str(destination.resolve()), "sha256": digest,
    }
    return normalized


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


class CudaTritonBackend:
    """Adapter for PyTorch/cuBLAS and registered or preserved Triton sources."""

    def _imports(self, implementation: str):
        try:
            import torch
        except ImportError as error:
            raise StageFailure("prerequisite", "PyTorch is required; install a CUDA-enabled PyTorch build") from error
        try:
            import triton  # noqa: F401
        except ImportError as error:
            raise StageFailure("prerequisite", "Triton is required; install a version compatible with PyTorch and CUDA") from error
        if not torch.cuda.is_available():
            raise StageFailure("prerequisite", "CUDA is unavailable; use a host with a supported NVIDIA GPU and CUDA-enabled PyTorch")
        if implementation == SOURCE_IMPLEMENTATION:
            return torch, None
        try:
            module = importlib.import_module(".triton_fc1", package=__package__)
        except Exception as error:
            raise StageFailure("prerequisite", f"cannot import packaged Triton kernel: {error}") from error
        return torch, module

    @staticmethod
    def _load_source(config: Mapping[str, Any]):
        source = config["candidate"]["source"]
        path = Path(source["path"])
        spec = importlib.util.spec_from_file_location(
            f"matmul_inspector_candidate_{source['sha256'][:16]}", path)
        if spec is None or spec.loader is None:
            raise StageFailure("source_loading", f"cannot create module loader for {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise StageFailure("source_loading", f"cannot import preserved candidate: {error}") from error
        entry = getattr(module, source["entry_point"], None)
        if not callable(entry):
            raise StageFailure(
                "interface", f"candidate must define callable {source['entry_point']}")
        try:
            signature = inspect.signature(entry)
            signature.bind(object(), object(), object(), {})
        except (TypeError, ValueError) as error:
            raise StageFailure(
                "interface", "matmul_into must accept (a, b, out, parameters)") from error
        return entry

    @staticmethod
    def _measure(torch, function: Callable[[], Any], iterations: int) -> dict[str, float]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        submission_end = time.perf_counter_ns()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        elapsed = float(start.elapsed_time(end))
        return {
            "cuda_event_elapsed_ms_per_operation": elapsed / iterations,
            "cuda_event_elapsed_ms_per_block": elapsed,
            "host_submission_ms_per_block": (submission_end - wall_start) / 1e6,
            "host_wall_ms_per_block": (wall_end - wall_start) / 1e6,
            "cuda_event_minus_host_wall_ms_per_block": (
                elapsed - (wall_end - wall_start) / 1e6),
        }

    def run(self, config: Mapping[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
        log("prerequisite: checking PyTorch, Triton, and CUDA")
        implementation = config["candidate"]["implementation"]
        torch, kernel = self._imports(implementation)
        source_entry = self._load_source(config) if implementation == SOURCE_IMPLEMENTATION else None
        try:
            device_index = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(device_index)
            versions = {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "triton": __import__("triton").__version__, "cuda_runtime": torch.version.cuda,
            }
            device = {
                "type": "cuda", "index": device_index, "name": device_properties.name,
                "compute_capability": f"{device_properties.major}.{device_properties.minor}",
                "total_memory_bytes": device_properties.total_memory,
            }
        except Exception as error:
            raise StageFailure("prerequisite", f"failed to query CUDA device: {error}") from error

        workload = config["workload"]
        log("setup: creating identical seeded FP16 inputs for both implementations")
        try:
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            a = torch.randn((workload["m"], workload["k"]), device="cuda", dtype=torch.float16)
            b = torch.randn((workload["k"], workload["n"]), device="cuda", dtype=torch.float16)
        except Exception as error:
            raise StageFailure("execution", f"input preparation failed: {error}") from error

        benchmark = config["benchmark"]
        mode = benchmark["mode"]
        parameters = config["candidate"]["parameters"]
        triton_config = (kernel.TritonGemmConfig(**parameters)
                         if implementation == "triton_fc1" else None)
        overwrite_verified: dict[str, bool | None] = {"baseline": None, "candidate": None}
        if mode == "allocating":
            baseline = lambda: torch.matmul(a, b)
            candidate = lambda: kernel.triton_gemm(a, b, triton_config)
            wrapper_work = {
                "baseline": "torch.matmul allocates its output per call",
                "candidate": "triton_gemm validates inputs, allocates output, constructs the grid, and launches per call",
            }
        else:
            try:
                baseline_output = torch.empty((workload["m"], workload["n"]),
                                              device=a.device, dtype=a.dtype)
                candidate_output = torch.empty_like(baseline_output)
                grid = ((kernel.triton.cdiv(workload["m"], triton_config.block_m)
                         * kernel.triton.cdiv(workload["n"], triton_config.block_n),)
                        if implementation == "triton_fc1" else None)
            except Exception as error:
                raise StageFailure("setup", f"preallocated output preparation failed: {error}") from error

            def baseline():
                return torch.matmul(a, b, out=baseline_output)

            def candidate():
                if implementation == SOURCE_IMPLEMENTATION:
                    returned = source_entry(a, b, candidate_output, dict(parameters))
                    if returned is not None:
                        raise StageFailure("interface", "matmul_into must return None")
                else:
                    kernel.fc1_kernel[grid](
                        a, b, candidate_output, workload["m"], workload["n"], workload["k"],
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        candidate_output.stride(0), candidate_output.stride(1),
                        BLOCK_M=triton_config.block_m, BLOCK_N=triton_config.block_n,
                        BLOCK_K=triton_config.block_k, GROUP_SIZE_M=triton_config.group_size_m,
                        num_warps=triton_config.num_warps, num_stages=triton_config.num_stages,
                    )
                return candidate_output

            baseline_output.fill_(float("nan"))
            candidate_output.fill_(float("nan"))
            wrapper_work = {
                "baseline": "torch.matmul(..., out=baseline_output) dispatches per call; output allocation is outside measurement",
                "candidate": ("the preserved source entry point runs per call with prepared inputs, output, and parameters; source loading, validation, and output allocation are outside measurement"
                              if implementation == SOURCE_IMPLEMENTATION else
                              "the prepared Triton grid launch wrapper runs per call; validation, grid construction, and output allocation are outside measurement"),
            }
        setup_ms: dict[str, float] = {}
        outputs: dict[str, Any] = {}
        for name, function in (("baseline", baseline), ("candidate", candidate)):
            log(f"compilation: first synchronized {name} invocation")
            started = time.perf_counter()
            try:
                outputs[name] = function()
                torch.cuda.synchronize()
            except Exception as error:
                stage = "compilation" if name == "candidate" else "execution"
                raise StageFailure(stage, f"{name} setup invocation failed: {error}") from error
            setup_ms[name] = (time.perf_counter() - started) * 1000.0
            if mode == "preallocated":
                overwrite_verified[name] = bool(torch.isfinite(outputs[name]).all().item())

        log("correctness: comparing candidate with PyTorch/cuBLAS reference")
        try:
            reference, actual = outputs["baseline"], outputs["candidate"]
            shape_match = tuple(reference.shape) == tuple(actual.shape)
            dtype_match = reference.dtype == actual.dtype
            reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
            candidate_nonfinite = int((~torch.isfinite(actual)).sum().item()) if shape_match else None
            if shape_match:
                tolerances = config["correctness"]
                passed = bool(dtype_match and reference_nonfinite == 0 and candidate_nonfinite == 0 and torch.allclose(
                    actual, reference, atol=tolerances["atol"], rtol=tolerances["rtol"]))
                if reference_nonfinite == 0 and candidate_nonfinite == 0:
                    difference = (reference.float() - actual.float()).abs()
                    max_error = float(difference.max().item())
                    mean_error = float(difference.mean().item())
                else:
                    max_error = mean_error = None
            else:
                max_error = mean_error = None
                passed = False
        except Exception as error:
            raise StageFailure("correctness", f"output comparison failed: {error}") from error
        correctness = {
            "passed": passed, "shape_match": shape_match, "dtype_match": dtype_match,
            "reference_shape": list(reference.shape), "candidate_shape": list(actual.shape),
            "max_absolute_error": max_error, "mean_absolute_error": mean_error,
            "reference_nonfinite_count": reference_nonfinite,
            "candidate_nonfinite_count": candidate_nonfinite,
            "atol": config["correctness"]["atol"], "rtol": config["correctness"]["rtol"],
            "reference": "pytorch_matmul (cuBLAS)",
            "candidate_accumulation": (
                "candidate-source-defined computation with preallocated FP16 output"
                if implementation == SOURCE_IMPLEMENTATION else
                "FP32 Triton tl.dot accumulator with FP16 output"),
            "reference_accumulation": "PyTorch/cuBLAS implementation-selected FP16 GEMM accumulation",
            "preallocated_output_overwrite_verified": overwrite_verified,
        }
        if implementation == SOURCE_IMPLEMENTATION and passed:
            validation_cases = []
            for validation_seed in (config["seed"] + 1, config["seed"] + 2):
                try:
                    torch.manual_seed(validation_seed)
                    torch.cuda.manual_seed_all(validation_seed)
                    validation_a = torch.randn(tuple(a.shape), device=a.device, dtype=a.dtype)
                    validation_b = torch.randn(tuple(b.shape), device=b.device, dtype=b.dtype)
                    validation_reference = torch.matmul(validation_a, validation_b)
                    candidate_output.fill_(float("nan"))
                    returned = source_entry(
                        validation_a, validation_b, candidate_output, dict(parameters))
                    if returned is not None:
                        raise StageFailure("interface", "matmul_into must return None")
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(candidate_output).all().item())
                    validation_passed = bool(
                        tuple(candidate_output.shape) == tuple(validation_reference.shape)
                        and candidate_output.dtype == validation_reference.dtype
                        and finite
                        and torch.allclose(candidate_output, validation_reference,
                                           atol=config["correctness"]["atol"],
                                           rtol=config["correctness"]["rtol"])
                    )
                    difference = (validation_reference.float() - candidate_output.float()).abs()
                    validation_cases.append({
                        "seed": validation_seed, "passed": validation_passed,
                        "overwrite_verified": finite,
                        "shape_match": tuple(candidate_output.shape) == tuple(validation_reference.shape),
                        "dtype_match": candidate_output.dtype == validation_reference.dtype,
                        "candidate_nonfinite_count": int((~torch.isfinite(candidate_output)).sum().item()),
                        "max_absolute_error": float(difference.max().item()) if finite else None,
                        "mean_absolute_error": float(difference.mean().item()) if finite else None,
                    })
                except StageFailure:
                    raise
                except Exception as error:
                    raise StageFailure("execution", f"source validation invocation failed: {error}") from error
            correctness["benchmark_input_seed"] = config["seed"]
            correctness["validation_seeds"] = [case["seed"] for case in validation_cases]
            correctness["validation_cases"] = validation_cases
            correctness["finite_validation_note"] = (
                "Passing these seeded inputs does not prove correctness for every possible input.")
            passed = all(case["passed"] for case in validation_cases)
            correctness["passed"] = passed
        partial = {
            "environment": {"device": device, "versions": versions},
            "setup": {"measurement": "host wall time through first synchronized invocation", "unit": "ms", "duration_ms": setup_ms},
            "correctness": correctness,
        }
        if not passed:
            return {**partial, "status": "failed", "failure": {
                "stage": "correctness", "exception_type": "CorrectnessFailure",
                "message": "candidate output did not satisfy shape, finiteness, and tolerance requirements",
                "traceback": None,
            }}

        log(f"benchmark: warming each implementation {benchmark['warmups']} times")
        try:
            for function in (baseline, candidate):
                for _ in range(benchmark["warmups"]):
                    function()
                torch.cuda.synchronize()
            log("benchmark: recording unranked timing-boundary control blocks")
            boundary_check = {}
            for name, function in (("baseline", baseline), ("candidate", candidate)):
                measurement = self._measure(torch, function, benchmark["iterations"])
                boundary_check[name] = {
                    **measurement,
                    "cuda_event_not_greater_than_synchronized_host_wall": (
                        measurement["cuda_event_minus_host_wall_ms_per_block"] <= 0.0),
                }
            rows = []
            for round_index in range(benchmark["rounds"]):
                order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
                row: dict[str, Any] = {"round": round_index + 1, "order": list(order)}
                for name in order:
                    measurement = self._measure(
                        torch, baseline if name == "baseline" else candidate, benchmark["iterations"])
                    row[f"{name}_latency_ms"] = measurement["cuda_event_elapsed_ms_per_operation"]
                    for field, value in measurement.items():
                        row[f"{name}_{field}"] = value
                rows.append(row)
        except Exception as error:
            raise StageFailure("benchmark", f"CUDA-event measurement failed: {error}") from error
        baseline_values = [row["baseline_latency_ms"] for row in rows]
        candidate_values = [row["candidate_latency_ms"] for row in rows]
        if not all(math.isfinite(value) and value > 0 for value in baseline_values + candidate_values):
            raise StageFailure("benchmark", "CUDA events returned a nonfinite or non-positive latency")
        baseline_summary, candidate_summary = _summary(baseline_values), _summary(candidate_values)
        return {**partial, "status": "success", "benchmark": {
            "mode": mode,
            "measurement": "steady-state block timing with CUDA events and host clocks",
            "unit": "ms", "units": {
                "cuda_event_elapsed_ms_per_operation": "ms/operation",
                "cuda_event_elapsed_ms_per_block": "ms/block",
                "host_submission_ms_per_block": "ms/block",
                "host_wall_ms_per_block": "ms/block",
                "cuda_event_minus_host_wall_ms_per_block": "ms/block",
            },
            "timing_boundaries": {
                "cuda_event": "device interval from an event before block submission to an event after all block operations; may include host-induced stream idle gaps",
                "host_submission": "host wall interval from before start-event submission through end-event submission, ending before synchronization",
                "host_wall": "host wall interval from before start-event submission through end-event completion",
            },
            "timing_boundary_check": {
                "purpose": "unranked pre-measurement check using the same operations per block",
                "placement": [
                    "host wall start", "CUDA start-event record", "operation submissions",
                    "CUDA end-event record", "host submission end", "end-event synchronize",
                    "host wall end", "CUDA elapsed-time query",
                ],
                "interpretation": (
                    "A positive event-minus-wall value is a concrete cross-timer discrepancy. "
                    "The placement brackets event completion, but the discrepancy alone does "
                    "not identify its cause."),
                "measurements": boundary_check,
            },
            "wrapper_work_per_call": wrapper_work,
            "speedup_definition": "ratio of baseline round-median CUDA-event ms/operation to candidate round-median CUDA-event ms/operation",
            "warmups_per_implementation": benchmark["warmups"],
            "iterations_per_measurement": benchmark["iterations"], "rounds": rows,
            "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
            "speedup_baseline_over_candidate": baseline_summary["median_ms"] / candidate_summary["median_ms"],
        }}


BACKENDS: dict[str, Callable[[], ExperimentBackend]] = {
    "triton_fc1": CudaTritonBackend,
    SOURCE_IMPLEMENTATION: CudaTritonBackend,
}


def _failure(stage: str, error: BaseException, include_traceback: bool = True) -> dict[str, Any]:
    return {"stage": stage, "exception_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc() if include_traceback else None}


def _worker(config: dict[str, Any], factory: Callable[[], ExperimentBackend], output) -> None:
    logs: list[str] = []
    def log(message: str) -> None:
        logs.append(f"{datetime.now(timezone.utc).isoformat()} {message}")
    try:
        payload = factory().run(config, log)
    except StageFailure as error:
        payload = {"status": "failed", "failure": _failure(error.stage, error)}
    except BaseException as error:
        payload = {"status": "failed", "failure": _failure("execution", error)}
    output.put({"payload": payload, "logs": logs})


def _git_metadata(repository: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=repository, text=True,
                              capture_output=True, check=True).stdout.strip()
    try:
        return {"commit": command("rev-parse", "HEAD"),
                "dirty": bool(command("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(raw_config: Mapping[str, Any], *, output_root: Path | str = "artifacts/experiments",
                   backend_factory: Callable[[], ExperimentBackend] | None = None) -> dict[str, Any]:
    """Run one isolated experiment and persist a record for success or failure."""
    created = datetime.now(timezone.utc)
    run_id = f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    run_directory = Path(output_root).resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "created_at_utc": created.isoformat(), "status": "failed",
        "run_directory": str(run_directory),
        "git": _git_metadata(Path(__file__).resolve().parents[2]),
        "environment": {
            "device": {"status": "not_collected"},
            "versions": {"python": sys.version.split()[0]},
        },
    }
    _write_json(run_directory / "submitted_config.json", raw_config)
    logs = [f"{datetime.now(timezone.utc).isoformat()} orchestration: worker started"]
    try:
        config = validate_config(raw_config)
        if config["candidate"]["implementation"] == SOURCE_IMPLEMENTATION:
            config = snapshot_candidate_source(
                config, run_directory / "source" / "candidate.py")
            source_identity = {
                key: config["candidate"]["source"][key]
                for key in ("sha256", "entry_point", "interface_version", "original_path", "path")
            }
            source_identity["implementation"] = SOURCE_IMPLEMENTATION
            source_identity["parameters"] = config["candidate"]["parameters"]
            result["source_identity"] = source_identity
            _write_json(run_directory / "source_identity.json", source_identity)
        _write_json(run_directory / "resolved_config.json", config)
    except BaseException as error:
        result["failure"] = _failure("configuration", error)
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_directory / "result.json", result)
        (run_directory / "stage.log").write_text("configuration validation failed\n", encoding="utf-8")
        return result

    factory = backend_factory or BACKENDS[config["candidate"]["implementation"]]
    context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    messages = context.Queue()
    process = context.Process(target=_worker, args=(config, factory, messages), daemon=False)
    process.start()
    try:
        process.join(float(config["timeout_seconds"]))
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()
            logs.append(f"{datetime.now(timezone.utc).isoformat()} timeout: worker terminated")
            result["failure"] = {
                "stage": "timeout", "exception_type": "TimeoutError",
                "message": f"GPU worker exceeded timeout_seconds={config['timeout_seconds']}",
                "traceback": None,
            }
        else:
            try:
                message = messages.get(timeout=1.0)
            except queue.Empty:
                result["failure"] = {
                    "stage": "execution", "exception_type": "WorkerProcessError",
                    "message": f"worker exited with code {process.exitcode} without returning a result",
                    "traceback": None,
                }
            else:
                logs.extend(message["logs"])
                result.update(message["payload"])
    except KeyboardInterrupt as error:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        result["failure"] = _failure("interruption", error)
    finally:
        messages.close()
        messages.join_thread()

    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["config"] = config
    _write_json(run_directory / "result.json", result)
    (run_directory / "stage.log").write_text("\n".join(logs) + ("\n" if logs else ""), encoding="utf-8")
    benchmark = result.get("benchmark")
    if benchmark:
        _write_json(run_directory / "timings.json", benchmark)
        with (run_directory / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
            measurement_fields = (
                "cuda_event_elapsed_ms_per_operation", "cuda_event_elapsed_ms_per_block",
                "host_submission_ms_per_block", "host_wall_ms_per_block",
                "cuda_event_minus_host_wall_ms_per_block",
            )
            fields = ["benchmark_mode", "round", "order", "baseline_latency_ms", "candidate_latency_ms"]
            fields += [f"{name}_{field}" for name in ("baseline", "candidate")
                       for field in measurement_fields]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in benchmark["rounds"]:
                writer.writerow({**row, "benchmark_mode": config["benchmark"]["mode"],
                                 "order": ">".join(row["order"])})
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a PyTorch/cuBLAS versus Triton FC1 experiment")
    result.add_argument("--config", type=Path, help="JSON configuration (defaults to the documented FC1 experiment)")
    result.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    result.add_argument("--print-default-config", action="store_true")
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return 0
    try:
        config = DEFAULT_CONFIG if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    result = run_experiment(config, output_root=args.output_root)
    print(f"run directory: {result['run_directory']}")
    if result["status"] != "success":
        failure = result["failure"]
        print(f"{failure['stage']} failure: {failure['message']}", file=sys.stderr)
        return 1
    benchmark = result["benchmark"]
    print(f"benchmark mode: {benchmark['mode']}")
    print(f"baseline median: {benchmark['baseline_summary']['median_ms']:.6f} ms")
    print(f"candidate median: {benchmark['candidate_summary']['median_ms']:.6f} ms")
    print(f"speedup (baseline/candidate): {benchmark['speedup_baseline_over_candidate']:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
