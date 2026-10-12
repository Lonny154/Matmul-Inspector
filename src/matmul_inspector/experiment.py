"""Reproducible baseline-versus-candidate experiment orchestration.

The orchestration and result schema deliberately have no CUDA dependency.  The
CUDA/Triton implementation is imported lazily by :class:`CudaTritonBackend`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
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


SCHEMA_VERSION = 1
DEFAULT_CONFIG: dict[str, Any] = {
    "workload": {"m": 512, "k": 768, "n": 3072, "dtype": "float16"},
    "seed": 0,
    "correctness": {"atol": 1e-2, "rtol": 1e-2},
    "benchmark": {"warmups": 100, "iterations": 500, "rounds": 10},
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
    for field in ("warmups", "iterations", "rounds"):
        _positive_int(benchmark.get(field), f"benchmark.{field}")
    timeout = config["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive number")

    baseline, candidate = config["baseline"], config["candidate"]
    if not isinstance(baseline, dict) or baseline.get("implementation") != "pytorch_matmul":
        raise ValueError("baseline.implementation must be 'pytorch_matmul'")
    if not isinstance(candidate, dict) or candidate.get("implementation") != "triton_fc1":
        raise ValueError("candidate.implementation must be 'triton_fc1'")
    parameters = candidate.get("parameters")
    expected = {"block_m", "block_n", "block_k", "num_warps", "num_stages", "group_size_m"}
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("candidate.parameters must contain exactly: " + ", ".join(sorted(expected)))
    for field in expected:
        _positive_int(parameters[field], f"candidate.parameters.{field}")
    if workload["k"] % parameters["block_k"]:
        raise ValueError("workload.k must be divisible by candidate.parameters.block_k")
    return config


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
    """Adapter for PyTorch/cuBLAS and the repository's established FC1 kernel."""

    def _imports(self):
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
        kernel_path = Path(__file__).resolve().parents[2] / "scripts" / "triton_fc1.py"
        if not kernel_path.is_file():
            raise StageFailure("prerequisite", f"repository kernel is missing: {kernel_path}")
        spec = importlib.util.spec_from_file_location("matmul_inspector_triton_fc1", kernel_path)
        if spec is None or spec.loader is None:
            raise StageFailure("prerequisite", f"cannot load Triton kernel: {kernel_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise StageFailure("prerequisite", f"cannot import Triton kernel: {error}") from error
        return torch, module

    @staticmethod
    def _measure(torch, function: Callable[[], Any], iterations: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) / iterations

    def run(self, config: Mapping[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
        log("prerequisite: checking PyTorch, Triton, and CUDA")
        torch, kernel = self._imports()
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

        parameters = config["candidate"]["parameters"]
        triton_config = kernel.TritonGemmConfig(**parameters)
        baseline = lambda: torch.matmul(a, b)
        candidate = lambda: kernel.triton_gemm(a, b, triton_config)
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

        log("correctness: comparing candidate with PyTorch/cuBLAS reference")
        try:
            reference, actual = outputs["baseline"], outputs["candidate"]
            shape_match = tuple(reference.shape) == tuple(actual.shape)
            reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
            candidate_nonfinite = int((~torch.isfinite(actual)).sum().item()) if shape_match else None
            if shape_match:
                tolerances = config["correctness"]
                passed = bool(reference_nonfinite == 0 and candidate_nonfinite == 0 and torch.allclose(
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
            "passed": passed, "shape_match": shape_match,
            "reference_shape": list(reference.shape), "candidate_shape": list(actual.shape),
            "max_absolute_error": max_error, "mean_absolute_error": mean_error,
            "reference_nonfinite_count": reference_nonfinite,
            "candidate_nonfinite_count": candidate_nonfinite,
            "atol": config["correctness"]["atol"], "rtol": config["correctness"]["rtol"],
            "reference": "pytorch_matmul (cuBLAS)",
            "candidate_accumulation": "FP32 Triton tl.dot accumulator with FP16 output",
            "reference_accumulation": "PyTorch/cuBLAS implementation-selected FP16 GEMM accumulation",
        }
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

        benchmark = config["benchmark"]
        log(f"benchmark: warming each implementation {benchmark['warmups']} times")
        try:
            for function in (baseline, candidate):
                for _ in range(benchmark["warmups"]):
                    function()
                torch.cuda.synchronize()
            rows = []
            for round_index in range(benchmark["rounds"]):
                order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
                row: dict[str, Any] = {"round": round_index + 1, "order": list(order)}
                for name in order:
                    row[f"{name}_latency_ms"] = self._measure(
                        torch, baseline if name == "baseline" else candidate, benchmark["iterations"])
                rows.append(row)
        except Exception as error:
            raise StageFailure("benchmark", f"CUDA-event measurement failed: {error}") from error
        baseline_values = [row["baseline_latency_ms"] for row in rows]
        candidate_values = [row["candidate_latency_ms"] for row in rows]
        if not all(math.isfinite(value) and value > 0 for value in baseline_values + candidate_values):
            raise StageFailure("benchmark", "CUDA events returned a nonfinite or non-positive latency")
        baseline_summary, candidate_summary = _summary(baseline_values), _summary(candidate_values)
        return {**partial, "status": "success", "benchmark": {
            "measurement": "mean steady-state device elapsed time per GEMM from CUDA events",
            "unit": "ms", "warmups_per_implementation": benchmark["warmups"],
            "iterations_per_measurement": benchmark["iterations"], "rounds": rows,
            "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
            "speedup_baseline_over_candidate": baseline_summary["median_ms"] / candidate_summary["median_ms"],
        }}


BACKENDS: dict[str, Callable[[], ExperimentBackend]] = {"triton_fc1": CudaTritonBackend}


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
            writer = csv.DictWriter(stream, fieldnames=("round", "order", "baseline_latency_ms", "candidate_latency_ms"))
            writer.writeheader()
            for row in benchmark["rounds"]:
                writer.writerow({**row, "order": ">".join(row["order"])})
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
    print(f"baseline median: {benchmark['baseline_summary']['median_ms']:.6f} ms")
    print(f"candidate median: {benchmark['candidate_summary']['median_ms']:.6f} ms")
    print(f"speedup (baseline/candidate): {benchmark['speedup_baseline_over_candidate']:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
