"""Real NVIDIA/Triton evaluation for sampled FP16 FC1 GEMM candidates.

PyTorch and Triton are imported only inside :class:`CudaTritonRunner`, keeping
the package and its CPU-only analysis tools free of mandatory GPU dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Protocol
import uuid

from .evaluation import (
    BenchmarkResult, CandidateIdentity, CompilationResult, CorrectnessResult,
    EvaluationIdentity, EvaluationLifecycle, EvaluationMetadata, EvaluationRecord,
    EvaluationStatus, FailureCategory, FailureInfo,
)
from .sampling import SamplingRecord, TritonGemmConfig, candidate_identifier


FC1_SHAPE = (512, 3072, 768)
SUPPORTED_PRECISIONS = frozenset({"fp16", "float16"})
SUPPORTED_HARDWARE_TARGETS = frozenset({"sm_89", "8.9"})


def _nvidia_driver_version(device_index: int) -> str | None:
    """Query the NVIDIA management interface without making it a dependency."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if len(versions) == 1 else None


@dataclass(frozen=True)
class TritonEvaluatorConfig:
    seed: int = 0
    atol: float = 1e-2
    rtol: float = 1e-2
    warmups: int = 100
    iterations_per_sample: int = 500
    samples: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in ("warmups", "iterations_per_sample", "samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TritonRunEvidence:
    device: Mapping[str, Any]
    versions: Mapping[str, Any]
    compilation_setup_host_us: float
    correctness: Mapping[str, Any]
    cuda_event_samples_us: tuple[float, ...]
    host_submission_samples_us: tuple[float, ...]
    host_wall_samples_us: tuple[float, ...]


class TritonRunnerFailure(RuntimeError):
    """A hardware-runner failure with a stable evaluation category."""

    def __init__(
        self,
        category: FailureCategory,
        message: str,
        *,
        versions: Mapping[str, Any] | None = None,
        device: Mapping[str, Any] | None = None,
        compilation_succeeded: bool | None = None,
        correctness: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.versions = dict(versions or {})
        self.device = dict(device or {})
        self.compilation_succeeded = compilation_succeeded
        self.correctness = dict(correctness) if correctness is not None else None
        self.traceback = traceback.format_exc()


class TritonRunner(Protocol):
    def run(
        self, candidate: SamplingRecord, config: TritonEvaluatorConfig,
    ) -> TritonRunEvidence:
        """Compile, validate, and benchmark one compatible candidate."""


def _summary(values: tuple[float, ...]) -> dict[str, float]:
    result = {
        "minimum_us": min(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "maximum_us": max(values),
    }
    if len(values) > 1:
        result["sample_standard_deviation_us"] = statistics.stdev(values)
    return result


class CudaTritonRunner:
    """Run the packaged FC1 kernel with preallocated tensors on CUDA."""

    @staticmethod
    def _measure(torch: Any, function: Callable[[], Any], iterations: int) -> dict[str, float]:
        # This matches the established experiment runner's CUDA-event and host
        # clock placement while reporting per-operation values in microseconds.
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
        event_ms = float(start.elapsed_time(end))
        return {
            "cuda_event_us_per_operation": event_ms * 1000.0 / iterations,
            "host_submission_us_per_operation": (
                (submission_end - wall_start) / 1000.0 / iterations),
            "host_wall_us_per_operation": (wall_end - wall_start) / 1000.0 / iterations,
        }

    def run(self, candidate: SamplingRecord,
            config: TritonEvaluatorConfig) -> TritonRunEvidence:
        try:
            import torch
            import triton
            from . import triton_fc1
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.INFRASTRUCTURE,
                f"PyTorch/Triton prerequisites are unavailable: {error}",
            ) from error

        versions = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() returned false")
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            device = {
                "type": "cuda", "index": device_index, "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
            }
            versions["nvidia_driver"] = _nvidia_driver_version(device_index)
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.INFRASTRUCTURE, f"failed to query CUDA device: {error}",
                versions=versions,
            ) from error
        if device["compute_capability"] != "8.9":
            raise TritonRunnerFailure(
                FailureCategory.INVALID_CANDIDATE,
                "Milestone 2B.1 requires an NVIDIA GPU with compute capability 8.9; "
                f"found {device['compute_capability']} ({device['name']})",
                versions=versions, device=device,
            )

        m, n, k = FC1_SHAPE
        try:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            b = torch.randn((k, n), device="cuda", dtype=torch.float16)
            reference = torch.empty((m, n), device="cuda", dtype=torch.float16)
            output = torch.empty_like(reference)
            torch.matmul(a, b, out=reference)
            torch.cuda.synchronize()
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.INFRASTRUCTURE, f"CUDA input/reference setup failed: {error}",
                versions=versions, device=device,
            ) from error

        parameters = candidate.parameters
        kernel_config = triton_fc1.TritonGemmConfig(
            block_m=parameters.block_m, block_n=parameters.block_n,
            block_k=parameters.block_k, num_warps=parameters.num_warps,
            num_stages=parameters.num_stages, group_size_m=1,
        )
        grid = (triton.cdiv(m, parameters.block_m) * triton.cdiv(n, parameters.block_n),)

        def launch() -> Any:
            triton_fc1.fc1_kernel[grid](
                a, b, output, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_M=kernel_config.block_m, BLOCK_N=kernel_config.block_n,
                BLOCK_K=kernel_config.block_k, GROUP_SIZE_M=kernel_config.group_size_m,
                num_warps=kernel_config.num_warps, num_stages=kernel_config.num_stages,
            )
            return output

        started = time.perf_counter_ns()
        try:
            output.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.COMPILATION, f"Triton setup/compilation invocation failed: {error}",
                versions=versions, device=device, compilation_succeeded=False,
            ) from error
        compilation_setup_host_us = (time.perf_counter_ns() - started) / 1000.0

        try:
            shape_match = tuple(output.shape) == tuple(reference.shape)
            dtype_match = output.dtype == reference.dtype
            reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
            candidate_nonfinite = int((~torch.isfinite(output)).sum().item()) if shape_match else None
            finite = reference_nonfinite == 0 and candidate_nonfinite == 0
            difference = ((reference.float() - output.float()).abs()
                          if shape_match and finite else None)
            correctness = {
                "passed": bool(
                    shape_match and dtype_match and finite
                    and torch.allclose(output, reference, atol=config.atol, rtol=config.rtol)
                ),
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "max_absolute_error": float(difference.max().item()) if difference is not None else None,
                "mean_absolute_error": float(difference.mean().item()) if difference is not None else None,
                "reference_nonfinite_count": reference_nonfinite,
                "candidate_nonfinite_count": candidate_nonfinite,
                "reference": "torch.matmul with preallocated FP16 output (PyTorch/cuBLAS)",
                "atol": config.atol,
                "rtol": config.rtol,
                "candidate_accumulation": "FP32 Triton tl.dot accumulator with FP16 output",
            }
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.INFRASTRUCTURE,
                f"correctness comparison infrastructure failed: {error}", versions=versions,
                device=device, compilation_succeeded=True,
            ) from error
        if not correctness["passed"]:
            raise TritonRunnerFailure(
                FailureCategory.CORRECTNESS,
                "candidate output did not satisfy shape, finiteness, and FP16 tolerance requirements",
                versions=versions, device=device, compilation_succeeded=True,
                correctness=correctness,
            )

        try:
            for _ in range(config.warmups):
                launch()
            torch.cuda.synchronize()
            measurements = tuple(
                self._measure(torch, launch, config.iterations_per_sample)
                for _ in range(config.samples)
            )
            cuda_samples = tuple(item["cuda_event_us_per_operation"] for item in measurements)
            submission_samples = tuple(
                item["host_submission_us_per_operation"] for item in measurements)
            wall_samples = tuple(item["host_wall_us_per_operation"] for item in measurements)
            if not all(math.isfinite(value) and value > 0 for value in (
                    cuda_samples + submission_samples + wall_samples)):
                raise RuntimeError("timers returned a nonfinite or non-positive latency")
        except Exception as error:
            raise TritonRunnerFailure(
                FailureCategory.BENCHMARK, f"CUDA-event benchmark failed: {error}",
                versions=versions, device=device, compilation_succeeded=True,
                correctness=correctness,
            ) from error
        return TritonRunEvidence(
            device=device, versions=versions,
            compilation_setup_host_us=compilation_setup_host_us,
            correctness=correctness, cuda_event_samples_us=cuda_samples,
            host_submission_samples_us=submission_samples,
            host_wall_samples_us=wall_samples,
        )


def validate_triton_candidate(candidate: SamplingRecord) -> None:
    """Validate identity and Milestone 2B FC1 compatibility without GPU imports."""
    if not isinstance(candidate, SamplingRecord):
        raise ValueError("candidate must be a SamplingRecord")
    reconstructed = TritonGemmConfig.from_dict(candidate.parameters.to_dict())
    if candidate.candidate_id != candidate_identifier(reconstructed, candidate.context):
        raise ValueError("candidate identity does not match its parameters and context")
    context = candidate.context
    if context.backend != "triton":
        raise ValueError(f"unsupported backend {context.backend!r}; expected 'triton'")
    if context.precision.lower() not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision {context.precision!r}; expected FP16")
    if (context.m, context.n, context.k) != FC1_SHAPE:
        raise ValueError(
            f"unsupported workload {(context.m, context.n, context.k)}; expected {FC1_SHAPE}"
        )
    if context.hardware_target.lower() not in SUPPORTED_HARDWARE_TARGETS:
        raise ValueError(
            f"unsupported hardware target {context.hardware_target!r}; expected SM 8.9"
        )
    if context.k % reconstructed.block_k:
        raise ValueError(f"K={context.k} must be divisible by block_k={reconstructed.block_k}")


class TritonGemmEvaluator:
    """Evaluate sampled FC1 configurations with the packaged Triton kernel."""

    def __init__(
        self,
        config: TritonEvaluatorConfig = TritonEvaluatorConfig(),
        *,
        runner: TritonRunner | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._runner = runner or CudaTritonRunner()
        self._id_factory = id_factory or (lambda: f"evaluation-{uuid.uuid4().hex}")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _metadata(self, candidate: SamplingRecord,
                  software_version: str | None = None) -> EvaluationMetadata:
        context = candidate.context
        return EvaluationMetadata(
            hardware_target=context.hardware_target, backend=context.backend,
            workload_shape=(context.m, context.n, context.k), precision=context.precision,
            timestamp=self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            software_version=software_version,
        )

    @staticmethod
    def _validate_candidate(candidate: SamplingRecord) -> None:
        validate_triton_candidate(candidate)

    def evaluate(self, candidate: SamplingRecord) -> EvaluationRecord:
        identity = EvaluationIdentity(self._id_factory())
        lifecycle = EvaluationLifecycle()
        try:
            validate_triton_candidate(candidate)
        except (TypeError, ValueError) as error:
            lifecycle.transition(EvaluationStatus.FAILED)
            context_candidate = candidate if isinstance(candidate, SamplingRecord) else None
            metadata = (self._metadata(context_candidate) if context_candidate else
                        EvaluationMetadata("unknown", "unknown", FC1_SHAPE, "unknown",
                                           self._clock().astimezone(timezone.utc).isoformat()
                                           .replace("+00:00", "Z")))
            candidate_id = (candidate.candidate_id if context_candidate else "unknown-candidate")
            return EvaluationRecord(
                evaluation_id=identity, candidate_id=CandidateIdentity(candidate_id),
                status=lifecycle.status, lifecycle=lifecycle.history, metadata=metadata,
                failure=FailureInfo(FailureCategory.INVALID_CANDIDATE, str(error)),
                extensions={"candidate_parameters": (
                    candidate.parameters.to_dict() if context_candidate else None)},
            )

        try:
            evidence = self._runner.run(candidate, self.config)
        except TritonRunnerFailure as error:
            return self._failure_record(identity, candidate, error)
        except Exception as error:
            wrapped = TritonRunnerFailure(
                FailureCategory.INFRASTRUCTURE,
                f"unexpected evaluator infrastructure error: {error}",
            )
            wrapped.traceback = traceback.format_exc()
            return self._failure_record(identity, candidate, wrapped)

        for status in (EvaluationStatus.COMPILATION, EvaluationStatus.CORRECTNESS_VALIDATION,
                       EvaluationStatus.BENCHMARKING, EvaluationStatus.COMPLETED):
            lifecycle.transition(status)
        versions = evidence.versions
        compiler_version = str(versions.get("triton")) if versions.get("triton") else None
        cuda_summary = _summary(evidence.cuda_event_samples_us)
        extensions = self._extensions(candidate, evidence)
        return EvaluationRecord(
            evaluation_id=identity, candidate_id=CandidateIdentity(candidate.candidate_id),
            status=lifecycle.status, lifecycle=lifecycle.history,
            metadata=self._metadata(candidate, _software_version(versions)),
            compilation=CompilationResult(True, compiler_version,
                                          "packaged Triton FC1 kernel compiled and launched"),
            correctness=CorrectnessResult(True, "passed FP16 comparison against PyTorch reference"),
            benchmark=BenchmarkResult(
                latency=cuda_summary["median_us"],
                measurement_type="median CUDA-event kernel latency",
                unit="us", repetitions=self.config.samples * self.config.iterations_per_sample,
                synthetic=False,
            ),
            extensions=extensions,
        )
    def _failure_record(self, identity: EvaluationIdentity, candidate: SamplingRecord,
                        error: TritonRunnerFailure) -> EvaluationRecord:
        lifecycle = EvaluationLifecycle()
        compilation = None
        correctness = None
        if error.category is FailureCategory.COMPILATION:
            lifecycle.transition(EvaluationStatus.COMPILATION)
            compilation = CompilationResult(False, _optional_string(error.versions.get("triton")),
                                            str(error))
        elif error.category is FailureCategory.CORRECTNESS:
            lifecycle.transition(EvaluationStatus.COMPILATION)
            lifecycle.transition(EvaluationStatus.CORRECTNESS_VALIDATION)
            compilation = CompilationResult(True, _optional_string(error.versions.get("triton")))
            correctness = CorrectnessResult(False, str(error))
        elif error.category is FailureCategory.BENCHMARK:
            lifecycle.transition(EvaluationStatus.COMPILATION)
            lifecycle.transition(EvaluationStatus.CORRECTNESS_VALIDATION)
            lifecycle.transition(EvaluationStatus.BENCHMARKING)
            compilation = CompilationResult(True, _optional_string(error.versions.get("triton")))
            correctness = CorrectnessResult(True, "correctness passed before benchmark failure")
        lifecycle.transition(EvaluationStatus.FAILED)
        extensions: dict[str, Any] = {
            "candidate_parameters": candidate.parameters.to_dict(),
            "environment": {"device": error.device or None, "versions": error.versions or None},
            "failure_traceback": error.traceback,
        }
        if error.correctness is not None:
            extensions["correctness_evidence"] = error.correctness
        return EvaluationRecord(
            evaluation_id=identity, candidate_id=CandidateIdentity(candidate.candidate_id),
            status=lifecycle.status, lifecycle=lifecycle.history,
            metadata=self._metadata(candidate, _software_version(error.versions)),
            compilation=compilation, correctness=correctness,
            failure=FailureInfo(error.category, str(error)), extensions=extensions,
        )

    def _extensions(self, candidate: SamplingRecord,
                    evidence: TritonRunEvidence) -> dict[str, Any]:
        return {
            "measurement_provenance": "real_hardware",
            "candidate_parameters": candidate.parameters.to_dict(),
            "kernel": {"implementation": "packaged triton_fc1.fc1_kernel", "group_size_m": 1},
            "environment": {"device": dict(evidence.device), "versions": dict(evidence.versions)},
            "compilation": {
                "first_synchronized_invocation_host_us": evidence.compilation_setup_host_us,
                "included_in_benchmark": False,
            },
            "correctness_evidence": dict(evidence.correctness),
            "benchmark_distribution": {
                "statistic_reported_as_latency": "median of CUDA-event samples",
                "cuda_event_samples_us": list(evidence.cuda_event_samples_us),
                "cuda_event_summary": _summary(evidence.cuda_event_samples_us),
                "host_submission_samples_us_per_operation": list(
                    evidence.host_submission_samples_us),
                "host_submission_summary": _summary(evidence.host_submission_samples_us),
                "host_wall_samples_us_per_operation": list(evidence.host_wall_samples_us),
                "host_wall_summary": _summary(evidence.host_wall_samples_us),
                "warmup_launches": self.config.warmups,
                "iterations_per_sample": self.config.iterations_per_sample,
                "sample_count": self.config.samples,
                "total_timed_launches": self.config.samples * self.config.iterations_per_sample,
                "timing_method": (
                    "CUDA events bracket each block of launches; end event synchronized before "
                    "elapsed-time query; host submission and synchronized host-wall clocks recorded "
                    "separately; all values normalized per launch"
                ),
            },
        }


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _software_version(versions: Mapping[str, Any]) -> str | None:
    fields = [f"{name}={versions[name]}"
              for name in ("torch", "triton", "cuda_runtime", "nvidia_driver")
              if versions.get(name) is not None]
    return "; ".join(fields) if fields else None
