"""Controlled, paired FC1 baseline-versus-candidate benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import uuid

from .evaluation import (
    BenchmarkResult, CandidateIdentity, CompilationResult, CorrectnessResult,
    EvaluationIdentity, EvaluationMetadata, EvaluationRecord, EvaluationStatus,
    FailureCategory, FailureInfo, write_evaluation_jsonl,
)
from .sampling import SamplingRecord
from .triton_evaluator import FC1_SHAPE, validate_triton_candidate


CONTROLLED_BENCHMARK_SCHEMA_VERSION = 1
BASELINE_ID = "pytorch_cublas"


@dataclass(frozen=True)
class ControlledBenchmarkConfig:
    scheduling_seed: int = 42
    input_seed: int = 0
    rounds: int = 10
    samples_per_round: int = 3
    launches_per_sample: int = 100
    warmups: int = 100
    atol: float = 1e-2
    rtol: float = 1e-2

    def __post_init__(self) -> None:
        for name in ("scheduling_seed", "input_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("rounds", "samples_per_round", "launches_per_sample", "warmups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class TimingObservation:
    cuda_event_us: float
    host_submission_us: float
    host_wall_us: float

    def __post_init__(self) -> None:
        for name in ("cuda_event_us", "host_submission_us", "host_wall_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class PreparedCandidate:
    sampled: SamplingRecord
    evaluation_id: str
    compilation: CompilationResult
    correctness: CorrectnessResult
    correctness_evidence: Mapping[str, Any]
    setup_extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkPreparation:
    environment: Mapping[str, Any]
    baseline_description: Mapping[str, Any]
    eligible: tuple[PreparedCandidate, ...]
    failed_evaluations: tuple[EvaluationRecord, ...] = ()


class ControlledBenchmarkBackend(Protocol):
    def prepare(self, candidates: Sequence[SamplingRecord],
                config: ControlledBenchmarkConfig) -> BenchmarkPreparation: ...
    def warmup(self, implementation_ids: Sequence[str], launches: int) -> None: ...
    def measure(self, implementation_id: str, launches: int) -> TimingObservation: ...


def build_round_schedule(candidate_ids: Sequence[str], rounds: int,
                         seed: int) -> list[list[str]]:
    """Return deterministic independently shuffled orders including the baseline."""
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    identifiers = list(candidate_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    if BASELINE_ID in identifiers:
        raise ValueError(f"candidate ID {BASELINE_ID!r} is reserved")
    randomizer = random.Random(seed)
    schedule = []
    for _ in range(rounds):
        order = [BASELINE_ID, *identifiers]
        randomizer.shuffle(order)
        schedule.append(order)
    return schedule


def latency_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("latency measurements must not be empty")
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and value > 0 for value in normalized):
        raise ValueError("latency measurements must be finite positive numbers")
    return {
        "median_us": statistics.median(normalized),
        "mean_us": statistics.fmean(normalized),
        "minimum_us": min(normalized),
        "maximum_us": max(normalized),
        "sample_standard_deviation_us": (
            statistics.stdev(normalized) if len(normalized) > 1 else 0.0),
        "observation_count": len(normalized),
    }


def comparison_statistics(baseline_round_medians: Sequence[float],
                          candidate_round_medians: Sequence[float]) -> dict[str, Any]:
    if len(baseline_round_medians) != len(candidate_round_medians) or not baseline_round_medians:
        raise ValueError("paired baseline and candidate rounds must have equal non-zero length")
    baseline = latency_statistics(baseline_round_medians)
    candidate = latency_statistics(candidate_round_medians)
    paired_speedups = [base / current
                       for base, current in zip(baseline_round_medians, candidate_round_medians)]
    speedup = baseline["median_us"] / candidate["median_us"]
    return {
        "speedup_baseline_over_candidate": speedup,
        "latency_reduction_percent": (1.0 - candidate["median_us"] / baseline["median_us"]) * 100.0,
        "paired_round_speedups": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedup_mean": statistics.fmean(paired_speedups),
        "paired_speedup_sample_standard_deviation": (
            statistics.stdev(paired_speedups) if len(paired_speedups) > 1 else 0.0),
        "paired_round_count": len(paired_speedups),
    }


class ControlledBenchmarkRunner:
    def __init__(self, backend: ControlledBenchmarkBackend, *,
                 id_factory: Callable[[], str] | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.backend = backend
        self._id_factory = id_factory or (lambda: f"controlled-{uuid.uuid4().hex}")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def run(self, candidates: Sequence[SamplingRecord],
            config: ControlledBenchmarkConfig = ControlledBenchmarkConfig()) -> dict[str, Any]:
        if not candidates:
            raise ValueError("controlled benchmark requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("controlled benchmark candidate IDs must be unique")
        preparation = self.backend.prepare(candidates, config)
        eligible_ids = [item.sampled.candidate_id for item in preparation.eligible]
        schedule = build_round_schedule(eligible_ids, config.rounds, config.scheduling_seed)
        measurements: dict[str, list[list[TimingObservation]]] = {
            identifier: [[] for _ in range(config.rounds)]
            for identifier in (BASELINE_ID, *eligible_ids)
        }
        failures: list[dict[str, Any]] = []
        active = set(eligible_ids)
        self.backend.warmup((BASELINE_ID, *eligible_ids), config.warmups)
        executed_orders: list[list[str]] = []
        for round_index, planned_order in enumerate(schedule):
            executed = []
            for identifier in planned_order:
                if identifier != BASELINE_ID and identifier not in active:
                    continue
                executed.append(identifier)
                try:
                    for _ in range(config.samples_per_round):
                        measurements[identifier][round_index].append(
                            self.backend.measure(identifier, config.launches_per_sample))
                except Exception as error:
                    failures.append({
                        "implementation_id": identifier, "round": round_index + 1,
                        "category": (FailureCategory.INFRASTRUCTURE.value
                                     if identifier == BASELINE_ID else FailureCategory.BENCHMARK.value),
                        "description": str(error),
                    })
                    if identifier == BASELINE_ID:
                        raise RuntimeError(f"baseline benchmark failed: {error}") from error
                    active.remove(identifier)
            executed_orders.append(executed)

        summaries = _summarize_measurements(measurements, eligible_ids, active)
        evaluations = list(preparation.failed_evaluations)
        prepared_by_id = {item.sampled.candidate_id: item for item in preparation.eligible}
        failure_by_id = {item["implementation_id"]: item for item in failures}
        for identifier in eligible_ids:
            prepared = prepared_by_id[identifier]
            if identifier in active:
                evaluations.append(_completed_evaluation(
                    prepared, summaries[identifier], preparation.environment, config,
                    self._clock(),
                ))
            else:
                evaluations.append(_benchmark_failure_evaluation(
                    prepared, failure_by_id[identifier], preparation.environment, self._clock(),
                ))
        experiment_id = self._id_factory()
        manifest = {
            "schema_version": CONTROLLED_BENCHMARK_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "created_at": self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "workload": {"m": FC1_SHAPE[0], "n": FC1_SHAPE[1], "k": FC1_SHAPE[2],
                         "precision": "fp16"},
            "hardware_target": "sm_89",
            "environment": dict(preparation.environment),
            "config": {
                "scheduling_seed": config.scheduling_seed, "input_seed": config.input_seed,
                "rounds": config.rounds, "samples_per_round": config.samples_per_round,
                "launches_per_sample": config.launches_per_sample, "warmups": config.warmups,
                "atol": config.atol, "rtol": config.rtol,
            },
            "sampling": {
                "seeds": sorted({candidate.seed for candidate in candidates}),
                "search_space_ids": sorted({candidate.search_space_id for candidate in candidates}),
                "samplers": sorted({candidate.sampler for candidate in candidates}),
            },
            "candidate_ids": candidate_ids,
            "candidates": [
                {"candidate_id": candidate.candidate_id,
                 "parameters": candidate.parameters.to_dict(),
                 "sampling_order": candidate.sampling_order}
                for candidate in candidates
            ],
            "evaluation_ids": [item.evaluation_id.value for item in evaluations],
            "evaluation_records": [
                {"candidate_id": item.candidate_id.value,
                 "evaluation_id": item.evaluation_id.value,
                 "status": item.status.value}
                for item in evaluations
            ],
            "baseline": {**dict(preparation.baseline_description), **summaries[BASELINE_ID]},
            "round_execution_order": executed_orders,
            "implementations": summaries,
            "failures": [
                {"candidate_id": item.candidate_id.value,
                 "category": item.failure.category.value,
                 "description": item.failure.description}
                for item in evaluations if item.failure is not None
            ],
            "artifacts": {
                "evaluations_jsonl": "evaluations.jsonl",
                "manifest_json": "experiment.json",
                "report_markdown": "report.md",
            },
            "extensions": {},
        }
        validate_manifest(manifest)
        return {"manifest": manifest, "evaluations": evaluations}


def _observation_dict(value: TimingObservation) -> dict[str, float]:
    return {"cuda_event_us": value.cuda_event_us,
            "host_submission_us": value.host_submission_us,
            "host_wall_us": value.host_wall_us}


def _summarize_measurements(measurements: Mapping[str, list[list[TimingObservation]]],
                            eligible_ids: Sequence[str], active: set[str]) -> dict[str, Any]:
    result = {}
    for identifier, rounds in measurements.items():
        raw_rounds = [[_observation_dict(item) for item in current] for current in rounds]
        cuda_rounds = [[item.cuda_event_us for item in current] for current in rounds]
        nonempty = [values for values in cuda_rounds if values]
        flat = [value for values in nonempty for value in values]
        summary = latency_statistics(flat) if flat else None
        round_medians = [statistics.median(values) if values else None for values in cuda_rounds]
        result[identifier] = {
            "implementation": ("PyTorch torch.matmul(..., out=...) / cuBLAS"
                               if identifier == BASELINE_ID else "Triton fc1_kernel"),
            "status": ("completed" if identifier == BASELINE_ID or identifier in active else "failed"),
            "raw_rounds": raw_rounds,
            "per_round_median_us": round_medians,
            "summary": summary,
            "comparison_to_baseline": None,
        }
    baseline_medians = result[BASELINE_ID]["per_round_median_us"]
    for identifier in eligible_ids:
        candidate_medians = result[identifier]["per_round_median_us"]
        paired = [(base, candidate) for base, candidate in zip(baseline_medians, candidate_medians)
                  if base is not None and candidate is not None]
        if identifier in active and len(paired) == len(baseline_medians):
            result[identifier]["comparison_to_baseline"] = comparison_statistics(
                [item[0] for item in paired], [item[1] for item in paired])
    return result


def _metadata(prepared: PreparedCandidate, environment: Mapping[str, Any],
              timestamp: datetime) -> EvaluationMetadata:
    versions = environment.get("versions") or {}
    version = "; ".join(f"{name}={versions[name]}" for name in
                        ("torch", "triton", "cuda_runtime") if versions.get(name)) or None
    context = prepared.sampled.context
    return EvaluationMetadata(
        context.hardware_target, context.backend, (context.m, context.n, context.k),
        context.precision, timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        version,
    )


def _completed_evaluation(prepared: PreparedCandidate, summary: Mapping[str, Any],
                          environment: Mapping[str, Any], config: ControlledBenchmarkConfig,
                          timestamp: datetime) -> EvaluationRecord:
    distribution = summary["summary"]
    return EvaluationRecord(
        evaluation_id=EvaluationIdentity(prepared.evaluation_id),
        candidate_id=CandidateIdentity(prepared.sampled.candidate_id),
        status=EvaluationStatus.COMPLETED,
        lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                   EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.BENCHMARKING,
                   EvaluationStatus.COMPLETED),
        metadata=_metadata(prepared, environment, timestamp), compilation=prepared.compilation,
        correctness=prepared.correctness,
        benchmark=BenchmarkResult(
            distribution["median_us"], "interleaved median CUDA-event kernel latency", "us",
            config.rounds * config.samples_per_round * config.launches_per_sample, False,
        ),
        extensions={**dict(prepared.setup_extensions),
                    "correctness_evidence": dict(prepared.correctness_evidence),
                    "controlled_benchmark": {
                        "summary": distribution,
                        "per_round_median_us": summary["per_round_median_us"],
                        "comparison_to_baseline": summary["comparison_to_baseline"],
                    }},
    )


def _benchmark_failure_evaluation(prepared: PreparedCandidate, failure: Mapping[str, Any],
                                  environment: Mapping[str, Any],
                                  timestamp: datetime) -> EvaluationRecord:
    return EvaluationRecord(
        evaluation_id=EvaluationIdentity(prepared.evaluation_id),
        candidate_id=CandidateIdentity(prepared.sampled.candidate_id),
        status=EvaluationStatus.FAILED,
        lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                   EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.BENCHMARKING,
                   EvaluationStatus.FAILED),
        metadata=_metadata(prepared, environment, timestamp), compilation=prepared.compilation,
        correctness=prepared.correctness,
        failure=FailureInfo(FailureCategory.BENCHMARK, failure["description"]),
        extensions={**dict(prepared.setup_extensions),
                    "correctness_evidence": dict(prepared.correctness_evidence),
                    "benchmark_failure": dict(failure)},
    )


_MANIFEST_FIELDS = {
    "schema_version", "experiment_id", "created_at", "workload", "hardware_target",
    "environment", "config", "sampling", "candidate_ids", "candidates",
    "evaluation_ids", "evaluation_records", "baseline",
    "round_execution_order", "implementations", "failures", "artifacts", "extensions",
}


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("controlled benchmark manifest must be an object")
    if value.get("schema_version") != CONTROLLED_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"unsupported controlled benchmark schema: {value.get('schema_version')!r}")
    missing = sorted(_MANIFEST_FIELDS - set(value))
    unknown = sorted(set(value) - _MANIFEST_FIELDS)
    if missing:
        raise ValueError(f"controlled benchmark manifest missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown controlled benchmark manifest fields: {', '.join(unknown)}")
    if not isinstance(value["extensions"], Mapping):
        raise ValueError("controlled benchmark extensions must be an object")
    return dict(value)


def write_controlled_benchmark(directory: str | Path, result: Mapping[str, Any]) -> Path:
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        manifest = validate_manifest(result["manifest"])
        evaluations = result["evaluations"]
        _atomic_text(temporary / "experiment.json",
                     json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        write_evaluation_jsonl(temporary / "evaluations.jsonl", evaluations)
        _atomic_text(temporary / "report.md", render_markdown_report(manifest))
        temporary.replace(destination)
    except BaseException:
        for path in temporary.iterdir() if temporary.exists() else ():
            path.unlink()
        if temporary.exists():
            temporary.rmdir()
        raise
    return destination


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_controlled_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read controlled benchmark manifest: {error}") from error
    return validate_manifest(value)


def render_markdown_report(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    workload = manifest["workload"]
    device = (manifest["environment"].get("device") or {})
    lines = [
        f"# Controlled FC1 benchmark `{manifest['experiment_id']}`", "",
        (f"Workload: M={workload['m']}, N={workload['n']}, K={workload['k']}, "
         f"precision={workload['precision']}. Target: {manifest['hardware_target']}. "
         f"Detected GPU: {device.get('name', 'unknown')}."), "",
        "| Implementation | Median latency (us) | Speedup vs baseline | Correctness |",
        "|---|---:|---:|---|",
    ]
    baseline = manifest["implementations"][BASELINE_ID]
    baseline_median = baseline["summary"]["median_us"]
    lines.append(f"| PyTorch/cuBLAS | {baseline_median:.3f} | 1.000× | Reference |")
    measured = []
    for identifier in manifest["candidate_ids"]:
        implementation = manifest["implementations"].get(identifier)
        if not implementation or implementation["status"] != "completed":
            lines.append(f"| `{identifier}` | — | — | FAIL |")
            continue
        median = implementation["summary"]["median_us"]
        speedup = implementation["comparison_to_baseline"]["speedup_baseline_over_candidate"]
        lines.append(f"| `{identifier}` | {median:.3f} | {speedup:.3f}× | PASS |")
        measured.append((median, identifier))
    lines.extend(["", "## Method", "",
                  (f"{manifest['config']['rounds']} independently shuffled rounds; "
                   f"{manifest['config']['samples_per_round']} samples per implementation per "
                   f"round; {manifest['config']['launches_per_sample']} launches per sample. "
                   "CUDA events provide device elapsed time. Host submission and synchronized "
                   "host-wall clocks remain separate in the raw manifest."), ""])
    if measured:
        median, identifier = min(measured)
        lines.extend(["## Fastest measured candidate", "",
                      f"`{identifier}` at {median:.3f} us median CUDA-event latency.", "",
                      "This ranking covers only successfully measured candidates in this experiment. "
                      "It is not a global-optimum claim.", ""])
    lines.extend(["## Variability and limitations", "",
                  "The JSON manifest contains raw observations, per-round medians, sample standard "
                  "deviations, and paired round speedups. Small differences are descriptive and are "
                  "not claimed to be statistically meaningful. GPU clocks, thermals, background "
                  "activity, and implementation-path differences remain possible confounders.", ""])
    if manifest["failures"]:
        lines.extend(["## Failures", ""])
        for failure in manifest["failures"]:
            lines.append(f"- `{failure['candidate_id']}`: {failure['category']}: "
                         f"{failure['description']}")
        lines.append("")
    lines.extend(["## Artifacts", "",
                  "- Raw manifest: `experiment.json`",
                  "- Candidate evaluations: `evaluations.jsonl`", ""])
    return "\n".join(lines)


class CudaControlledBenchmarkBackend:
    """Use shared CUDA inputs and preallocated outputs for paired FC1 timing."""

    def __init__(self, *, id_factory: Callable[[], str] | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self._id_factory = id_factory or (lambda: f"evaluation-{uuid.uuid4().hex}")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._torch: Any = None
        self._launches: dict[str, Callable[[], Any]] = {}

    def prepare(self, candidates: Sequence[SamplingRecord],
                config: ControlledBenchmarkConfig) -> BenchmarkPreparation:
        try:
            import torch
            import triton
            from . import triton_fc1
        except Exception as error:
            raise RuntimeError(f"PyTorch/Triton prerequisites are unavailable: {error}") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        if (properties.major, properties.minor) != (8, 9):
            raise RuntimeError(f"SM 8.9 is required; found {properties.major}.{properties.minor}")
        environment = {
            "device": {"type": "cuda", "index": device_index, "name": properties.name,
                       "compute_capability": "8.9", "total_memory_bytes": properties.total_memory},
            "versions": {"torch": torch.__version__, "triton": triton.__version__,
                         "cuda_runtime": torch.version.cuda},
        }
        m, n, k = FC1_SHAPE
        torch.manual_seed(config.input_seed)
        torch.cuda.manual_seed_all(config.input_seed)
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        baseline_output = torch.empty((m, n), device="cuda", dtype=torch.float16)

        def baseline() -> Any:
            return torch.matmul(a, b, out=baseline_output)

        baseline()
        torch.cuda.synchronize()
        self._torch = torch
        self._launches = {BASELINE_ID: baseline}
        eligible = []
        failures = []
        for sampled in candidates:
            validation_error = None
            try:
                validate_triton_candidate(sampled)
            except ValueError as error:
                validation_error = str(error)
            if validation_error:
                failures.append(_preparation_failure(sampled, FailureCategory.INVALID_CANDIDATE,
                                                     validation_error, environment, self._id_factory(),
                                                     self._clock()))
                continue
            parameters = sampled.parameters
            output = torch.empty_like(baseline_output)
            grid = (triton.cdiv(m, parameters.block_m) * triton.cdiv(n, parameters.block_n),)

            def launch(output=output, parameters=parameters, grid=grid):
                triton_fc1.fc1_kernel[grid](
                    a, b, output, m, n, k,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    output.stride(0), output.stride(1),
                    BLOCK_M=parameters.block_m, BLOCK_N=parameters.block_n,
                    BLOCK_K=parameters.block_k, GROUP_SIZE_M=1,
                    num_warps=parameters.num_warps, num_stages=parameters.num_stages,
                )
                return output
            started = time.perf_counter_ns()
            try:
                output.fill_(float("nan"))
                launch()
                torch.cuda.synchronize()
            except Exception as error:
                failures.append(_preparation_failure(
                    sampled, FailureCategory.COMPILATION,
                    f"Triton setup/compilation invocation failed: {error}", environment,
                    self._id_factory(), self._clock(), compilation_succeeded=False,
                ))
                continue
            setup_us = (time.perf_counter_ns() - started) / 1000.0
            reference_finite = bool(torch.isfinite(baseline_output).all().item())
            candidate_finite = bool(torch.isfinite(output).all().item())
            finite = reference_finite and candidate_finite
            difference = ((baseline_output.float() - output.float()).abs() if finite else None)
            correctness_evidence = {
                "passed": bool(finite
                               and torch.allclose(output, baseline_output,
                                                  atol=config.atol, rtol=config.rtol)),
                "max_absolute_error": (float(difference.max().item())
                                       if difference is not None else None),
                "mean_absolute_error": (float(difference.mean().item())
                                        if difference is not None else None),
                "reference_nonfinite_count": int((~torch.isfinite(baseline_output)).sum().item()),
                "candidate_nonfinite_count": int((~torch.isfinite(output)).sum().item()),
                "reference": "torch.matmul(a, b, out=baseline_output) / PyTorch-cuBLAS",
                "atol": config.atol, "rtol": config.rtol,
                "shared_input_tensors": True,
            }
            if not correctness_evidence["passed"]:
                failures.append(_preparation_failure(
                    sampled, FailureCategory.CORRECTNESS,
                    "candidate failed FP16 correctness validation", environment,
                    self._id_factory(), self._clock(), compilation_succeeded=True,
                    correctness_evidence=correctness_evidence,
                ))
                continue
            self._launches[sampled.candidate_id] = launch
            eligible.append(PreparedCandidate(
                sampled=sampled, evaluation_id=self._id_factory(),
                compilation=CompilationResult(True, triton.__version__,
                                              "packaged Triton FC1 kernel compiled and launched"),
                correctness=CorrectnessResult(True, "passed shared-input PyTorch comparison"),
                correctness_evidence=correctness_evidence,
                setup_extensions={"candidate_parameters": parameters.to_dict(),
                                  "compilation_setup_host_us": setup_us},
            ))
        return BenchmarkPreparation(
            environment=environment,
            baseline_description={
                "implementation_id": BASELINE_ID,
                "operation": "torch.matmul(a, b, out=baseline_output)",
                "library": "PyTorch dispatch, normally cuBLAS on CUDA",
                "output_allocation": "preallocated FP16 output reused for every launch",
                "input_layout": "shared contiguous row-major FP16 tensors",
            },
            eligible=tuple(eligible), failed_evaluations=tuple(failures),
        )

    def warmup(self, implementation_ids: Sequence[str], launches: int) -> None:
        for identifier in implementation_ids:
            for _ in range(launches):
                self._launches[identifier]()
            self._torch.cuda.synchronize()

    def measure(self, implementation_id: str, launches: int) -> TimingObservation:
        from .triton_evaluator import CudaTritonRunner
        measured = CudaTritonRunner._measure(
            self._torch, self._launches[implementation_id], launches)
        return TimingObservation(
            measured["cuda_event_us_per_operation"],
            measured["host_submission_us_per_operation"],
            measured["host_wall_us_per_operation"],
        )


def _preparation_failure(
    sampled: SamplingRecord, category: FailureCategory, description: str,
    environment: Mapping[str, Any], evaluation_id: str, timestamp: datetime, *,
    compilation_succeeded: bool | None = None,
    correctness_evidence: Mapping[str, Any] | None = None,
) -> EvaluationRecord:
    lifecycle = [EvaluationStatus.PENDING]
    compilation = None
    correctness = None
    if category is FailureCategory.COMPILATION:
        lifecycle.append(EvaluationStatus.COMPILATION)
        compilation = CompilationResult(False, description=description)
    elif category is FailureCategory.CORRECTNESS:
        lifecycle.extend((EvaluationStatus.COMPILATION, EvaluationStatus.CORRECTNESS_VALIDATION))
        compilation = CompilationResult(True)
        correctness = CorrectnessResult(False, description)
    lifecycle.append(EvaluationStatus.FAILED)
    prepared = PreparedCandidate(sampled, evaluation_id, compilation or CompilationResult(False),
                                 correctness or CorrectnessResult(False),
                                 correctness_evidence or {})
    return EvaluationRecord(
        evaluation_id=EvaluationIdentity(evaluation_id),
        candidate_id=CandidateIdentity(sampled.candidate_id), status=EvaluationStatus.FAILED,
        lifecycle=tuple(lifecycle), metadata=_metadata(prepared, environment, timestamp),
        compilation=compilation, correctness=correctness,
        failure=FailureInfo(category, description),
        extensions={"candidate_parameters": sampled.parameters.to_dict(),
                    "correctness_evidence": dict(correctness_evidence or {})},
    )
