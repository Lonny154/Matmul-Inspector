"""CPU-only evaluation contracts and deterministic mock evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol
import uuid

from .sampling import SamplingRecord


EVALUATION_SCHEMA_VERSION = 1


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    COMPILATION = "compilation"
    CORRECTNESS_VALIDATION = "correctness_validation"
    BENCHMARKING = "benchmarking"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCategory(str, Enum):
    INVALID_CANDIDATE = "invalid_or_unsupported_candidate"
    COMPILATION = "compilation_failure"
    CORRECTNESS = "correctness_failure"
    BENCHMARK = "benchmark_failure"
    INFRASTRUCTURE = "evaluation_infrastructure_error"


class MockOutcome(str, Enum):
    SUCCESS = "success"
    COMPILATION_FAILURE = "compilation_failure"
    CORRECTNESS_FAILURE = "correctness_failure"
    BENCHMARK_FAILURE = "benchmark_failure"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


@dataclass(frozen=True)
class EvaluationIdentity:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("evaluation identity must be a non-empty string")


@dataclass(frozen=True)
class CandidateIdentity:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("candidate identity must be a non-empty string")


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    path: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("artifact kind must be a non-empty string")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("artifact path must be a non-empty string")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or not self.sha256.strip()):
            raise ValueError("artifact sha256 must be null or a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class CompilationResult:
    succeeded: bool
    compiler_version: str | None = None
    description: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise ValueError("compilation succeeded must be boolean")


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("correctness passed must be boolean")


@dataclass(frozen=True)
class BenchmarkResult:
    latency: float
    measurement_type: str
    unit: str
    repetitions: int
    synthetic: bool
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.latency, bool) or not isinstance(self.latency, (int, float)):
            raise ValueError("benchmark latency must be a finite non-negative number")
        if not math.isfinite(self.latency) or self.latency < 0:
            raise ValueError("benchmark latency must be a finite non-negative number")
        for name in ("measurement_type", "unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"benchmark {name} must be a non-empty string")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) \
                or self.repetitions <= 0:
            raise ValueError("benchmark repetitions must be a positive integer")
        if not isinstance(self.synthetic, bool):
            raise ValueError("benchmark synthetic must be boolean")


@dataclass(frozen=True)
class FailureInfo:
    category: FailureCategory
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, FailureCategory):
            raise ValueError("failure category must be a FailureCategory")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("failure description must be a non-empty string")


@dataclass(frozen=True)
class EvaluationMetadata:
    hardware_target: str
    backend: str
    workload_shape: tuple[int, int, int]
    precision: str
    timestamp: str
    software_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("hardware_target", "backend", "precision", "timestamp"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"evaluation metadata {name} must be a non-empty string")
        if (not isinstance(self.workload_shape, tuple) or len(self.workload_shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                       for value in self.workload_shape)):
            raise ValueError("workload_shape must contain three positive integers")
        if self.software_version is not None and (
                not isinstance(self.software_version, str) or not self.software_version.strip()):
            raise ValueError("software_version must be null or a non-empty string")


_TRANSITIONS = {
    EvaluationStatus.PENDING: {EvaluationStatus.COMPILATION, EvaluationStatus.FAILED},
    EvaluationStatus.COMPILATION: {
        EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.FAILED,
    },
    EvaluationStatus.CORRECTNESS_VALIDATION: {
        EvaluationStatus.BENCHMARKING, EvaluationStatus.FAILED,
    },
    EvaluationStatus.BENCHMARKING: {EvaluationStatus.COMPLETED, EvaluationStatus.FAILED},
    EvaluationStatus.COMPLETED: set(),
    EvaluationStatus.FAILED: set(),
}


class EvaluationLifecycle:
    """Track and validate one evaluation's stage transitions."""

    def __init__(self) -> None:
        self._history = [EvaluationStatus.PENDING]

    @property
    def status(self) -> EvaluationStatus:
        return self._history[-1]

    @property
    def history(self) -> tuple[EvaluationStatus, ...]:
        return tuple(self._history)

    @property
    def terminal(self) -> bool:
        return not _TRANSITIONS[self.status]

    def transition(self, status: EvaluationStatus) -> None:
        if not isinstance(status, EvaluationStatus):
            raise ValueError("lifecycle transition requires an EvaluationStatus")
        if status not in _TRANSITIONS[self.status]:
            raise ValueError(f"invalid evaluation transition: {self.status.value} -> {status.value}")
        self._history.append(status)


def _validate_history(history: tuple[EvaluationStatus, ...], status: EvaluationStatus) -> None:
    if not history or history[0] is not EvaluationStatus.PENDING or history[-1] is not status:
        raise ValueError("lifecycle history must begin at pending and end at the record status")
    for current, following in zip(history, history[1:]):
        if following not in _TRANSITIONS[current]:
            raise ValueError(f"invalid evaluation transition: {current.value} -> {following.value}")


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: EvaluationIdentity
    candidate_id: CandidateIdentity
    status: EvaluationStatus
    lifecycle: tuple[EvaluationStatus, ...]
    metadata: EvaluationMetadata
    compilation: CompilationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    failure: FailureInfo | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_history(self.lifecycle, self.status)
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema: {self.schema_version!r}")
        if self.status is EvaluationStatus.COMPLETED:
            if self.failure is not None:
                raise ValueError("completed evaluations cannot contain failure information")
            if self.compilation is None or not self.compilation.succeeded:
                raise ValueError("completed evaluations require successful compilation")
            if self.correctness is None or not self.correctness.passed:
                raise ValueError("completed evaluations require passing correctness")
            if self.benchmark is None:
                raise ValueError("completed evaluations require a benchmark result")
        elif self.status is EvaluationStatus.FAILED:
            if self.failure is None:
                raise ValueError("failed evaluations require failure information")
            if self.benchmark is not None:
                raise ValueError("failed evaluations cannot contain a successful benchmark result")
            self._validate_failure_stage()
        elif self.failure is not None:
            raise ValueError("non-failed evaluations cannot contain failure information")

    def _validate_failure_stage(self) -> None:
        category = self.failure.category if self.failure else None
        previous = self.lifecycle[-2] if len(self.lifecycle) > 1 else None
        expected = {
            FailureCategory.COMPILATION: EvaluationStatus.COMPILATION,
            FailureCategory.CORRECTNESS: EvaluationStatus.CORRECTNESS_VALIDATION,
            FailureCategory.BENCHMARK: EvaluationStatus.BENCHMARKING,
        }
        if category in expected and previous is not expected[category]:
            raise ValueError(f"{category.value} must occur during {expected[category].value}")
        if category is FailureCategory.COMPILATION and (
                self.compilation is None or self.compilation.succeeded):
            raise ValueError("compilation failure requires an unsuccessful compilation result")
        if category is FailureCategory.CORRECTNESS and (
                self.correctness is None or self.correctness.passed):
            raise ValueError("correctness failure requires a failing correctness result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id.value,
            "candidate_id": self.candidate_id.value,
            "status": self.status.value,
            "lifecycle": [item.value for item in self.lifecycle],
            "metadata": {
                "hardware_target": self.metadata.hardware_target,
                "backend": self.metadata.backend,
                "workload_shape": list(self.metadata.workload_shape),
                "precision": self.metadata.precision,
                "timestamp": self.metadata.timestamp,
                "software_version": self.metadata.software_version,
            },
            "compilation": _compilation_to_dict(self.compilation),
            "correctness": (_correctness_to_dict(self.correctness)
                            if self.correctness is not None else None),
            "benchmark": _benchmark_to_dict(self.benchmark),
            "failure": ({"category": self.failure.category.value,
                         "description": self.failure.description} if self.failure else None),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        if not isinstance(value, Mapping):
            raise ValueError("evaluation record must be a JSON object")
        if "schema_version" not in value:
            raise ValueError("evaluation record is missing required field: schema_version")
        if value["schema_version"] == 1:
            return _parse_evaluation_v1(value)
        raise ValueError(f"unsupported evaluation schema: {value['schema_version']!r}")


def _compilation_to_dict(value: CompilationResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"succeeded": value.succeeded, "compiler_version": value.compiler_version,
            "description": value.description,
            "artifacts": [item.to_dict() for item in value.artifacts]}


def _correctness_to_dict(value: CorrectnessResult) -> dict[str, Any]:
    return {"passed": value.passed, "description": value.description}


def _benchmark_to_dict(value: BenchmarkResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"latency": value.latency, "measurement_type": value.measurement_type,
            "unit": value.unit, "repetitions": value.repetitions,
            "synthetic": value.synthetic,
            "artifacts": [item.to_dict() for item in value.artifacts]}


def _strict_fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")
    return value


def _artifact_from_dict(value: Any) -> ArtifactReference:
    item = _strict_fields(value, {"kind", "path", "sha256"}, "artifact reference")
    return ArtifactReference(kind=item["kind"], path=item["path"], sha256=item["sha256"])


def _artifacts_from_value(value: Any) -> tuple[ArtifactReference, ...]:
    if not isinstance(value, list):
        raise ValueError("artifacts must be an array")
    return tuple(_artifact_from_dict(item) for item in value)


def _parse_evaluation_v1(value: Mapping[str, Any]) -> EvaluationRecord:
    expected = {"schema_version", "evaluation_id", "candidate_id", "status", "lifecycle",
                "metadata", "compilation", "correctness", "benchmark", "failure",
                "artifacts", "extensions"}
    record = _strict_fields(value, expected, "version-1 evaluation record")
    metadata = _strict_fields(
        record["metadata"], {"hardware_target", "backend", "workload_shape", "precision",
                             "timestamp", "software_version"}, "evaluation metadata",
    )
    shape = metadata["workload_shape"]
    if not isinstance(shape, list):
        raise ValueError("evaluation metadata workload_shape must be an array")
    compilation = None
    if record["compilation"] is not None:
        raw = _strict_fields(record["compilation"],
                             {"succeeded", "compiler_version", "description", "artifacts"},
                             "compilation result")
        compilation = CompilationResult(raw["succeeded"], raw["compiler_version"],
                                        raw["description"], _artifacts_from_value(raw["artifacts"]))
    correctness = None
    if record["correctness"] is not None:
        raw = _strict_fields(record["correctness"], {"passed", "description"},
                             "correctness result")
        correctness = CorrectnessResult(raw["passed"], raw["description"])
    benchmark = None
    if record["benchmark"] is not None:
        raw = _strict_fields(record["benchmark"],
                             {"latency", "measurement_type", "unit", "repetitions",
                              "synthetic", "artifacts"}, "benchmark result")
        benchmark = BenchmarkResult(raw["latency"], raw["measurement_type"], raw["unit"],
                                    raw["repetitions"], raw["synthetic"],
                                    _artifacts_from_value(raw["artifacts"]))
    failure = None
    if record["failure"] is not None:
        raw = _strict_fields(record["failure"], {"category", "description"}, "failure")
        failure = FailureInfo(FailureCategory(raw["category"]), raw["description"])
    if not isinstance(record["lifecycle"], list):
        raise ValueError("lifecycle must be an array")
    if not isinstance(record["extensions"], Mapping):
        raise ValueError("extensions must be an object")
    try:
        return EvaluationRecord(
            evaluation_id=EvaluationIdentity(record["evaluation_id"]),
            candidate_id=CandidateIdentity(record["candidate_id"]),
            status=EvaluationStatus(record["status"]),
            lifecycle=tuple(EvaluationStatus(item) for item in record["lifecycle"]),
            metadata=EvaluationMetadata(
                metadata["hardware_target"], metadata["backend"], tuple(shape),
                metadata["precision"], metadata["timestamp"], metadata["software_version"],
            ),
            compilation=compilation, correctness=correctness, benchmark=benchmark,
            failure=failure, artifacts=_artifacts_from_value(record["artifacts"]),
            extensions=dict(record["extensions"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid version-1 evaluation record: {error}") from error


class Evaluator(Protocol):
    def evaluate(self, candidate: SamplingRecord) -> EvaluationRecord:
        """Evaluate one sampled candidate without modifying it."""


@dataclass(frozen=True)
class MockEvaluatorConfig:
    outcome: MockOutcome = MockOutcome.SUCCESS
    synthetic_latency: float = 1.0
    measurement_type: str = "synthetic_mock_latency"
    unit: str = "synthetic_ms"
    repetitions: int = 10
    software_version: str | None = "mock-evaluator-1"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, MockOutcome):
            raise ValueError("outcome must be a MockOutcome")
        # Reuse the benchmark contract even when a controlled failure means it
        # will not be attached to the final record.
        BenchmarkResult(self.synthetic_latency, self.measurement_type, self.unit,
                        self.repetitions, synthetic=True)


class MockEvaluator:
    """Deterministic CPU-only evaluator producing explicitly synthetic data."""

    def __init__(
        self,
        config: MockEvaluatorConfig = MockEvaluatorConfig(),
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._id_factory = id_factory or (lambda: f"evaluation-{uuid.uuid4().hex}")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def evaluate(self, candidate: SamplingRecord) -> EvaluationRecord:
        if not isinstance(candidate, SamplingRecord):
            raise TypeError("candidate must be a SamplingRecord")
        lifecycle = EvaluationLifecycle()
        metadata = EvaluationMetadata(
            hardware_target=candidate.context.hardware_target,
            backend=candidate.context.backend,
            workload_shape=(candidate.context.m, candidate.context.n, candidate.context.k),
            precision=candidate.context.precision,
            timestamp=self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            software_version=self.config.software_version,
        )
        identity = EvaluationIdentity(self._id_factory())
        common = {"evaluation_id": identity, "candidate_id": CandidateIdentity(candidate.candidate_id),
                  "metadata": metadata, "extensions": {"measurement_provenance": "synthetic"}}
        if self.config.outcome is MockOutcome.INFRASTRUCTURE_ERROR:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(status=lifecycle.status, lifecycle=lifecycle.history,
                                    failure=FailureInfo(FailureCategory.INFRASTRUCTURE,
                                                        "controlled mock infrastructure error"),
                                    **common)
        lifecycle.transition(EvaluationStatus.COMPILATION)
        if self.config.outcome is MockOutcome.COMPILATION_FAILURE:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history,
                compilation=CompilationResult(False, self.config.software_version,
                                              "controlled mock compilation failure"),
                failure=FailureInfo(FailureCategory.COMPILATION,
                                    "controlled mock compilation failure"), **common,
            )
        compilation = CompilationResult(True, self.config.software_version,
                                        "synthetic compilation succeeded")
        lifecycle.transition(EvaluationStatus.CORRECTNESS_VALIDATION)
        if self.config.outcome is MockOutcome.CORRECTNESS_FAILURE:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
                correctness=CorrectnessResult(False, "controlled synthetic mismatch"),
                failure=FailureInfo(FailureCategory.CORRECTNESS,
                                    "controlled mock correctness failure"), **common,
            )
        correctness = CorrectnessResult(True, "synthetic correctness check passed")
        lifecycle.transition(EvaluationStatus.BENCHMARKING)
        if self.config.outcome is MockOutcome.BENCHMARK_FAILURE:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
                correctness=correctness,
                failure=FailureInfo(FailureCategory.BENCHMARK,
                                    "controlled mock benchmark failure"), **common,
            )
        benchmark = BenchmarkResult(
            self.config.synthetic_latency, self.config.measurement_type, self.config.unit,
            self.config.repetitions, synthetic=True,
        )
        lifecycle.transition(EvaluationStatus.COMPLETED)
        return EvaluationRecord(
            status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
            correctness=correctness, benchmark=benchmark, **common,
        )


def write_evaluation_jsonl(path: str | Path, records: Iterable[EvaluationRecord]) -> None:
    """Atomically replace a JSONL file after serializing every record."""
    destination = Path(path)
    materialized = tuple(records)
    if not all(isinstance(record, EvaluationRecord) for record in materialized):
        raise TypeError("evaluation JSONL records must be EvaluationRecord objects")
    identities = [record.evaluation_id.value for record in materialized]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"duplicate evaluation IDs: {', '.join(duplicates)}")
    serialized = [json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)
                  for record in materialized]
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for line in serialized:
                stream.write(line + "\n")
            stream.flush()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_evaluation_jsonl(path: str | Path) -> list[EvaluationRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid evaluation record on line {line_number}: {error}") from error
    return records
