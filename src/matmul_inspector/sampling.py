"""CPU-only candidate generation for discrete kernel search spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


PARAMETER_NAMES = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
RECORD_SCHEMA_VERSION = 1


class SearchSpaceExhausted(ValueError):
    """Raised when a unique sample asks for more candidates than remain."""


@dataclass(frozen=True)
class SearchContext:
    """Workload and target metadata that gives a candidate its meaning."""

    m: int
    n: int
    k: int
    precision: str
    backend: str = "triton"
    hardware_target: str = "unspecified"

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("precision", "backend", "hardware_target"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class TritonGemmConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in PARAMETER_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TritonGemmConfig":
        if not isinstance(value, Mapping) or set(value) != set(PARAMETER_NAMES):
            raise ValueError(f"candidate parameters must contain exactly: {', '.join(PARAMETER_NAMES)}")
        return cls(**{name: value[name] for name in PARAMETER_NAMES})


ConstraintPredicate = Callable[[TritonGemmConfig, SearchContext], bool]


@dataclass(frozen=True)
class CandidateConstraint:
    """A named, pure cross-parameter or target-specific validity rule.

    Predicates must remain deterministic and must not depend on mutable external
    state. Search spaces cache their first complete validation result.
    """

    name: str
    predicate: ConstraintPredicate = field(repr=False, compare=False)
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("constraint name must be a non-empty string")
        if not callable(self.predicate):
            raise ValueError("constraint predicate must be callable")

    def accepts(self, config: TritonGemmConfig, context: SearchContext) -> bool:
        result = self.predicate(config, context)
        if not isinstance(result, bool):
            raise TypeError(f"constraint {self.name!r} must return bool")
        return result


class TritonGemmSearchSpace:
    """Validated discrete values plus explicit candidate-level constraints."""

    def __init__(
        self,
        values: Mapping[str, Sequence[int]],
        context: SearchContext,
        *,
        constraints: Sequence[CandidateConstraint] = (),
        identifier: str = "default",
    ) -> None:
        if context.backend != "triton":
            raise ValueError("TritonGemmSearchSpace requires context.backend == 'triton'")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("search space identifier must be a non-empty string")
        if not isinstance(values, Mapping) or set(values) != set(PARAMETER_NAMES):
            raise ValueError(f"search space must contain exactly: {', '.join(PARAMETER_NAMES)}")
        normalized: dict[str, tuple[int, ...]] = {}
        for name in PARAMETER_NAMES:
            candidates = values[name]
            if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence) or not candidates:
                raise ValueError(f"{name} must be a non-empty sequence")
            result = tuple(candidates)
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in result):
                raise ValueError(f"{name} values must be positive integers")
            if len(set(result)) != len(result):
                raise ValueError(f"{name} values must not contain duplicates")
            normalized[name] = result
        if not all(isinstance(item, CandidateConstraint) for item in constraints):
            raise ValueError("constraints must contain CandidateConstraint objects")
        self._values = MappingProxyType(normalized)
        self._context = context
        self._constraints = tuple(constraints)
        self._identifier = identifier
        self._valid_candidates_cache: tuple[TritonGemmConfig, ...] | None = None

    @property
    def values(self) -> Mapping[str, tuple[int, ...]]:
        return self._values

    @property
    def context(self) -> SearchContext:
        return self._context

    @property
    def constraints(self) -> tuple[CandidateConstraint, ...]:
        return self._constraints

    @property
    def identifier(self) -> str:
        return self._identifier

    def validation_errors(self, config: TritonGemmConfig) -> list[str]:
        errors = [f"{name}={getattr(config, name)} is outside the search space"
                  for name in PARAMETER_NAMES if getattr(config, name) not in self.values[name]]
        for constraint in self.constraints:
            if not constraint.accepts(config, self.context):
                errors.append(f"constraint {constraint.name!r} rejected the candidate")
        return errors

    def validate(self, config: TritonGemmConfig) -> None:
        errors = self.validation_errors(config)
        if errors:
            raise ValueError("; ".join(errors))

    def valid_candidates(self) -> tuple[TritonGemmConfig, ...]:
        if self._valid_candidates_cache is not None:
            return self._valid_candidates_cache
        candidates = []
        for combination in itertools.product(*(self.values[name] for name in PARAMETER_NAMES)):
            config = TritonGemmConfig(**dict(zip(PARAMETER_NAMES, combination)))
            if not self.validation_errors(config):
                candidates.append(config)
        self._valid_candidates_cache = tuple(candidates)
        return self._valid_candidates_cache


def candidate_identifier(config: TritonGemmConfig, context: SearchContext) -> str:
    """Return a stable identity based on semantic context, never timestamps."""
    payload = {"context": asdict(context), "parameters": config.to_dict()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "candidate-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


class _BaseSampler:
    name = "base"

    def __init__(self, search_space: TritonGemmSearchSpace, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.search_space = search_space
        self.seed = seed
        self._random = random.Random(seed)
        self._seen: set[TritonGemmConfig] = set()

    def _available(self, unique: bool) -> list[TritonGemmConfig]:
        candidates = list(self.search_space.valid_candidates())
        if not candidates:
            raise SearchSpaceExhausted("the search space contains no valid candidates")
        return [item for item in candidates if not unique or item not in self._seen]

    def sample(self, count: int = 1, *, unique: bool = False) -> list[TritonGemmConfig]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        available = self._available(unique)
        if unique and count > len(available):
            raise SearchSpaceExhausted(
                f"requested {count} unique candidates but only {len(available)} remain"
            )
        chosen = self._choose(available, count, unique)
        if unique:
            self._seen.update(chosen)
        return chosen

    def _choose(self, available: list[TritonGemmConfig], count: int,
                unique: bool) -> list[TritonGemmConfig]:
        raise NotImplementedError


class UniformSampler(_BaseSampler):
    name = "uniform"

    def _choose(self, available: list[TritonGemmConfig], count: int,
                unique: bool) -> list[TritonGemmConfig]:
        if unique:
            return self._random.sample(available, count)
        return self._random.choices(available, k=count)


class WeightedSampler(_BaseSampler):
    """Sample using the product of independent per-parameter value weights.

    Unique samples use sequential probability-proportional selection without
    replacement. Only strictly positive-weight candidates are eligible.
    """

    name = "weighted"

    def __init__(self, search_space: TritonGemmSearchSpace, *, seed: int,
                 weights: Mapping[str, Mapping[int, float]]) -> None:
        super().__init__(search_space, seed=seed)
        if not isinstance(weights, Mapping) or set(weights) != set(PARAMETER_NAMES):
            raise ValueError(f"weights must contain exactly: {', '.join(PARAMETER_NAMES)}")
        normalized: dict[str, dict[int, float]] = {}
        for name in PARAMETER_NAMES:
            supplied = weights[name]
            if not isinstance(supplied, Mapping) or set(supplied) != set(search_space.values[name]):
                raise ValueError(f"weights for {name} must match its search-space values")
            converted = {value: float(supplied[value]) for value in search_space.values[name]}
            if any(not math.isfinite(weight) or weight < 0 for weight in converted.values()):
                raise ValueError(f"weights for {name} must be finite and non-negative")
            total = sum(converted.values())
            if total <= 0:
                raise ValueError(f"weights for {name} must include a positive value")
            normalized[name] = {value: weight / total for value, weight in converted.items()}
        self.weights = normalized

    def _candidate_weight(self, config: TritonGemmConfig) -> float:
        return math.prod(self.weights[name][getattr(config, name)] for name in PARAMETER_NAMES)

    def _available(self, unique: bool) -> list[TritonGemmConfig]:
        valid = super()._available(unique)
        eligible = [item for item in valid if self._candidate_weight(item) > 0]
        if not eligible:
            raise SearchSpaceExhausted(
                "no remaining valid candidates have strictly positive sampling weight"
            )
        return eligible

    def _choose(self, available: list[TritonGemmConfig], count: int,
                unique: bool) -> list[TritonGemmConfig]:
        chosen: list[TritonGemmConfig] = []
        pool = list(available)
        for _ in range(count):
            candidate_weights = [self._candidate_weight(item) for item in pool]
            index = self._random.choices(range(len(pool)), weights=candidate_weights, k=1)[0]
            chosen.append(pool[index])
            if unique:
                pool.pop(index)
        return chosen


@dataclass(frozen=True)
class SamplingRecord:
    candidate_id: str
    search_space_id: str
    sampler: str
    seed: int
    parameters: TritonGemmConfig
    context: SearchContext
    sampling_order: int
    timestamp: str
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "search_space_id": self.search_space_id,
            "sampler": self.sampler,
            "seed": self.seed,
            "parameters": self.parameters.to_dict(),
            "context": asdict(self.context),
            "sampling_order": self.sampling_order,
            "timestamp": self.timestamp,
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingRecord":
        if not isinstance(value, Mapping):
            raise ValueError("sampling record must be a JSON object")
        if "schema_version" not in value:
            raise ValueError("sampling record is missing required field: schema_version")
        version = value["schema_version"]
        if version == 1:
            return _parse_sampling_record_v1(value)
        raise ValueError(f"unsupported sampling record schema: {version!r}")


def _parse_sampling_record_v1(value: Mapping[str, Any]) -> SamplingRecord:
    """Strictly parse the stable version-1 sampling-record schema."""
    expected = {"schema_version", "candidate_id", "search_space_id", "sampler", "seed",
                "parameters", "context", "sampling_order", "timestamp", "extensions"}
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"version-1 sampling record is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown version-1 sampling record fields: {', '.join(unknown)}")
    if not isinstance(value["context"], Mapping):
        raise ValueError("version-1 sampling record context must be an object")
    if not isinstance(value["extensions"], Mapping):
        raise ValueError("version-1 sampling record extensions must be an object")
    try:
        parameters = TritonGemmConfig.from_dict(value["parameters"])
        context = SearchContext(**value["context"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid version-1 sampling record data: {error}") from error
    record = SamplingRecord(
        candidate_id=value["candidate_id"], search_space_id=value["search_space_id"],
        sampler=value["sampler"], seed=value["seed"], parameters=parameters,
        context=context, sampling_order=value["sampling_order"],
        timestamp=value["timestamp"], extensions=dict(value["extensions"]),
    )
    if record.candidate_id != candidate_identifier(record.parameters, record.context):
        raise ValueError("candidate_id does not match parameters and context")
    if (isinstance(record.sampling_order, bool)
            or not isinstance(record.sampling_order, int) or record.sampling_order < 0):
        raise ValueError("sampling_order must be a non-negative integer")
    return record


def make_sampling_records(
    candidates: Iterable[TritonGemmConfig], sampler: _BaseSampler, *,
    start_order: int = 0, timestamp: str | None = None,
) -> list[SamplingRecord]:
    recorded_at = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return [
        SamplingRecord(
            candidate_id=candidate_identifier(config, sampler.search_space.context),
            search_space_id=sampler.search_space.identifier, sampler=sampler.name,
            seed=sampler.seed, parameters=config, context=sampler.search_space.context,
            sampling_order=start_order + index, timestamp=recorded_at,
        )
        for index, config in enumerate(candidates)
    ]


def write_jsonl(path: str | Path, records: Iterable[SamplingRecord]) -> None:
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> list[SamplingRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(SamplingRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid sampling record on line {line_number}: {error}") from error
    return records
