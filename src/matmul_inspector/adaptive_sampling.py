"""Offline, reproducible feedback-guided sampling for bounded search spaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence
import uuid

from .controlled_benchmark import read_controlled_manifest
from .evaluation import EvaluationRecord, EvaluationStatus, read_evaluation_jsonl
from .sampling import (
    PARAMETER_NAMES, CandidateConstraint, SearchContext, SearchSpaceExhausted,
    TritonGemmConfig, TritonGemmSearchSpace, candidate_identifier,
)


ADAPTIVE_STATE_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "parameter_reward_mixture_v1"
SUPPORTED_MEASUREMENTS = frozenset({
    "median CUDA-event kernel latency",
    "interleaved median CUDA-event kernel latency",
})


def _config_key(config: TritonGemmConfig) -> tuple[int, ...]:
    return tuple(getattr(config, name) for name in PARAMETER_NAMES)


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class AdaptiveSampler:
    """Parameter-wise offline learning with candidate-level epsilon exploration."""

    name = "adaptive_parameter_weighting"

    def __init__(
        self,
        search_space: TritonGemmSearchSpace,
        *,
        seed: int,
        epsilon: float = 0.25,
        prior_strength: float = 2.0,
        exclude_observed_from_unique: bool = True,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) \
                or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("epsilon must be finite and in (0, 1]")
        if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) \
                or not math.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be finite and positive")
        candidates = search_space.valid_candidates()
        if not candidates:
            raise SearchSpaceExhausted("the search space contains no valid candidates")
        self.search_space = search_space
        self.seed = seed
        self.epsilon = float(epsilon)
        self.prior_strength = float(prior_strength)
        self.exclude_observed_from_unique = exclude_observed_from_unique
        self._random = random.Random(seed)
        self._seen: set[TritonGemmConfig] = set()
        self._candidate_probabilities = self._uniform_distribution()
        self._parameter_scores = self._uniform_parameter_scores()
        self.feedback_source_ids: list[str] = []
        self.environment_fingerprint: str | None = None
        self.update_history: list[dict[str, Any]] = []

    def _uniform_distribution(self) -> dict[TritonGemmConfig, float]:
        candidates = self.search_space.valid_candidates()
        probability = 1.0 / len(candidates)
        return {candidate: probability for candidate in candidates}

    def _uniform_parameter_scores(self) -> dict[str, dict[int, float]]:
        return {name: {value: 0.5 for value in self.search_space.values[name]}
                for name in PARAMETER_NAMES}

    def candidate_probabilities(self) -> dict[TritonGemmConfig, float]:
        return dict(self._candidate_probabilities)

    def parameter_scores(self) -> dict[str, dict[int, float]]:
        return {name: dict(values) for name, values in self._parameter_scores.items()}

    def update(self, records: Iterable[EvaluationRecord], *, source_id: str) -> dict[str, Any]:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("feedback source_id must be a non-empty string")
        if self.feedback_source_ids and source_id not in self.feedback_source_ids:
            raise ValueError(
                "cannot combine unmatched feedback sources in one adaptive state; "
                f"existing={self.feedback_source_ids[0]!r}, requested={source_id!r}"
            )
        if self.update_history:
            raise ValueError(
                "this algorithm version accepts one offline feedback update; reconstruct the "
                "state with the complete deduplicated record set to update again"
            )
        scores_before = self.parameter_scores()
        snapshots = list(records)
        by_evaluation: dict[str, EvaluationRecord] = {}
        exclusions = Counter()
        failures = Counter()
        for record in snapshots:
            if not isinstance(record, EvaluationRecord):
                exclusions["not_an_evaluation_record"] += 1
                continue
            evaluation_id = record.evaluation_id.value
            if evaluation_id in by_evaluation:
                exclusions["duplicate_evaluation_id"] += 1
                continue
            by_evaluation[evaluation_id] = record
            if record.status is EvaluationStatus.FAILED and record.failure is not None:
                failures[record.failure.category.value] += 1

        compatible = []
        preeligible = []
        for record in by_evaluation.values():
            reason, recovered = self._eligibility(record)
            if reason is not None:
                exclusions[reason] += 1
                continue
            preeligible.append((record, recovered))
            compatible.append(record.metadata.software_version or "unknown")
        selected_environment = self.environment_fingerprint
        if selected_environment is None and compatible:
            counts = Counter(compatible)
            selected_environment = sorted(counts, key=lambda item: (-counts[item], item))[0]
        if self.environment_fingerprint is not None and selected_environment != self.environment_fingerprint:
            raise ValueError("feedback environment does not match saved adaptive state")

        observations: dict[TritonGemmConfig, list[float]] = defaultdict(list)
        eligible_evaluations = 0
        for record, recovered in preeligible:
            fingerprint = record.metadata.software_version or "unknown"
            if fingerprint != selected_environment:
                exclusions["software_environment_mismatch"] += 1
                continue
            observations[recovered].append(float(record.benchmark.latency))
            eligible_evaluations += 1

        candidate_latencies = {
            config: statistics.median(latencies) for config, latencies in observations.items()
        }
        rewards = self._rewards(candidate_latencies)
        if rewards:
            self._learn(rewards)
            if self.exclude_observed_from_unique:
                self._seen.update(rewards)
            self.environment_fingerprint = selected_environment
        if source_id not in self.feedback_source_ids:
            self.feedback_source_ids.append(source_id)
        summary = {
            "source_id": source_id,
            "input_record_count": len(snapshots),
            "unique_evaluation_count": len(by_evaluation),
            "eligible_evaluation_count": eligible_evaluations,
            "eligible_candidate_count": len(candidate_latencies),
            "excluded_observation_count": sum(exclusions.values()),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "failure_categories": dict(sorted(failures.items())),
            "repeated_candidate_policy": "median latency per candidate; each candidate contributes once",
            "software_environment": selected_environment,
            "candidate_latencies_us": {
                candidate_identifier(config, self.search_space.context): latency
                for config, latency in sorted(candidate_latencies.items(), key=lambda item: _config_key(item[0]))
            },
            "candidate_rewards": {
                candidate_identifier(config, self.search_space.context): reward
                for config, reward in sorted(rewards.items(), key=lambda item: _config_key(item[0]))
            },
            "parameter_scores_before": scores_before,
            "parameter_scores_after": self.parameter_scores(),
            "reward_formula": "(maximum_latency - latency) / (maximum_latency - minimum_latency)",
            "exploration_mixture": {
                "epsilon": self.epsilon,
                "minimum_candidate_probability": self.epsilon / len(self._candidate_probabilities),
            },
        }
        self.update_history.append(summary)
        return json.loads(json.dumps(summary, allow_nan=False))

    def _eligibility(self, record: EvaluationRecord) -> tuple[str | None, TritonGemmConfig | None]:
        if record.status is not EvaluationStatus.COMPLETED:
            return "evaluation_not_completed", None
        if record.correctness is None or not record.correctness.passed:
            return "correctness_not_passed", None
        benchmark = record.benchmark
        if benchmark is None:
            return "benchmark_missing", None
        if benchmark.synthetic:
            return "synthetic_measurement", None
        if benchmark.unit != "us":
            return "unsupported_measurement_unit", None
        if benchmark.measurement_type not in SUPPORTED_MEASUREMENTS:
            return "unsupported_measurement_type", None
        if not math.isfinite(benchmark.latency) or benchmark.latency <= 0:
            return "invalid_latency", None
        context = self.search_space.context
        metadata = record.metadata
        if metadata.workload_shape != (context.m, context.n, context.k):
            return "workload_mismatch", None
        if metadata.precision.lower() != context.precision.lower():
            return "precision_mismatch", None
        if metadata.backend != context.backend:
            return "backend_mismatch", None
        if metadata.hardware_target.lower() != context.hardware_target.lower():
            return "hardware_target_mismatch", None
        raw = record.extensions.get("candidate_parameters")
        try:
            recovered = TritonGemmConfig.from_dict(raw)
        except (TypeError, ValueError):
            return "candidate_parameters_unrecoverable", None
        if record.candidate_id.value != candidate_identifier(recovered, context):
            return "candidate_identity_mismatch", None
        if self.search_space.validation_errors(recovered):
            return "candidate_outside_search_space", None
        return None, recovered

    @staticmethod
    def _rewards(latencies: Mapping[TritonGemmConfig, float]) -> dict[TritonGemmConfig, float]:
        if not latencies:
            return {}
        minimum, maximum = min(latencies.values()), max(latencies.values())
        if maximum == minimum:
            return {candidate: 0.5 for candidate in latencies}
        return {candidate: (maximum - latency) / (maximum - minimum)
                for candidate, latency in latencies.items()}

    def _learn(self, rewards: Mapping[TritonGemmConfig, float]) -> None:
        scores: dict[str, dict[int, float]] = {}
        for name in PARAMETER_NAMES:
            scores[name] = {}
            for value in self.search_space.values[name]:
                observed = [reward for candidate, reward in rewards.items()
                            if getattr(candidate, name) == value]
                scores[name][value] = (
                    self.prior_strength * 0.5 + sum(observed)
                ) / (self.prior_strength + len(observed))
        candidates = self.search_space.valid_candidates()
        raw = {candidate: math.prod(scores[name][getattr(candidate, name)]
                                    for name in PARAMETER_NAMES)
               for candidate in candidates}
        total = sum(raw.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("learned candidate weights are not finite and positive")
        learned = {candidate: value / total for candidate, value in raw.items()}
        uniform = 1.0 / len(candidates)
        mixed = {candidate: (1.0 - self.epsilon) * learned[candidate]
                 + self.epsilon * uniform for candidate in candidates}
        normalization = sum(mixed.values())
        self._candidate_probabilities = {
            candidate: probability / normalization for candidate, probability in mixed.items()
        }
        self._parameter_scores = scores

    def sample(self, count: int = 1, *, unique: bool = False) -> list[TritonGemmConfig]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        available = [candidate for candidate in self.search_space.valid_candidates()
                     if not unique or candidate not in self._seen]
        if not available:
            raise SearchSpaceExhausted("no candidates remain available")
        if unique and count > len(available):
            raise SearchSpaceExhausted(
                f"requested {count} unique candidates but only {len(available)} remain"
            )
        chosen = []
        pool = list(available)
        for _ in range(count):
            weights = [self._candidate_probabilities[candidate] for candidate in pool]
            index = self._random.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool[index])
            if unique:
                pool.pop(index)
        if unique:
            self._seen.update(chosen)
        return chosen

    def to_dict(self) -> dict[str, Any]:
        candidates = self.search_space.valid_candidates()
        return {
            "schema_version": ADAPTIVE_STATE_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "search_space_id": self.search_space.identifier,
            "search_space": {name: list(self.search_space.values[name]) for name in PARAMETER_NAMES},
            "valid_candidate_ids": [candidate_identifier(item, self.search_space.context)
                                    for item in candidates],
            "context": asdict(self.search_space.context),
            "seed": self.seed,
            "epsilon": self.epsilon,
            "prior_strength": self.prior_strength,
            "exclude_observed_from_unique": self.exclude_observed_from_unique,
            "feedback_source_ids": list(self.feedback_source_ids),
            "environment_fingerprint": self.environment_fingerprint,
            "parameter_scores": {name: {str(value): score for value, score in values.items()}
                                 for name, values in self._parameter_scores.items()},
            "candidate_probabilities": {
                candidate_identifier(candidate, self.search_space.context): probability
                for candidate, probability in self._candidate_probabilities.items()
            },
            "seen_candidate_ids": [candidate_identifier(candidate, self.search_space.context)
                                   for candidate in sorted(self._seen, key=_config_key)],
            "random_state": self._random.getstate(),
            "update_history": json.loads(json.dumps(self.update_history, allow_nan=False)),
            "extensions": {},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdaptiveSampler":
        expected = {
            "schema_version", "algorithm_version", "search_space_id", "search_space",
            "valid_candidate_ids", "context", "seed", "epsilon", "prior_strength",
            "exclude_observed_from_unique", "feedback_source_ids", "environment_fingerprint",
            "parameter_scores", "candidate_probabilities", "seen_candidate_ids",
            "random_state", "update_history", "extensions",
        }
        if not isinstance(value, Mapping):
            raise ValueError("adaptive state must be an object")
        if value.get("schema_version") != ADAPTIVE_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported adaptive state schema: {value.get('schema_version')!r}")
        if value.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError(f"unsupported adaptive algorithm: {value.get('algorithm_version')!r}")
        missing, unknown = sorted(expected - set(value)), sorted(set(value) - expected)
        if missing:
            raise ValueError(f"adaptive state missing fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"unknown adaptive state fields: {', '.join(unknown)}")
        context = SearchContext(**value["context"])
        raw_space = value["search_space"]
        base_space = TritonGemmSearchSpace(raw_space, context, identifier=value["search_space_id"])
        allowed = frozenset(value["valid_candidate_ids"])
        actual = {candidate_identifier(item, context) for item in base_space.valid_candidates()}
        if not allowed or not allowed.issubset(actual):
            raise ValueError("adaptive state valid candidates do not match the search definition")
        constraint = CandidateConstraint(
            "restored-valid-candidate-set",
            lambda candidate, restored_context: candidate_identifier(candidate, restored_context) in allowed,
        )
        space = TritonGemmSearchSpace(raw_space, context, constraints=(constraint,),
                                      identifier=value["search_space_id"])
        sampler = cls(
            space, seed=value["seed"], epsilon=value["epsilon"],
            prior_strength=value["prior_strength"],
            exclude_observed_from_unique=value["exclude_observed_from_unique"],
        )
        by_id = {candidate_identifier(item, context): item for item in space.valid_candidates()}
        probabilities = value["candidate_probabilities"]
        if set(probabilities) != set(by_id):
            raise ValueError("adaptive candidate probabilities do not match valid candidates")
        parsed = {by_id[identifier]: float(probability)
                  for identifier, probability in probabilities.items()}
        if not all(math.isfinite(item) and item > 0 for item in parsed.values()) \
                or not math.isclose(sum(parsed.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("adaptive candidate probabilities must be positive and sum to one")
        sampler._candidate_probabilities = parsed
        sampler._parameter_scores = {
            name: {int(raw): float(score) for raw, score in value["parameter_scores"][name].items()}
            for name in PARAMETER_NAMES
        }
        try:
            sampler._seen = {by_id[identifier] for identifier in value["seen_candidate_ids"]}
        except KeyError as error:
            raise ValueError(f"unknown seen candidate ID: {error.args[0]}") from error
        sampler._random.setstate(_tuple_tree(value["random_state"]))
        sampler.feedback_source_ids = list(value["feedback_source_ids"])
        sampler.environment_fingerprint = value["environment_fingerprint"]
        sampler.update_history = json.loads(json.dumps(value["update_history"], allow_nan=False))
        return sampler


def write_adaptive_state(path: str | Path, sampler: AdaptiveSampler) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(sampler.to_dict(), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_adaptive_state(path: str | Path) -> AdaptiveSampler:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read adaptive state: {error}") from error
    return AdaptiveSampler.from_dict(value)


def load_controlled_feedback(directory: str | Path) -> tuple[str, list[EvaluationRecord]]:
    root = Path(directory)
    manifest = read_controlled_manifest(root / "experiment.json")
    records = read_evaluation_jsonl(root / "evaluations.jsonl")
    expected = set(manifest["evaluation_ids"])
    actual = {record.evaluation_id.value for record in records}
    if expected != actual:
        raise ValueError("controlled manifest and evaluation JSONL identities do not match")
    return manifest["experiment_id"], records
