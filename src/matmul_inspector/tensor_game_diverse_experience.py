"""Diverse, leakage-safe TensorGame MCTS experience collection and audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, matrix_multiplication_tensor, subtract_action, sum_actions, validate_action, zero_tensor
from .tensor_game_dataset import SyntheticDatasetConfig, action_to_dict, sample_action, tensor_to_list
from .tensor_game_experience import (
    CollectionConfig,
    REWARD_CONVENTION,
    ReplayBuffer,
    action_from_payload,
    action_identity,
    action_payload,
    load_buffer,
    play_episode,
    save_buffer,
    tensor_digest,
)
from .tensor_game_neural_mcts import ConstantValueEvaluator
from .tensor_game_search import HeuristicValueEvaluator, ResidualInformedPolicy

DIVERSE_COLLECTION_VERSION = 1
GENERATOR_VERSION = "tg4b2a.diverse_targets.v1"
SPLITS = ("train", "validation", "test")
SEARCH_CONFIGURATIONS = ("residual_heuristic", "residual_optimistic_max_moves")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class DiverseTargetConfig:
    target_count: int = 200
    construction_lengths: tuple[int, ...] = (1, 2, 3, 4)
    shape: tuple[int, int, int] = (4, 4, 4)
    coefficients: tuple[int, ...] = (-1, 0, 1)
    zero_probability: float = 0.5
    seed: int = 42
    max_generation_attempts: int = 100_000

    def __post_init__(self) -> None:
        if self.target_count <= 0 or self.target_count % len(self.construction_lengths):
            raise ValueError("target count must be positive and balanced across construction lengths")
        if self.shape != (4, 4, 4) or self.coefficients != (-1, 0, 1):
            raise ValueError("TG-4B.2A uses shape 4x4x4 and coefficients {-1,0,1}")
        if not 0 <= self.zero_probability < 1 or self.max_generation_attempts <= 0:
            raise ValueError("invalid generation settings")


@dataclass(frozen=True)
class DiverseCollectionConfig:
    seeds: tuple[int, ...] = (3, 11)
    simulations: int = 20
    proposals: int = 12
    node_limit: int = 200
    wall_time_limit: float = 2.0
    move_limit_extra: int = 0

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("collection seeds must be nonempty and unique")
        if min(self.simulations, self.proposals, self.node_limit) <= 0 or self.wall_time_limit <= 0:
            raise ValueError("collection budgets must be positive")
        if self.move_limit_extra < 0:
            raise ValueError("move-limit extension must be nonnegative")


def _balanced_split_schedule(lengths: Sequence[int], per_length: int) -> list[tuple[int, str]]:
    """Allocate 70/15/15 globally and differ the rounding side by length."""
    total = len(lengths) * per_length
    desired = {"train": round(total * 0.70), "validation": round(total * 0.15)}
    desired["test"] = total - desired["train"] - desired["validation"]
    counts = {length: Counter() for length in lengths}
    remaining = {length: per_length for length in lengths}
    offset = 0
    for split in SPLITS:
        assigned = 0
        while assigned < desired[split]:
            length = lengths[offset % len(lengths)]
            offset += 1
            if remaining[length] == 0:
                continue
            counts[length][split] += 1
            remaining[length] -= 1
            assigned += 1
    return [
        (length, split)
        for length in lengths
        for split in SPLITS
        for _ in range(counts[length][split])
    ]


def generate_diverse_targets(config: DiverseTargetConfig) -> dict[str, Any]:
    """Generate exact unique targets with collision-free known trajectories."""
    rng = random.Random(config.seed)
    per_length = config.target_count // len(config.construction_lengths)
    slots = _balanced_split_schedule(config.construction_lengths, per_length)
    rng.shuffle(slots)
    sampling = SyntheticDatasetConfig(
        shape=config.shape,
        construction_lengths=config.construction_lengths,
        decompositions_per_length=1,
        coefficients=config.coefficients,
        zero_probability=config.zero_probability,
        seed=config.seed,
    )
    reserved_digest = tensor_digest(matrix_multiplication_tensor(2, 2, 2))
    targets = []
    target_digests: set[str] = set()
    residual_owner: dict[str, str] = {}
    factor_multisets: set[str] = set()
    statistics = {
        "attempts": 0,
        "accepted_targets": 0,
        "rejected_zero_targets": 0,
        "rejected_duplicate_targets": 0,
        "rejected_reserved_targets": 0,
        "rejected_cross_target_residual_collisions": 0,
        "rejected_duplicate_factor_multisets": 0,
    }
    for index, (length, split) in enumerate(slots):
        while True:
            statistics["attempts"] += 1
            if statistics["attempts"] > config.max_generation_attempts:
                raise RuntimeError("diverse target generation exhausted its attempt limit")
            origin_seed = rng.randrange(2**63)
            local_rng = random.Random(origin_seed)
            factors = tuple(sample_action(sampling, local_rng) for _ in range(length))
            target = sum_actions(factors, config.shape)
            if target == zero_tensor(config.shape):
                statistics["rejected_zero_targets"] += 1
                continue
            digest = tensor_digest(target)
            if digest == reserved_digest:
                statistics["rejected_reserved_targets"] += 1
                continue
            if digest in target_digests:
                statistics["rejected_duplicate_targets"] += 1
                continue
            multiset_digest = _digest(sorted(action_identity(action) for action in factors))
            if multiset_digest in factor_multisets:
                statistics["rejected_duplicate_factor_multisets"] += 1
                continue
            residual = target
            residual_digests = []
            for action in factors:
                residual_digests.append(tensor_digest(residual))
                residual = subtract_action(residual, action)
            if residual != zero_tensor(config.shape):
                raise AssertionError("generated factors did not reconstruct their target")
            if any(state in residual_owner for state in residual_digests):
                statistics["rejected_cross_target_residual_collisions"] += 1
                continue
            target_id = f"tg4b2a-t{index:04d}-{digest[:12]}"
            record = {
                "target_id": target_id,
                "target": tensor_to_list(target),
                "target_digest": digest,
                "content_digest": _digest(tensor_to_list(target)),
                "construction_length": length,
                "construction_length_semantics": "verified_upper_bound_not_proven_tensor_rank",
                "split": split,
                "origin_seed": origin_seed,
                "generator_seed": config.seed,
                "generator_version": GENERATOR_VERSION,
                "shape": list(config.shape),
                "coefficients": list(config.coefficients),
                "canonical_factors": [action_to_dict(action) for action in factors],
                "canonical_factor_multiset_digest": multiset_digest,
                "known_trajectory_residual_digests": residual_digests,
                "reference_target": False,
            }
            targets.append(record)
            target_digests.add(digest)
            factor_multisets.add(multiset_digest)
            residual_owner.update((state, target_id) for state in residual_digests)
            break
    statistics["accepted_targets"] = len(targets)
    manifest = {
        "diverse_collection_version": DIVERSE_COLLECTION_VERSION,
        "generator_version": GENERATOR_VERSION,
        "config": asdict(config),
        "curriculum_id": f"tg4b2a-targets-{_digest([asdict(config), targets])[:16]}",
        "reserved_evaluation_target_digest": reserved_digest,
        "targets": targets,
        "target_splits": {row["target_digest"]: row["split"] for row in targets},
        "statistics": statistics,
    }
    validate_target_curriculum(manifest)
    return manifest


def validate_target_curriculum(curriculum: dict[str, Any]) -> None:
    if curriculum.get("diverse_collection_version") != DIVERSE_COLLECTION_VERSION:
        raise ValueError("unsupported diverse target curriculum")
    config = DiverseTargetConfig(**{
        **curriculum["config"],
        "construction_lengths": tuple(curriculum["config"]["construction_lengths"]),
        "shape": tuple(curriculum["config"]["shape"]),
        "coefficients": tuple(curriculum["config"]["coefficients"]),
    })
    targets = curriculum["targets"]
    if len(targets) != config.target_count:
        raise ValueError("target curriculum count mismatch")
    digests = [row["target_digest"] for row in targets]
    if len(set(digests)) != len(digests) or curriculum["reserved_evaluation_target_digest"] in digests:
        raise ValueError("duplicate or reserved target in curriculum")
    states = []
    for row in targets:
        target = tuple(tuple(tuple(line) for line in plane) for plane in row["target"])
        factors = tuple(RankOneAction(value["u"], value["v"], value["w"]) for value in row["canonical_factors"])
        if sum_actions(factors, config.shape) != target or tensor_digest(target) != row["target_digest"]:
            raise ValueError("target reconstruction or digest mismatch")
        states.extend(row["known_trajectory_residual_digests"])
    if len(states) != len(set(states)):
        raise ValueError("known trajectory residual collision across targets")
    counts = Counter((row["construction_length"], row["split"]) for row in targets)
    per_length = config.target_count // len(config.construction_lengths)
    for length in config.construction_lengths:
        if sum(counts[length, split] for split in SPLITS) != per_length:
            raise ValueError("construction-length curriculum is imbalanced")
    split_counts = Counter(row["split"] for row in targets)
    expected_train = round(config.target_count * 0.70)
    expected_validation = round(config.target_count * 0.15)
    if split_counts != Counter(train=expected_train, validation=expected_validation, test=config.target_count - expected_train - expected_validation):
        raise ValueError("target partition counts do not match 70/15/15")


def planned_episode_schedule(
    curriculum: dict[str, Any],
    collection: DiverseCollectionConfig,
    stage: str,
) -> list[dict[str, Any]]:
    if stage not in {"smoke", "pilot", "full"}:
        raise ValueError("stage must be smoke, pilot, or full")
    grouped = defaultdict(list)
    for target in sorted(curriculum["targets"], key=lambda row: row["target_id"]):
        grouped[target["construction_length"]].append(target)
    if stage == "smoke":
        selected = [grouped[length][0] for length in sorted(grouped)]
        seeds = collection.seeds[:1]
    elif stage == "pilot":
        allocation = {length: 2 + int(index < 2) for index, length in enumerate(sorted(grouped))}
        selected = [target for length in sorted(grouped) for target in grouped[length][:allocation[length]]]
        seeds = collection.seeds
    else:
        selected = list(curriculum["targets"])
        seeds = collection.seeds
    return [
        {
            "target_id": target["target_id"],
            "target_digest": target["target_digest"],
            "split": target["split"],
            "construction_length": target["construction_length"],
            "seed": seed,
            "configuration_id": configuration,
        }
        for target in selected
        for seed in seeds
        for configuration in SEARCH_CONFIGURATIONS
    ]


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "mean": None, "maximum": None}
    return {"minimum": min(values), "mean": sum(values) / len(values), "maximum": max(values)}


def audit_buffer(
    buffer: ReplayBuffer,
    curriculum: dict[str, Any],
    collection_wall_seconds: float,
    initial_resident_set_bytes: int = 0,
) -> dict[str, Any]:
    targets = {row["target_digest"]: row for row in curriculum["targets"]}
    outcomes = Counter(episode["outcome"] for episode in buffer.episodes)
    outcome_by_length: dict[str, Counter] = defaultdict(Counter)
    outcome_by_configuration: dict[str, Counter] = defaultdict(Counter)
    state_counts = Counter(row["state_digest"] for row in buffer.decisions.values())
    state_targets: dict[str, set[str]] = defaultdict(set)
    state_splits: dict[str, set[str]] = defaultdict(set)
    action_counts = Counter()
    supports = []
    entropies = []
    concentrations = []
    divergences = []
    simulations = []
    return_targets = []
    policy_eligible = 0
    value_eligible = 0
    invalid_action_records = 0
    invalid_policy_targets = 0
    invalid_value_targets = 0
    episode_by_id = {episode["episode_id"]: episode for episode in buffer.episodes}
    target_by_episode = {
        episode["episode_id"]: targets[episode["initial_target_digest"]]
        for episode in buffer.episodes
    }
    for episode in buffer.episodes:
        target = targets[episode["initial_target_digest"]]
        outcome_by_length[str(target["construction_length"])][episode["outcome"]] += 1
        outcome_by_configuration[episode["configuration_id"].split(":", 1)[0]][episode["outcome"]] += 1
    for row in buffer.decisions.values():
        target = target_by_episode[row["episode_id"]]
        state_targets[row["state_digest"]].add(target["target_digest"])
        state_splits[row["state_digest"]].add(target["split"])
        support = row["sampled_actions"]
        probabilities = [item["search_probability"] for item in support]
        valid_actions = True
        for item in support:
            try:
                action = action_from_payload(item["action"])
                validate_action(action, (4, 4, 4), frozenset((-1, 0, 1)))
                if action_payload(action) != item["action"] or action_identity(action) != item["action_id"]:
                    valid_actions = False
            except (KeyError, TypeError, ValueError):
                valid_actions = False
        valid_policy = (
            bool(support)
            and len({item["action_id"] for item in support}) == len(support)
            and all(math.isfinite(value) and value >= 0 for value in probabilities)
            and math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
            and valid_actions
        )
        if valid_policy:
            policy_eligible += 1
        else:
            invalid_policy_targets += 1
        if not valid_actions:
            invalid_action_records += 1
        supports.append(len(support))
        positives = [value for value in probabilities if value > 0]
        entropies.append(-sum(value * math.log(value) for value in positives))
        concentrations.append(max(probabilities))
        divergences.append(sum(
            item["search_probability"] * math.log(item["search_probability"] / item["prior"])
            for item in support if item["search_probability"] > 0
        ))
        for item in support:
            action_counts[item["action_id"]] += 1
        simulations.append(row["simulations_completed"])
        episode = episode_by_id[row["episode_id"]]
        decision_index = episode["decision_ids"].index(row["experience_id"])
        expected_return = None if episode["outcome"] == "interrupted" else (
            (episode["max_moves"] if episode["outcome"] == "success" else -(episode["max_moves"] + 1))
            - (len(episode["decision_ids"]) - decision_index)
        )
        valid_value = (
            row["outcome_censored"] == (episode["outcome"] == "interrupted")
            and row["observed_return_target"] == expected_return
            and row["return_target_semantics"] == REWARD_CONVENTION
        )
        if not valid_value:
            invalid_value_targets += 1
        if valid_value and not row["outcome_censored"] and row["observed_return_target"] is not None:
            value_eligible += 1
            return_targets.append(row["observed_return_target"])
    episode_times = [row["collection_wall_seconds"] for row in buffer.episodes]
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        memory *= 1024
    split_targets = Counter(row["split"] for row in curriculum["targets"])
    length_targets = Counter(str(row["construction_length"]) for row in curriculum["targets"])
    def outcome_summary(values: Counter) -> dict[str, Any]:
        total = sum(values.values())
        return {
            "counts": {name: values[name] for name in ("success", "truncated", "interrupted")},
            "rates": {name: values[name] / total if total else 0.0 for name in ("success", "truncated", "interrupted")},
        }
    return {
        "audit_version": DIVERSE_COLLECTION_VERSION,
        "target_diversity": {
            "originating_targets": len(curriculum["targets"]),
            "unique_exact_targets": len({row["target_digest"] for row in curriculum["targets"]}),
            "targets_by_construction_length": dict(sorted(length_targets.items())),
            "targets_by_partition": dict(split_targets),
            "generation_statistics": curriculum["statistics"],
        },
        "state_diversity": {
            "played_decisions": len(buffer.decisions),
            "unique_residuals": len(state_counts),
            "repeated_state_decisions": sum(count - 1 for count in state_counts.values()),
            "frequency_distribution": dict(sorted(Counter(state_counts.values()).items())),
            "cross_target_residual_overlap": sum(len(values) > 1 for values in state_targets.values()),
            "cross_split_residual_overlap": sum(len(values) > 1 for values in state_splits.values()),
        },
        "action_diversity": {
            "unique_canonical_actions": len(action_counts),
            "frequency_distribution": dict(sorted(Counter(action_counts.values()).items())),
            "support_size": _percentiles(supports),
            "visit_distribution_entropy": _percentiles(entropies),
            "maximum_search_probability": _percentiles(concentrations),
            "search_policy_prior_kl": _percentiles(divergences),
        },
        "outcomes": {
            "counts": {name: outcomes[name] for name in ("success", "truncated", "interrupted")},
            "rates": outcome_summary(outcomes)["rates"],
            "by_construction_length": {key: outcome_summary(value) for key, value in sorted(outcome_by_length.items())},
            "by_search_configuration": {key: outcome_summary(value) for key, value in sorted(outcome_by_configuration.items())},
            "return_targets": {**_percentiles(return_targets), "count": len(return_targets), "frequency": dict(sorted(Counter(return_targets).items()))},
        },
        "training_readiness": {
            "eligible_policy_decisions": policy_eligible,
            "eligible_value_decisions": value_eligible,
            "censored_value_decisions": len(buffer.decisions) - value_eligible,
            "invalid_policy_targets": invalid_policy_targets,
            "invalid_action_records": invalid_action_records,
            "invalid_value_targets": invalid_value_targets,
            "independent_target_groups_collected": len({episode["initial_target_digest"] for episode in buffer.episodes}),
            "effective_unique_training_states": len({
                row["state_digest"] for row in buffer.decisions.values()
                if target_by_episode[row["episode_id"]]["split"] == "train"
            }),
            "reward_convention": REWARD_CONVENTION,
            "shape": [4, 4, 4],
            "coefficients": [-1, 0, 1],
        },
        "cost": {
            "collection_wall_seconds": collection_wall_seconds,
            "episodes_per_second": len(buffer.episodes) / collection_wall_seconds if collection_wall_seconds else 0.0,
            "decisions_per_second": len(buffer.decisions) / collection_wall_seconds if collection_wall_seconds else 0.0,
            "simulations_per_decision": _percentiles(simulations),
            "episode_seconds": _percentiles(episode_times),
            "maximum_resident_set_bytes": memory,
            "maximum_resident_set_growth_bytes": max(0, memory - initial_resident_set_bytes),
        },
        "replay_validation": buffer.validate(),
    }


def collect_diverse_experience(
    curriculum: dict[str, Any],
    collection: DiverseCollectionConfig,
    stage: str,
    output: Path,
    *,
    episode_limit: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    schedule = planned_episode_schedule(curriculum, collection, stage)
    if episode_limit is not None:
        if episode_limit < 0:
            raise ValueError("episode limit must be nonnegative")
        schedule = schedule[:episode_limit]
    by_id = {row["target_id"]: row for row in curriculum["targets"]}
    experiment_id = f"tg4b2a-{stage}-{curriculum['curriculum_id']}"
    search = CollectionConfig(
        simulations=collection.simulations,
        max_actions=collection.proposals,
        node_limit=collection.node_limit,
        wall_time_seconds=collection.wall_time_limit,
        coefficients=(-1, 0, 1),
    )
    buffer = ReplayBuffer(capacity_episodes=max(1, len(schedule)))
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        initial_memory *= 1024
    started = time.perf_counter()
    for item in schedule:
        target_record = by_id[item["target_id"]]
        target = tuple(tuple(tuple(line) for line in plane) for plane in target_record["target"])
        max_moves = target_record["construction_length"] + collection.move_limit_extra
        optimistic = item["configuration_id"] == "residual_optimistic_max_moves"
        episode, decisions = play_episode(
            target,
            max_moves,
            policy_factory=lambda: ResidualInformedPolicy((4, 4, 4), (-1, 0, 1)),
            evaluator_factory=(lambda moves=max_moves: ConstantValueEvaluator(float(moves))) if optimistic else HeuristicValueEvaluator,
            config=search,
            seed=item["seed"],
            experiment_id=experiment_id,
            configuration_id=f"{item['configuration_id']}:{item['target_id']}",
            model_provenance={
                "policy": "residual_informed",
                "value": "constant_max_moves" if optimistic else "nonzero_heuristic",
                "trained_model_dependency": False,
                "target_id": item["target_id"],
                "target_split": item["split"],
                "construction_length_upper_bound": item["construction_length"],
            },
            episode_wall_time_seconds=collection.wall_time_limit,
        )
        buffer.add(episode, decisions)
    wall = time.perf_counter() - started
    audit = audit_buffer(buffer, curriculum, wall, initial_memory)
    metadata = {
        "diverse_collection_version": DIVERSE_COLLECTION_VERSION,
        "generator_version": GENERATOR_VERSION,
        "curriculum_id": curriculum["curriculum_id"],
        "target_generator_config": curriculum["config"],
        "target_generation_statistics": curriculum["statistics"],
        "target_splits": curriculum["target_splits"],
        "target_records": curriculum["targets"],
        "stage": stage,
        "planned_episode_count": len(planned_episode_schedule(curriculum, collection, stage)),
        "executed_episode_count": len(buffer.episodes),
        "interrupted_collection": len(buffer.episodes) < len(planned_episode_schedule(curriculum, collection, stage)),
        "schedule": schedule,
        "collection_config": asdict(collection),
        "search_configurations": list(SEARCH_CONFIGURATIONS),
        "move_limit_policy": "construction_length_upper_bound_plus_configured_extra",
        "software": {"python": platform.python_version(), "platform": platform.platform()},
    }
    save_buffer(output, buffer, metadata)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    evidence_size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    audit["cost"]["serialized_evidence_bytes"] = evidence_size
    audit["cost"]["serialized_bytes_per_episode"] = evidence_size / len(buffer.episodes) if buffer.episodes else 0.0
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _write_report(output / "report.md", metadata, audit)
    validate_diverse_experiment(output)
    return audit


def _write_report(path: Path, metadata: dict[str, Any], audit: dict[str, Any]) -> None:
    targets = audit["target_diversity"]
    states = audit["state_diversity"]
    actions = audit["action_diversity"]
    outcomes = audit["outcomes"]["counts"]
    readiness = audit["training_readiness"]
    cost = audit["cost"]
    lines = [
        "# TG-4B.2A diverse MCTS experience", "",
        f"Stage: `{metadata['stage']}`. Executed {metadata['executed_episode_count']} of {metadata['planned_episode_count']} planned episodes.",
        f"Search used {metadata['collection_config']['simulations']} simulations, {metadata['collection_config']['proposals']} proposals, and a {metadata['collection_config']['wall_time_limit']:.3f} second per-episode limit.", "",
        "## Correctness and outcomes", "",
        f"Exact replay valid: **{audit['replay_validation']['valid']}**.",
        f"Success: {outcomes['success']}. Truncated: {outcomes['truncated']}. Interrupted: {outcomes['interrupted']}.",
        "A truncated search is bounded negative experience. It is not evidence that a decomposition does not exist.", "",
        "## Diversity and split isolation", "",
        f"The curriculum contains {targets['originating_targets']} unique targets. The executed stage covers {readiness['independent_target_groups_collected']} independent targets.",
        f"Played decisions: {states['played_decisions']}. Unique residuals: {states['unique_residuals']}. Repeated-state decisions: {states['repeated_state_decisions']}.",
        f"Cross-target residual overlaps: {states['cross_target_residual_overlap']}. Cross-split residual overlaps: {states['cross_split_residual_overlap']}.",
        f"Unique canonical sampled actions: {actions['unique_canonical_actions']}. Mean support size: {actions['support_size']['mean']:.3f}.", "",
        "## Offline-learning readiness", "",
        f"Eligible policy decisions: {readiness['eligible_policy_decisions']}. Eligible value decisions: {readiness['eligible_value_decisions']}.",
        f"Effective unique training states in this stage: {readiness['effective_unique_training_states']}.",
        "Targets were partitioned before MCTS. Decisions from one target remain in that target's partition.", "",
        "## Cost and limitations", "",
        f"Collection wall time: {cost['collection_wall_seconds']:.6f} seconds. Episodes/s: {cost['episodes_per_second']:.3f}. Decisions/s: {cost['decisions_per_second']:.3f}.",
        f"Serialized evidence: {cost['serialized_evidence_bytes']} bytes.",
        "Episodes from the same target and repeated residuals are correlated evidence. They must not be treated as independent observations.",
        "This collection audit does not train a model or demonstrate improved discovery.", "",
        "## Recommendation", "",
        "Use the pilot to check outcome imbalance and state coverage. Run the full collection only if its projected runtime and storage remain acceptable. TG-4B.2B should train ablations without accessing held-out test targets.", "",
    ]
    path.write_text("\n".join(lines))


def validate_diverse_experiment(output: Path) -> dict[str, Any]:
    required = {"manifest.json", "episodes.json", "decisions.json", "replay_validation.json", "dataset_audit.json", "report.md"}
    if not required.issubset(path.name for path in output.iterdir()):
        raise ValueError("diverse experiment is missing required evidence")
    manifest = json.loads((output / "manifest.json").read_text())
    metadata = manifest.get("metadata", {})
    if metadata.get("diverse_collection_version") != DIVERSE_COLLECTION_VERSION:
        raise ValueError("unsupported diverse collection manifest")
    buffer = load_buffer(output)
    audit = json.loads((output / "dataset_audit.json").read_text())
    if not buffer.validate()["valid"] or not audit.get("replay_validation", {}).get("valid"):
        raise ValueError("diverse experience failed replay")
    if manifest["episode_count"] != len(buffer.episodes) or manifest["decision_count"] != len(buffer.decisions):
        raise ValueError("diverse manifest counts do not match replay")
    if audit["state_diversity"]["played_decisions"] != len(buffer.decisions):
        raise ValueError("dataset audit decision count mismatch")
    if audit["state_diversity"]["cross_split_residual_overlap"]:
        raise ValueError("residual state leaked across target partitions")
    readiness = audit["training_readiness"]
    if any(readiness[key] for key in ("invalid_policy_targets", "invalid_action_records", "invalid_value_targets")):
        raise ValueError("diverse experience contains invalid learning targets")
    return {"valid": True, "episodes": len(buffer.episodes), "decisions": len(buffer.decisions)}
