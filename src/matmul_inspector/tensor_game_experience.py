"""Versioned MCTS decision collection and exact TensorGame replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable, Sequence, TypedDict

from .tensor_game import RankOneAction, Tensor, TensorGame, tensor_shape
from .tensor_game_search import ActionPolicy, LeafEvaluator, SampledMCTS, SearchConfig, canonicalize_action

SCHEMA_VERSION = 1
REWARD_CONVENTION = "edge_minus_one_terminal_success_max_moves_truncation_minus_max_moves_plus_one"


class DecisionRecord(TypedDict):
    schema_version: int
    experience_id: str
    episode_id: str
    search_config_id: str
    state_digest: str
    residual: list
    search_depth: int
    remaining_moves: int
    sampled_actions: list[dict[str, Any]]
    selected_action: dict[str, list[int]]
    selected_action_id: str
    behavior_probability: float
    behavior_rule: str
    model_provenance: dict[str, Any]
    simulations_permitted: int
    simulations_completed: int
    search_termination: str
    random_seed: int
    source_experiment_id: str
    next_state_digest: str
    step_reward: int
    terminated: bool
    truncated: bool
    observed_return_target: int | None
    return_target_semantics: str
    outcome_censored: bool


class EpisodeRecord(TypedDict):
    schema_version: int
    episode_id: str
    source_experiment_id: str
    configuration_id: str
    search_config_id: str
    initial_target_digest: str
    initial_target: list
    shape: list[int]
    coefficients: list[int]
    max_moves: int
    decision_ids: list[str]
    selected_actions: list[dict[str, list[int]]]
    final_residual_digest: str
    final_residual: list
    outcome: str
    stopping_reason: str
    total_action_reward: int
    verified_decomposition_length: int | None
    exact_reconstruction: bool
    model_provenance: dict[str, Any]
    seed: int
    collection_wall_seconds: float
    internal_simulations_are_episode_steps: bool


def tensor_payload(tensor: Tensor):
    return [[list(row) for row in plane] for plane in tensor]


def tensor_digest(tensor: Tensor) -> str:
    return hashlib.sha256(json.dumps(tensor, separators=(",", ":")).encode()).hexdigest()


def action_payload(action: RankOneAction):
    action = canonicalize_action(action)
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def action_from_payload(value) -> RankOneAction:
    return canonicalize_action(RankOneAction(value["u"], value["v"], value["w"]))


def action_identity(action: RankOneAction) -> str:
    return hashlib.sha256(json.dumps(action_payload(action), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stable_id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def search_policy(root_actions, temperature: float) -> tuple[dict[str, Any], ...]:
    """Normalize root visits over finite sampled support, with a prior fallback."""
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and nonnegative")
    if not root_actions:
        return ()
    visits = [item.visits for item in root_actions]
    if temperature == 0:
        best = max(range(len(root_actions)), key=lambda i: (visits[i], root_actions[i].prior, -i))
        probabilities = [1.0 if index == best else 0.0 for index in range(len(root_actions))]
        zero_visit_treatment = "argmax_visits_then_prior_then_order"
    elif any(visits):
        exponent = 1.0 / temperature
        weights = [float(visit) ** exponent for visit in visits]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        zero_visit_treatment = "zero_probability"
    else:
        total = sum(item.prior for item in root_actions)
        probabilities = [item.prior / total for item in root_actions]
        zero_visit_treatment = "normalized_original_prior_fallback"
    return tuple({
        "action": action_payload(item.action), "action_id": action_identity(item.action),
        "prior": item.prior, "visits": item.visits, "q_value": item.value,
        "search_probability": probabilities[index],
        "zero_visit_treatment": zero_visit_treatment,
    } for index, item in enumerate(root_actions))


def select_behavior_action(policy: Sequence[dict[str, Any]], rng: random.Random, temperature: float):
    if not policy:
        raise ValueError("cannot select from empty search support")
    if temperature == 0:
        index = max(range(len(policy)), key=lambda i: (policy[i]["search_probability"], -i))
        return policy[index], 1.0
    threshold = rng.random()
    cumulative = 0.0
    for item in policy:
        cumulative += item["search_probability"]
        if threshold <= cumulative:
            return item, item["search_probability"]
    return policy[-1], policy[-1]["search_probability"]


@dataclass(frozen=True)
class CollectionConfig:
    simulations: int = 40
    max_actions: int = 12
    c_puct: float = 1.5
    node_limit: int = 200
    wall_time_seconds: float = 5.0
    search_temperature: float = 1.0
    behavior_temperature: float = 0.0
    coefficients: tuple[int, ...] = (-1, 0, 1)


def play_episode(
    target: Tensor,
    max_moves: int,
    *,
    policy_factory: Callable[[], ActionPolicy],
    evaluator_factory: Callable[[], LeafEvaluator],
    config: CollectionConfig,
    seed: int,
    experiment_id: str,
    configuration_id: str,
    model_provenance: dict[str, Any],
    stop_after_decisions: int | None = None,
) -> tuple[EpisodeRecord, list[DecisionRecord]]:
    game = TensorGame(target, max_moves, frozenset(config.coefficients))
    rng = random.Random(seed)
    search_config_id = _stable_id({**asdict(config), "max_moves": max_moves, "configuration": configuration_id})
    episode_id = _stable_id([experiment_id, configuration_id, tensor_digest(target), max_moves, seed])
    decisions: list[DecisionRecord] = []
    started = time.perf_counter()
    interruption = None
    while not game.terminated and not game.truncated:
        if stop_after_decisions is not None and len(decisions) >= stop_after_decisions:
            interruption = "configured_stop_condition"
            break
        depth = len(game.history)
        residual_before = game.residual
        decision_seed = rng.randrange(2**63)
        result = SampledMCTS(
            residual_before, policy_factory(),
            SearchConfig(
                simulations=config.simulations, max_actions=config.max_actions,
                max_moves=max_moves - depth, seed=decision_seed, c_puct=config.c_puct,
                node_limit=config.node_limit, max_wall_time_seconds=config.wall_time_seconds,
            ), evaluator_factory(),
        ).run()
        improved = search_policy(result.root_actions, config.search_temperature)
        if not improved:
            interruption = "empty_sampled_support"
            break
        behavior_policy = search_policy(result.root_actions, config.behavior_temperature)
        chosen, behavior_probability = select_behavior_action(behavior_policy, rng, config.behavior_temperature)
        action = action_from_payload(chosen["action"])
        step = game.step(action)
        decision_core = {
            "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
            "search_config_id": search_config_id, "state_digest": tensor_digest(residual_before),
            "residual": tensor_payload(residual_before), "search_depth": depth,
            "remaining_moves": max_moves - depth, "sampled_actions": list(improved),
            "selected_action": action_payload(action), "selected_action_id": action_identity(action),
            "behavior_probability": behavior_probability,
            "behavior_rule": f"sample_search_policy_temperature_{config.behavior_temperature}",
            "model_provenance": model_provenance,
            "simulations_permitted": config.simulations, "simulations_completed": result.simulations_run,
            "search_termination": result.termination_reason, "random_seed": decision_seed,
            "source_experiment_id": experiment_id, "next_state_digest": tensor_digest(game.residual),
            "step_reward": step.reward, "terminated": step.terminated, "truncated": step.truncated,
        }
        decision_core["experience_id"] = _stable_id(decision_core)
        decisions.append(decision_core)
    if game.terminated:
        outcome = "success"
        stopping_reason = "exact_zero_residual"
    elif game.truncated:
        outcome = "truncated"
        stopping_reason = "move_limit_reached"
    else:
        outcome = "interrupted"
        stopping_reason = interruption or "infrastructure_interruption"
    terminal_base = max_moves if game.terminated else -(max_moves + 1)
    for index, decision in enumerate(decisions):
        future_edges = len(decisions) - index
        decision["observed_return_target"] = None if outcome == "interrupted" else terminal_base - future_edges
        decision["return_target_semantics"] = REWARD_CONVENTION
        decision["outcome_censored"] = outcome == "interrupted"
    episode = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
        "source_experiment_id": experiment_id, "configuration_id": configuration_id,
        "search_config_id": search_config_id, "initial_target_digest": tensor_digest(target),
        "initial_target": tensor_payload(target), "shape": list(tensor_shape(target)),
        "coefficients": list(config.coefficients), "max_moves": max_moves,
        "decision_ids": [row["experience_id"] for row in decisions],
        "selected_actions": [row["selected_action"] for row in decisions],
        "final_residual_digest": tensor_digest(game.residual), "final_residual": tensor_payload(game.residual),
        "outcome": outcome, "stopping_reason": stopping_reason,
        "total_action_reward": -len(decisions),
        "verified_decomposition_length": len(decisions) if game.terminated else None,
        "exact_reconstruction": game.terminated and game.reconstructs_target(),
        "model_provenance": model_provenance, "seed": seed,
        "collection_wall_seconds": time.perf_counter() - started,
        "internal_simulations_are_episode_steps": False,
    }
    return episode, decisions


class ReplayBuffer:
    def __init__(self, *, capacity_episodes: int | None = None):
        if capacity_episodes is not None and capacity_episodes <= 0:
            raise ValueError("capacity must be positive")
        self.capacity_episodes = capacity_episodes
        self.episodes: list[dict[str, Any]] = []
        self.decisions: dict[str, dict[str, Any]] = {}

    def add(self, episode, decisions):
        if any(item["episode_id"] == episode["episode_id"] for item in self.episodes):
            raise ValueError("duplicate episode ID")
        if len(set(episode["decision_ids"])) != len(episode["decision_ids"]):
            raise ValueError("duplicate decision reference")
        for row in decisions:
            if row["experience_id"] in self.decisions:
                raise ValueError("duplicate experience ID")
        self.episodes.append(episode)
        self.decisions.update((row["experience_id"], row) for row in decisions)
        while self.capacity_episodes is not None and len(self.episodes) > self.capacity_episodes:
            removed = self.episodes.pop(0)
            for identifier in removed["decision_ids"]:
                self.decisions.pop(identifier)

    def validate(self):
        reports = [replay_episode(episode, self.decisions) for episode in self.episodes]
        return {"valid": all(row["valid"] for row in reports), "episodes": reports}


def replay_episode(episode, decisions):
    target = tuple(tuple(tuple(row) for row in plane) for plane in episode["initial_target"])
    game = TensorGame(target, episode["max_moves"], frozenset(episode["coefficients"]))
    errors = []
    for identifier in episode["decision_ids"]:
        row = decisions.get(identifier)
        if row is None:
            errors.append(f"missing decision {identifier}")
            break
        identity_payload = {
            key: value for key, value in row.items()
            if key not in {"experience_id", "observed_return_target", "return_target_semantics", "outcome_censored"}
        }
        if _stable_id(identity_payload) != identifier:
            errors.append(f"experience identity mismatch {identifier}")
            break
        if row["episode_id"] != episode["episode_id"] or row["state_digest"] != tensor_digest(game.residual):
            errors.append(f"state/reference mismatch {identifier}")
            break
        probabilities = [item["search_probability"] for item in row["sampled_actions"]]
        if not probabilities or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            errors.append(f"invalid search policy {identifier}")
            break
        if len({item["action_id"] for item in row["sampled_actions"]}) != len(row["sampled_actions"]):
            errors.append(f"duplicate canonical action {identifier}")
            break
        step = game.step(action_from_payload(row["selected_action"]))
        if tensor_digest(game.residual) != row["next_state_digest"] or step.reward != row["step_reward"]:
            errors.append(f"transition mismatch {identifier}")
            break
    if tensor_digest(game.residual) != episode["final_residual_digest"]:
        errors.append("final residual mismatch")
    if -len(game.history) != episode["total_action_reward"]:
        errors.append("reward mismatch")
    exact = game.terminated and game.reconstructs_target()
    if exact != episode["exact_reconstruction"]:
        errors.append("exact reconstruction mismatch")
    return {"episode_id": episode["episode_id"], "valid": not errors, "errors": errors, "replayed_steps": len(game.history)}


def save_buffer(output: Path, buffer: ReplayBuffer, metadata: dict[str, Any]):
    output.mkdir(parents=True, exist_ok=False)
    validation = buffer.validate()
    if not validation["valid"]:
        raise ValueError("refusing to save invalid replay buffer")
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reward_convention": REWARD_CONVENTION, "capacity_episodes": buffer.capacity_episodes,
        "episode_count": len(buffer.episodes), "decision_count": len(buffer.decisions),
        "metadata": metadata, "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "episodes.json").write_text(json.dumps(buffer.episodes, indent=2, sort_keys=True) + "\n")
    (output / "decisions.json").write_text(json.dumps(list(buffer.decisions.values()), indent=2, sort_keys=True) + "\n")
    (output / "replay_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    outcomes = {name: sum(row["outcome"] == name for row in buffer.episodes) for name in ("success", "truncated", "interrupted")}
    entropies = []
    concentrations = []
    divergences = []
    for row in buffer.decisions.values():
        probabilities = [x["search_probability"] for x in row["sampled_actions"] if x["search_probability"] > 0]
        entropies.append(-sum(p * math.log(p) for p in probabilities))
        concentrations.append(max(probabilities))
        divergences.append(sum(
            item["search_probability"] * math.log(item["search_probability"] / item["prior"])
            for item in row["sampled_actions"] if item["search_probability"] > 0
        ))
    state_counts = {}
    for row in buffer.decisions.values():
        state_counts[row["state_digest"]] = state_counts.get(row["state_digest"], 0) + 1
    wall = sum(row["collection_wall_seconds"] for row in buffer.episodes)
    summary = [
        "# TensorGame experience collection", "",
        f"Episodes: {len(buffer.episodes)}. Decisions: {len(buffer.decisions)}.", "",
        f"Success: {outcomes['success']}. Truncated: {outcomes['truncated']}. Interrupted: {outcomes['interrupted']}.", "",
        f"Unique residual states: {len(state_counts)}. Repeated-state decisions: {sum(count - 1 for count in state_counts.values())}.",
        f"Mean sampled-support policy entropy: {sum(entropies)/len(entropies) if entropies else 0:.6f}.",
        f"Mean maximum search probability: {sum(concentrations)/len(concentrations) if concentrations else 0:.6f}.",
        f"Mean search-policy versus prior KL divergence: {sum(divergences)/len(divergences) if divergences else 0:.6f}.",
        f"Collection wall time: {wall:.6f} seconds. Throughput: {len(buffer.decisions)/wall if wall else 0:.3f} decisions/s.",
        "", "Visit distributions cover sampled root actions only and are not optimal-action probabilities.", "",
    ]
    (output / "report.md").write_text("\n".join(summary))


def load_buffer(output: Path) -> ReplayBuffer:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("reward_convention") != REWARD_CONVENTION:
        raise ValueError("unsupported replay buffer")
    buffer = ReplayBuffer(capacity_episodes=manifest["capacity_episodes"])
    decisions = {row["experience_id"]: row for row in json.loads((output / "decisions.json").read_text())}
    for episode in json.loads((output / "episodes.json").read_text()):
        buffer.add(episode, [decisions[i] for i in episode["decision_ids"]])
    if not buffer.validate()["valid"]:
        raise ValueError("corrupt replay buffer")
    return buffer
