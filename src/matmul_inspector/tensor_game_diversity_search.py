"""Bounded CPU-only discovery of exact, structurally distinct TG decompositions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from typing import Callable, Sequence

from .tensor_game import RankOneAction, Tensor, matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition, subtract_action, zero_tensor
from .tensor_game_bounded_residual import action_from_payload, bounded_residual_candidates_memory_efficient, squared_norm
from .tensor_game_candidate_dataset import _record, canonical_actions, validate_decomposition

CandidateProvider = Callable[[Tensor, int], Sequence[RankOneAction]]


@dataclass(frozen=True)
class DiscoveryConfig:
    rank: int
    seed: int
    node_budget: int = 256
    beam_width: int = 128
    max_actions: int = 12
    mode: str = "baseline"
    paths_per_state: int = 1

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.node_budget <= 0 or self.beam_width <= 0 or self.max_actions <= 0:
            raise ValueError("rank and search budgets must be positive")
        if self.mode not in {"baseline", "diversity"}:
            raise ValueError("mode must be 'baseline' or 'diversity'")
        if self.paths_per_state <= 0:
            raise ValueError("paths_per_state must be positive")


def _prefix_id(actions: Sequence[RankOneAction]) -> str:
    value = [[list(a.u), list(a.v), list(a.w)] for a in canonical_actions(actions)]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def bounded_128_provider(residual: Tensor, limit: int) -> tuple[RankOneAction, ...]:
    rows, _ = bounded_residual_candidates_memory_efficient(residual, 128, limit)
    return tuple(action_from_payload(row["action"]) for row in rows)


def discover(config: DiscoveryConfig, provider: CandidateProvider = bounded_128_provider) -> dict:
    """Search every retained beam layer and collect all exact final trajectories."""
    started = time.perf_counter()
    target = matrix_multiplication_tensor(2, 2, 2)
    zero = zero_tensor((4, 4, 4))
    rng = random.Random(config.seed)
    frontier: list[tuple[Tensor, tuple[RankOneAction, ...]]] = [(target, ())]
    nodes_expanded = generated = duplicate_states = 0
    solutions: dict[str, tuple[RankOneAction, ...]] = {}
    depth_reached = 0
    exhausted = False
    for depth in range(config.rank):
        next_paths: dict[Tensor, dict[str, tuple[RankOneAction, ...]]] = {}
        for residual, path in frontier:
            if nodes_expanded >= config.node_budget:
                exhausted = True
                break
            nodes_expanded += 1
            if hasattr(provider, "set_context"):
                provider.set_context(depth, config.rank - depth)
            actions = tuple(provider(residual, config.max_actions))
            for action in actions:
                generated += 1
                candidate = path + (action,)
                child = subtract_action(residual, action)
                if child == zero:
                    if len(candidate) == config.rank:
                        validate_decomposition(candidate)
                        solutions.setdefault(_prefix_id(candidate), candidate)
                    continue
                if depth + 1 == config.rank:
                    continue
                key = _prefix_id(candidate)
                bucket = next_paths.setdefault(child, {})
                if key in bucket:
                    duplicate_states += 1
                elif len(bucket) < config.paths_per_state:
                    bucket[key] = candidate
                else:
                    duplicate_states += 1
        depth_reached = depth + 1
        if exhausted:
            break
        ranked = []
        for residual, paths in next_paths.items():
            score = squared_norm(residual)
            for key, path in paths.items():
                tie = rng.random() if config.mode == "diversity" else 0.0
                ranked.append((score, tie, residual, key, path))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        frontier = [(item[2], item[4]) for item in ranked[: config.beam_width]]
        if not frontier:
            break
    records = []
    provider_name = getattr(provider, "name", "bounded_128")
    known = {
        canonical_actions(standard_2x2_decomposition()): "conventional",
        canonical_actions(strassen_2x2_decomposition()): "strassen",
    }
    for index, actions in enumerate(solutions.values()):
        known_family = known.get(canonical_actions(actions))
        family = known_family or "independent_search"
        record = _record(f"search-{config.mode}-{config.seed}-{index}", family, actions, [])
        record.update({
            "discovery_method": f"{provider_name}_{config.mode}_beam_search",
            "search_seed": config.seed,
            "search_budget": {"nodes": config.node_budget, "beam_width": config.beam_width, "max_actions": config.max_actions},
            "parent_provenance": ({"relationship": "limited-canonical match", "known_seed": known_family} if known_family else None),
            "reproduction_config": asdict(config),
        })
        records.append(record)
    records.sort(key=lambda row: row["candidate_id"])
    return {
        "config": asdict(config),
        "success": bool(records),
        "solutions": records,
        "nodes_expanded": nodes_expanded,
        "candidate_edges_generated": generated,
        "duplicate_state_paths": duplicate_states,
        "depth_reached": depth_reached,
        "budget_exhausted": exhausted,
        "termination_reason": "exact_solution" if records else ("node_budget" if exhausted else "depth_limit_or_frontier_exhausted"),
        "wall_seconds": time.perf_counter() - started,
    }


class DiversityArchive:
    """Bounded exact archive. Novel signatures win deterministic retention ties."""

    def __init__(self, maximum: int = 64):
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        self.maximum = maximum
        self._records: dict[str, dict] = {}

    def add(self, record: dict) -> bool:
        actions = tuple(RankOneAction(tuple(a["u"]), tuple(a["v"]), tuple(a["w"])) for a in record.get("actions", ()))
        validate_decomposition(actions)
        rebuilt = _record(record.get("seed_id", "archive"), record.get("family_id", "independent_search"), actions, record.get("transformation_provenance", []))
        checked = ("candidate_id", "canonical_id", "structural_signature_id", "static_cost")
        if any(rebuilt[key] != record.get(key) for key in checked) or not record.get("exact_validation"):
            raise ValueError("record hash or exact-validation flag is invalid")
        if rebuilt["candidate_id"] in self._records:
            return False
        self._records[rebuilt["candidate_id"]] = dict(record)
        if len(self._records) > self.maximum:
            counts = Counter(row["structural_signature_id"] for row in self._records.values())
            victim = max(self._records.values(), key=lambda row: (counts[row["structural_signature_id"]], row["candidate_id"]))
            del self._records[victim["candidate_id"]]
        return rebuilt["candidate_id"] in self._records

    @property
    def records(self) -> tuple[dict, ...]:
        return tuple(sorted(self._records.values(), key=lambda row: row["candidate_id"]))
