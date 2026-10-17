"""Bounded CPU-only discovery of exact, structurally distinct TG decompositions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from typing import Callable, Literal, Sequence

from .tensor_game import RankOneAction, Tensor, matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition, subtract_action, zero_tensor
from .tensor_game_bounded_residual import action_from_payload, action_identity, bounded_residual_candidates_memory_efficient, squared_norm
from .tensor_game_candidate_dataset import _record, canonical_actions, validate_decomposition
from .tensor_game_rank_feasibility import flattening_rank_lower_bound

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
    admission_policy: Literal["global", "provenance_quota"] = "global"
    baseline_capacity: int | None = None
    alternative_capacity: int = 0
    feasibility_gate: Literal["off", "flattening_rank"] = "off"
    feasibility_activation_budget: int = 3

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.node_budget <= 0 or self.beam_width <= 0 or self.max_actions <= 0:
            raise ValueError("rank and search budgets must be positive")
        if self.mode not in {"baseline", "diversity"}:
            raise ValueError("mode must be 'baseline' or 'diversity'")
        if self.paths_per_state <= 0:
            raise ValueError("paths_per_state must be positive")
        if self.admission_policy not in {"global", "provenance_quota"}:
            raise ValueError("admission_policy must be 'global' or 'provenance_quota'")
        if self.feasibility_gate not in {"off", "flattening_rank"}:
            raise ValueError("feasibility_gate must be 'off' or 'flattening_rank'")
        if self.feasibility_activation_budget < 0:
            raise ValueError("feasibility_activation_budget must be nonnegative")
        if self.admission_policy == "provenance_quota":
            baseline = self.beam_width if self.baseline_capacity is None else self.baseline_capacity
            if baseline < 0 or self.alternative_capacity < 0 or baseline + self.alternative_capacity != self.beam_width:
                raise ValueError("provenance capacities must be nonnegative and sum to beam_width")


def _prefix_id(actions: Sequence[RankOneAction]) -> str:
    value = [[list(a.u), list(a.v), list(a.w)] for a in canonical_actions(actions)]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def bounded_128_provider(residual: Tensor, limit: int) -> tuple[RankOneAction, ...]:
    rows, _ = bounded_residual_candidates_memory_efficient(residual, 128, limit)
    return tuple(action_from_payload(row["action"]) for row in rows)


def feasibility_gate_decision(
    residual: Tensor,
    remaining_budget: int,
    *,
    gate: Literal["off", "flattening_rank"] = "off",
    activation_budget: int = 3,
) -> dict:
    """Return an exact optional prune decision without scoring the residual."""
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be nonnegative")
    if activation_budget < 0:
        raise ValueError("activation_budget must be nonnegative")
    active = gate == "flattening_rank" and remaining_budget <= activation_budget
    if gate not in {"off", "flattening_rank"}:
        raise ValueError("gate must be 'off' or 'flattening_rank'")
    if not active:
        return {
            "active": False,
            "remaining_budget": remaining_budget,
            "flattening_ranks": None,
            "lower_bound": None,
            "prune": False,
        }
    bound = flattening_rank_lower_bound(residual)
    return {
        "active": True,
        "remaining_budget": remaining_budget,
        "flattening_ranks": bound["flattening_ranks"],
        "lower_bound": bound["lower_bound"],
        "prune": bound["lower_bound"] > remaining_budget,
    }


def _admit_ranked(config: DiscoveryConfig, ranked):
    if config.admission_policy == "global":
        return ranked[: config.beam_width], 0
    baseline_capacity = config.beam_width if config.baseline_capacity is None else config.baseline_capacity
    baseline = [item for item in ranked if item[5] == "baseline"]
    alternative = [item for item in ranked if item[5] == "alternative"]
    admitted = baseline[:baseline_capacity] + alternative[:config.alternative_capacity]
    admitted_ids = {(item[2], item[3]) for item in admitted}
    remaining = [item for item in ranked if (item[2], item[3]) not in admitted_ids]
    admitted += remaining[: config.beam_width - len(admitted)]
    admitted.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    capacities = {"baseline": baseline_capacity, "alternative": config.alternative_capacity}
    eligible_by = Counter(item[5] for item in ranked)
    unused = sum(
        max(0, capacities[name] - min(capacities[name], eligible_by[name]))
        for name in capacities
    )
    return admitted, unused


def discover(
    config: DiscoveryConfig,
    provider: CandidateProvider = bounded_128_provider,
    *,
    diagnostic_trajectories: dict[str, Sequence[RankOneAction]] | None = None,
) -> dict:
    """Search every retained beam layer and collect all exact final trajectories."""
    started = time.perf_counter()
    target = matrix_multiplication_tensor(2, 2, 2)
    zero = zero_tensor((4, 4, 4))
    rng = random.Random(config.seed)
    frontier: list[tuple[Tensor, tuple[RankOneAction, ...]]] = [(target, ())]
    nodes_expanded = generated = duplicate_states = provider_calls = 0
    feasibility_checked = feasibility_pruned = feasibility_ns = 0
    solutions: dict[str, tuple[RankOneAction, ...]] = {}
    depth_reached = 0
    exhausted = False
    depth_diagnostics = []
    trajectory_rows = []
    trajectory_states = {}
    for name, actions in (diagnostic_trajectories or {}).items():
        residual = target
        states = [residual]
        for action in actions:
            residual = subtract_action(residual, action)
            states.append(residual)
        trajectory_states[name] = (tuple(actions), states)
    for depth in range(config.rank):
        # Each state owns one deterministic path under the normal configuration.
        # The provenance set records collisions without consuming another slot.
        next_paths: dict[Tensor, dict[str, tuple[tuple[RankOneAction, ...], str]]] = {}
        pruned_paths: dict[Tensor, dict[str, tuple[tuple[RankOneAction, ...], str]]] = {}
        state_insertion_order: list[Tensor] = []
        provenance_by_state: dict[Tensor, set[str]] = {}
        generated_by = Counter(); valid_by = Counter(); duplicate_by = Counter()
        invalid_by = Counter(); provider_calls_before = getattr(provider, "calls", 0)
        expanded_this_depth = 0
        checked_this_depth = pruned_this_depth = check_ns_this_depth = 0
        rank_histogram = Counter()
        trajectory_stage = {name: {
            "trajectory": name,
            "depth": depth + 1,
            "parent_state_present": any(residual == states[depth] for residual, _ in frontier) if depth < len(actions) else False,
            "candidate_generated": False,
            "child_valid": False,
            "child_feasibility_pruned": False,
            "child_survived_deduplication": False,
            "child_admitted": False,
        } for name, (actions, states) in trajectory_states.items()}
        for residual, path in frontier:
            if nodes_expanded >= config.node_budget:
                exhausted = True
                break
            nodes_expanded += 1
            expanded_this_depth += 1
            provider_calls += 1
            if hasattr(provider, "set_context"):
                provider.set_context(depth, config.rank - depth)
            if hasattr(provider, "propose_with_provenance"):
                proposals = tuple(provider.propose_with_provenance(residual, config.max_actions))
            else:
                proposals = tuple((action, "baseline") for action in provider(residual, config.max_actions))
            for proposal in proposals:
                if hasattr(proposal, "action") and hasattr(proposal, "provenance"):
                    action, provenance = proposal.action, proposal.provenance
                else:
                    action, provenance = proposal
                generated_by[provenance] += 1
                for name, (known_actions, states) in trajectory_states.items():
                    if depth < len(known_actions) and residual == states[depth] and action_identity(action) == action_identity(known_actions[depth]):
                        trajectory_stage[name]["candidate_generated"] = True
                generated += 1
                candidate = path + (action,)
                try:
                    child = subtract_action(residual, action)
                except (TypeError, ValueError):
                    invalid_by[provenance] += 1
                    continue
                valid_by[provenance] += 1
                for name, (known_actions, states) in trajectory_states.items():
                    if depth < len(known_actions) and residual == states[depth] and action_identity(action) == action_identity(known_actions[depth]) and child == states[depth + 1]:
                        trajectory_stage[name]["child_valid"] = True
                if child == zero:
                    if len(candidate) == config.rank:
                        validate_decomposition(candidate)
                        solutions.setdefault(_prefix_id(candidate), candidate)
                    continue
                if depth + 1 == config.rank:
                    continue
                key = _prefix_id(candidate)
                remaining_budget = config.rank - (depth + 1)
                if remaining_budget < 0:
                    raise AssertionError("post-action remaining budget cannot be negative")
                gate_started = time.perf_counter_ns()
                decision = feasibility_gate_decision(
                    child,
                    remaining_budget,
                    gate=config.feasibility_gate,
                    activation_budget=config.feasibility_activation_budget,
                )
                gate_elapsed = time.perf_counter_ns() - gate_started
                if decision["active"]:
                    feasibility_checked += 1
                    checked_this_depth += 1
                    feasibility_ns += gate_elapsed
                    check_ns_this_depth += gate_elapsed
                    rank_histogram[decision["lower_bound"]] += 1
                if decision["prune"]:
                    feasibility_pruned += 1
                    pruned_this_depth += 1
                    if child not in next_paths and child not in pruned_paths:
                        state_insertion_order.append(child)
                    bucket = pruned_paths.setdefault(child, {})
                    if key not in bucket and len(bucket) < config.paths_per_state:
                        bucket[key] = (candidate, provenance)
                    for name, (known_actions, states) in trajectory_states.items():
                        if (
                            depth < len(known_actions)
                            and residual == states[depth]
                            and action_identity(action) == action_identity(known_actions[depth])
                            and child == states[depth + 1]
                        ):
                            trajectory_stage[name]["child_feasibility_pruned"] = True
                    continue
                if child not in next_paths and child not in pruned_paths:
                    state_insertion_order.append(child)
                bucket = next_paths.setdefault(child, {})
                provenance_by_state.setdefault(child, set()).add(provenance)
                if key in bucket:
                    duplicate_states += 1
                    duplicate_by[provenance] += 1
                elif len(bucket) < config.paths_per_state:
                    bucket[key] = (candidate, provenance)
                else:
                    duplicate_states += 1
                    duplicate_by[provenance] += 1
        depth_reached = depth + 1
        ranked = []
        counterfactual_pruned_ranked = []
        for residual in state_insertion_order:
            paths = next_paths.get(residual, pruned_paths.get(residual, {}))
            score = squared_norm(residual)
            for key, (path, provenance) in paths.items():
                tie = rng.random() if config.mode == "diversity" else 0.0
                item = (score, tie, residual, key, path, provenance)
                if residual in pruned_paths:
                    counterfactual_pruned_ranked.append(item)
                else:
                    ranked.append(item)
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for name, (known_actions, states) in trajectory_states.items():
            if depth < len(known_actions):
                trajectory_stage[name]["child_survived_deduplication"] = states[depth + 1] in next_paths
        admitted, unused = _admit_ranked(config, ranked)
        counterfactual_ranked = list(ranked) + counterfactual_pruned_ranked
        counterfactual_ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        counterfactual_admitted, _ = _admit_ranked(config, counterfactual_ranked)
        pruned_residuals = set(pruned_paths)
        infeasible_would_enter = sum(item[2] in pruned_residuals for item in counterfactual_admitted)
        admitted_ids = {(item[2], item[3]) for item in admitted}
        counterfactual_ids = {(item[2], item[3]) for item in counterfactual_admitted}
        admitted_by = Counter(item[5] for item in admitted)
        eligible_by = Counter(item[5] for item in ranked)
        frontier = [(item[2], item[4]) for item in admitted]
        admitted_states = {item[0] for item in frontier}
        for name, (known_actions, states) in trajectory_states.items():
            if depth < len(known_actions):
                row = trajectory_stage[name]
                row["child_admitted"] = states[depth + 1] in admitted_states or states[depth + 1] == zero
                if not row["parent_state_present"]:
                    row["loss_stage"] = "parent_previously_pruned"
                elif not row["candidate_generated"]:
                    row["loss_stage"] = "candidate_missing_from_provider"
                elif not row["child_valid"]:
                    row["loss_stage"] = "child_invalid"
                elif depth + 1 == config.rank and states[depth + 1] != zero:
                    row["loss_stage"] = "child_ineligible_at_depth_limit"
                elif row["child_feasibility_pruned"]:
                    row["loss_stage"] = "child_pruned_by_feasibility_gate"
                elif not row["child_survived_deduplication"] and states[depth + 1] != zero:
                    row["loss_stage"] = "child_removed_as_duplicate"
                elif not row["child_admitted"]:
                    row["loss_stage"] = "child_rejected_by_beam_admission"
                else:
                    row["loss_stage"] = None
                trajectory_rows.append(row)
        depth_diagnostics.append({
            "depth": depth + 1,
            "generated": dict(sorted(generated_by.items())),
            "valid": dict(sorted(valid_by.items())),
            "invalid": dict(sorted(invalid_by.items())),
            "removed_by_deduplication": dict(sorted(duplicate_by.items())),
            "mixed_provenance_states": sum(len(values) > 1 for values in provenance_by_state.values()),
            "eligible_after_deduplication": dict(sorted(eligible_by.items())),
            "admitted": dict(sorted(admitted_by.items())),
            "rejected_by_beam_capacity": {name: eligible_by[name] - admitted_by[name] for name in ("baseline", "alternative")},
            "unused_quota_slots_before_fill": unused,
            "beam_occupancy": len(frontier),
            "nodes_expanded": expanded_this_depth,
            "feasibility": {
                "gate": config.feasibility_gate,
                "remaining_budget": config.rank - (depth + 1),
                "checked_children": checked_this_depth,
                "flattening_lower_bound_histogram": dict(sorted(rank_histogram.items())),
                "pruned_children": pruned_this_depth,
                "surviving_checked_children": checked_this_depth - pruned_this_depth,
                "pruned_fraction": pruned_this_depth / checked_this_depth if checked_this_depth else 0.0,
                "check_seconds": check_ns_this_depth / 1e9,
                "infeasible_that_would_enter_beam": infeasible_would_enter,
                "admission_changed": admitted_ids != counterfactual_ids,
                "survivors_admitted_only_after_pruning": len(admitted_ids - counterfactual_ids),
            },
            "cumulative_provider_calls": getattr(provider, "calls", provider_calls_before + expanded_this_depth),
            "remaining_residual_nonzeros": dict(sorted(Counter(
                sum(value != 0 for plane in item[0] for row in plane for value in row) for item in frontier
            ).items())),
        })
        if exhausted:
            break
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
    termination_reason = "exact_solution" if records else ("node_budget" if exhausted else "depth_limit_or_frontier_exhausted")
    for row in depth_diagnostics:
        row["termination_reason"] = termination_reason if row is depth_diagnostics[-1] else None
    return {
        "config": asdict(config),
        "success": bool(records),
        "solutions": records,
        "nodes_expanded": nodes_expanded,
        "provider_calls": provider_calls,
        "candidate_edges_generated": generated,
        "duplicate_state_paths": duplicate_states,
        "depth_reached": depth_reached,
        "budget_exhausted": exhausted,
        "termination_reason": termination_reason,
        "depth_diagnostics": depth_diagnostics,
        "known_trajectory_diagnostics": trajectory_rows,
        "feasibility_gate_diagnostics": {
            "gate": config.feasibility_gate,
            "activation_budget": config.feasibility_activation_budget,
            "checked_children": feasibility_checked,
            "pruned_children": feasibility_pruned,
            "check_seconds": feasibility_ns / 1e9,
        },
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
