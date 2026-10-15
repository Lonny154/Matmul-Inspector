"""TG-4D candidate recall, proposal-union, and training-coverage diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, tensor_shape
from .tensor_game_experience import action_from_payload, action_identity, search_policy
from .tensor_game_learning import rank_complete_actions
from .tensor_game_neural_mcts import canonical_action_log_probability
from .tensor_game_search import (
    HeuristicValueEvaluator,
    InstrumentedPolicy,
    ProposalMetrics,
    ProposedAction,
    SampledMCTS,
    SearchConfig,
    canonicalize_action,
)

CUTOFFS = (1, 12, 32, 64, 128, 256)
SEEDS = (3, 11, 19, 27, 35, 43)


def _torch():
    import torch
    return torch


def decode(model, config, residual, cutoff: int) -> list[tuple[RankOneAction, float]]:
    torch = _torch()
    tensor = torch.tensor(residual, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits, _ = model(tensor)
    ranked = rank_complete_actions(logits[0], config.shape, cutoff)
    return [(action, canonical_action_log_probability(logits[0], action, config.shape)) for action, _ in ranked]


def recall_sweep(critical: dict[str, dict[str, Any]], models: dict[str, tuple[Any, Any]]) -> list[dict[str, Any]]:
    """Run the real bounded decoder independently at every requested cutoff."""
    rows = []
    for target, fixture in sorted(critical.items()):
        gold = action_from_payload(fixture["gold_action"])
        gold_id = action_identity(gold)
        decoded = {
            cutoff: {name: decode(model, config, fixture["residual"], cutoff)
                     for name, (model, config) in models.items()}
            for cutoff in CUTOFFS
        }
        largest = {name: [action_identity(action) for action, _ in decoded[256][name]] for name in models}
        for seed in SEEDS:
            for name in ("frozen", "trained"):
                record = {
                    "target_group_id": target, "seed": seed, "policy": name,
                    "required_action_id": gold_id,
                    "bounded_rank_256": largest[name].index(gold_id) + 1 if gold_id in largest[name] else None,
                }
                for cutoff in CUTOFFS:
                    ids = [action_identity(action) for action, _ in decoded[cutoff][name]]
                    record[f"recall_at_{cutoff}"] = gold_id in ids
                    other = "trained" if name == "frozen" else "frozen"
                    other_ids = {action_identity(action) for action, _ in decoded[cutoff][other]}
                    record[f"overlap_at_{cutoff}"] = len(set(ids) & other_ids)
                rows.append(record)
    return rows


def aggregate_recall(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for policy in ("frozen", "trained"):
        selected = [row for row in rows if row["policy"] == policy]
        unique = {row["target_group_id"]: row for row in selected}.values()
        output[policy] = {
            "per_run_denominator": len(selected),
            "unique_target_denominator": len(list(unique)),
            "per_run_recall": {str(cutoff): sum(row[f"recall_at_{cutoff}"] for row in selected) for cutoff in CUTOFFS},
            "unique_target_recall": {
                str(cutoff): sum(row[f"recall_at_{cutoff}"] for row in {r["target_group_id"]: r for r in selected}.values())
                for cutoff in CUTOFFS
            },
        }
    return output


def _normalize(items: Sequence[tuple[RankOneAction, float, str]]) -> tuple[ProposedAction, ...]:
    total = sum(weight for _, weight, _ in items)
    return tuple(ProposedAction(canonicalize_action(action), weight / total, (source,)) for action, weight, source in items)


@dataclass
class ExperimentalCombinedPolicy(InstrumentedPolicy):
    """Experiment-only frozen/trained proposal composition."""

    frozen_model: Any
    frozen_config: Any
    trained_model: Any
    trained_config: Any
    mode: str
    name: str = field(init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"frozen_top12", "trained_top12", "union_top12", "alternating_hybrid12"}:
            raise ValueError("unknown experimental proposal mode")
        self.name = f"tg4d_{self.mode}"

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        del rng
        started = time.perf_counter()
        if remaining_moves <= 0 or tensor_shape(residual) != tuple(self.frozen_config.shape):
            return ()
        frozen = decode(self.frozen_model, self.frozen_config, residual, 12) if self.mode != "trained_top12" else []
        trained = decode(self.trained_model, self.trained_config, residual, 12) if self.mode != "frozen_top12" else []
        sources = {"frozen": frozen, "trained": trained}
        if self.mode == "frozen_top12":
            ordered = [(action, score, "frozen") for action, score in frozen]
        elif self.mode == "trained_top12":
            ordered = [(action, score, "trained") for action, score in trained]
        elif self.mode == "union_top12":
            ordered = [(action, score, source) for source in ("frozen", "trained") for action, score in sources[source]]
        else:
            ordered = []
            for index in range(12):
                for source in ("frozen", "trained"):
                    if index < len(sources[source]):
                        action, score = sources[source][index]
                        ordered.append((action, score, source))
            ordered = ordered[:max_actions]
        unique = []
        seen = set()
        for action, score, source in ordered:
            action = canonicalize_action(action)
            if action in seen:
                self.metrics.duplicates += 1
                continue
            seen.add(action)
            # Source-conditional probability; each source receives equal mass.
            denominator = sum(math.exp(value) for _, value in sources[source])
            source_mass = 0.5 if frozen and trained else 1.0
            unique.append((action, source_mass * math.exp(score) / denominator, source))
        unique = unique[:max_actions]
        self.metrics.attempted += len(ordered)
        self.metrics.valid += len(ordered)
        self.metrics.accepted += len(unique)
        self.metrics.generation_seconds += time.perf_counter() - started
        return _normalize(unique) if unique else ()


def run_proposal_experiments(
    critical: dict[str, dict[str, Any]], models: dict[str, tuple[Any, Any]]
) -> list[dict[str, Any]]:
    rows = []
    definitions = {
        "frozen_top12": {"budget": 12, "comparison": "equal_budget"},
        "trained_top12": {"budget": 12, "comparison": "equal_budget"},
        "alternating_hybrid12": {"budget": 12, "comparison": "equal_budget"},
        "union_top12": {"budget": 24, "comparison": "expanded_budget"},
    }
    for target, fixture in sorted(critical.items()):
        gold_id = action_identity(action_from_payload(fixture["gold_action"]))
        for seed in SEEDS:
            for mode, definition in definitions.items():
                policy = ExperimentalCombinedPolicy(
                    models["frozen"][0], models["frozen"][1], models["trained"][0], models["trained"][1], mode
                )
                started = time.perf_counter()
                result = SampledMCTS(
                    fixture["target"], policy,
                    SearchConfig(simulations=200, max_actions=definition["budget"], max_moves=1,
                                 seed=seed, c_puct=1.5, node_limit=200, max_wall_time_seconds=5.0),
                    HeuristicValueEvaluator(),
                ).run()
                wall = time.perf_counter() - started
                behavior = search_policy(result.root_actions, 0.0)
                chosen = next(row for row in behavior if row["search_probability"] == 1.0)
                candidate_ids = [action_identity(row.action) for row in result.root_actions]
                rows.append({
                    "target_group_id": target, "seed": seed, "configuration": mode,
                    "comparison_class": definition["comparison"], "candidate_budget": definition["budget"],
                    "unique_candidate_count": len(candidate_ids), "candidate_evaluations": result.accepted_proposals,
                    "search_expansions": result.expanded_nodes, "unique_states": result.unique_states,
                    "simulations_completed": result.simulations_run,
                    "proposal_generation_seconds": result.proposal_generation_seconds,
                    "search_wall_seconds": result.search_wall_seconds, "experiment_wall_seconds": wall,
                    "required_action_recalled": gold_id in candidate_ids,
                    "selected_action_id": chosen["action_id"],
                    "exact_discovery": chosen["action_id"] == gold_id,
                })
    return rows


def aggregate_proposals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in sorted({row["configuration"] for row in rows}):
        selected = [row for row in rows if row["configuration"] == name]
        output[name] = {
            "comparison_class": selected[0]["comparison_class"], "candidate_budget": selected[0]["candidate_budget"],
            "trials": len(selected), "unique_targets": len({row["target_group_id"] for row in selected}),
            "exact_action_recall": sum(row["required_action_recalled"] for row in selected),
            "exact_discoveries": sum(row["exact_discovery"] for row in selected),
            "mean_unique_candidates": sum(row["unique_candidate_count"] for row in selected) / len(selected),
            "mean_candidate_evaluations": sum(row["candidate_evaluations"] for row in selected) / len(selected),
            "mean_search_expansions": sum(row["search_expansions"] for row in selected) / len(selected),
            "mean_search_wall_seconds": sum(row["search_wall_seconds"] for row in selected) / len(selected),
            "runtime_is_descriptive": True,
        }
    return output


def audit_training_coverage(experience: Path, critical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decisions = json.loads((experience / "decisions.json").read_text(encoding="utf-8"))
    episodes = {row["episode_id"]: row for row in json.loads((experience / "episodes.json").read_text(encoding="utf-8"))}
    critical_ids = {target: action_identity(action_from_payload(row["gold_action"])) for target, row in critical.items()}
    action_frequency = Counter()
    action_probability = Counter()
    split_frequency: dict[str, Counter] = defaultdict(Counter)
    split_probability: dict[str, Counter] = defaultdict(Counter)
    for decision in decisions:
        split = decision["model_provenance"]["target_split"]
        for action in decision["sampled_actions"]:
            identifier = action["action_id"]
            action_frequency[identifier] += 1
            action_probability[identifier] += action["search_probability"]
            split_frequency[split][identifier] += 1
            split_probability[split][identifier] += action["search_probability"]
    train_counts = sorted(split_frequency["train"].values())
    records = []
    for target, identifier in sorted(critical_ids.items()):
        matching = [
            (decision, action)
            for decision in decisions if decision["model_provenance"]["target_split"] == "train"
            for action in decision["sampled_actions"] if action["action_id"] == identifier
        ]
        records.append({
            "target_group_id": target, "action_id": identifier,
            "all_support_occurrences": action_frequency[identifier],
            "all_target_probability_mass": action_probability[identifier],
            "train_support_occurrences": split_frequency["train"][identifier],
            "train_target_probability_mass": split_probability["train"][identifier],
            "validation_support_occurrences": split_frequency["validation"][identifier],
            "test_support_occurrences": split_frequency["test"][identifier],
            "nonzero_supervised_target": split_probability["train"][identifier] > 0,
            "natural_example_weight_when_present": 1.0 if split_frequency["train"][identifier] else 0.0,
            "effective_cross_entropy_target_mass": split_probability["train"][identifier],
            "selected_behavior_action_occurrences": sum(decision["selected_action_id"] == identifier for decision, _ in matching),
            "occurrences_from_successful_episodes": sum(episodes[decision["episode_id"]]["outcome"] == "success" for decision, _ in matching),
            "maximum_single_target_probability": max((action["search_probability"] for _, action in matching), default=0.0),
            "training_decisions_where_action_absent": sum(
                identifier not in {action["action_id"] for action in row["sampled_actions"]}
                for row in decisions if row["model_provenance"]["target_split"] == "train"
            ),
        })
    return {
        "training_decisions": sum(row["model_provenance"]["target_split"] == "train" for row in decisions),
        "distinct_training_support_actions": len(split_frequency["train"]),
        "training_action_occurrence_distribution": {
            "minimum": train_counts[0], "median": train_counts[len(train_counts) // 2], "maximum": train_counts[-1],
            "singletons": sum(value == 1 for value in train_counts),
        },
        "critical_actions": records,
        "interpretation": "Exact canonical action identity across recorded sampled supports; absence shows missing direct supervision, not why the model generalized poorly.",
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
