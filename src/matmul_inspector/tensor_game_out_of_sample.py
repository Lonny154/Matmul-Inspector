"""Preregistered TG-4E out-of-sample policy evaluation utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, subtract_action, tensor_shape
from .tensor_game_candidate_coverage import decode
from .tensor_game_diverse_experience import DiverseTargetConfig, generate_diverse_targets
from .tensor_game_experience import action_from_payload, action_identity
from .tensor_game_search import (
    HeuristicValueEvaluator, InstrumentedPolicy, ProposalMetrics, ProposedAction,
    SampledMCTS, SearchConfig, canonicalize_action,
)

VERSION = 1
CONFIGURATIONS = ("frozen_top12", "trained_top12", "alternating_hybrid12")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_tuple(value):
    return tuple(tuple(tuple(line) for line in plane) for plane in value)


def axis_orbit_digests(target) -> set[str]:
    """Conservative equivalence check under the six tensor-axis permutations."""
    shape = tensor_shape(target)
    output = set()
    for permutation in itertools.permutations(range(3)):
        new_shape = tuple(shape[index] for index in permutation)
        transformed = tuple(tuple(tuple(
            target[tuple((i, j, k)[permutation.index(axis)] for axis in range(3))[0]]
                  [tuple((i, j, k)[permutation.index(axis)] for axis in range(3))[1]]
                  [tuple((i, j, k)[permutation.index(axis)] for axis in range(3))[2]]
            for k in range(new_shape[2])) for j in range(new_shape[1])) for i in range(new_shape[0]))
        output.add(digest(transformed))
    return output


def known_target_orbits(paths: Sequence[Path]) -> set[str]:
    result = set()
    for path in paths:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                rows = payload
            else:
                metadata = payload.get("metadata", {})
                rows = payload.get("targets", payload.get("decompositions", metadata.get("target_records", [])))
        for row in rows:
            value = row.get("target") or row.get("initial_target")
            if value is not None:
                result.update(axis_orbit_digests(tensor_tuple(value)))
    return result


def generate_fresh_targets(*, known_paths: Sequence[Path], seed: int = 20261014,
                           per_depth: int = 4) -> dict[str, Any]:
    known = known_target_orbits(known_paths)
    pool = generate_diverse_targets(DiverseTargetConfig(target_count=64, seed=seed))
    selected = []
    selected_orbits = set()
    rejected = Counter()
    for row in pool["targets"]:
        depth = row["construction_length"]
        if sum(item["construction_length"] == depth for item in selected) >= per_depth:
            continue
        target = tensor_tuple(row["target"])
        orbit = axis_orbit_digests(target)
        if orbit & known:
            rejected["known_exact_or_axis_permutation"] += 1
            continue
        if orbit & selected_orbits:
            rejected["fresh_exact_or_axis_permutation"] += 1
            continue
        copied = dict(row)
        copied["target_id"] = f"tg4e-d{depth}-{sum(item['construction_length'] == depth for item in selected):02d}-{row['target_digest'][:12]}"
        copied["split"] = "fresh_out_of_sample"
        selected.append(copied)
        selected_orbits.update(orbit)
    counts = Counter(row["construction_length"] for row in selected)
    if counts != Counter({1: per_depth, 2: per_depth, 3: per_depth, 4: per_depth}):
        raise RuntimeError(f"could not generate balanced fresh targets: {counts}")
    return {
        "version": VERSION, "generator": "filtered_tg4b2a_diverse_targets_v1",
        "generator_seed": seed, "pool_size": 64, "per_constructed_depth": per_depth,
        "equivalence_exclusion": "exact tensor plus all six axis permutations",
        "known_source_files": [str(path) for path in known_paths],
        "known_orbit_digest_count": len(known), "rejections": dict(rejected),
        "construction_depth_semantics": "known construction upper bound; not proven minimum tensor rank",
        "targets": selected,
    }


def preregistration(targets: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION, "status": "frozen_before_model_evaluation",
        "research_question": "Does the trained policy add discovery or efficiency beyond the frozen policy on fresh targets?",
        "target_content_sha256": hashlib.sha256(canonical_json(targets)).hexdigest(),
        "target_count": len(targets["targets"]), "constructed_depths": [1, 2, 3, 4],
        "unique_targets_per_depth": 4, "search_seeds": [3, 19, 35],
        "configurations": list(CONFIGURATIONS),
        "search": {"simulations": 200, "candidate_budget": 12, "c_puct": 1.5,
                   "node_limit": 200, "wall_time_seconds": 5.0,
                   "maximum_moves": "target_constructed_depth", "value": "nonzero_heuristic"},
        "hybrid": {"selection": "alternate frozen then trained by within-model rank, starting frozen",
                   "deduplication": "canonical action identity with attribution union",
                   "fill": "continue deterministic alternation until 12 unique actions",
                   "priors": "uniform over returned candidates to avoid cross-model score calibration"},
        "primary_metrics": ["exact_discovery", "simulations_to_first_success", "expanded_nodes",
                            "accepted_proposals", "model_forward_passes", "search_wall_seconds"],
        "recall_metric": "generated-reference-action recall; not exhaustive oracle recall",
        "sample_size_justification": "16 balanced targets x 3 seeds x 3 policies = 144 searches; bounded by a 5-second search timeout",
    }


def alternating_merge(frozen: Sequence[RankOneAction], trained: Sequence[RankOneAction], budget: int = 12):
    """Return deterministic unique actions and complete source attribution."""
    stream = []
    for index in range(max(len(frozen), len(trained))):
        if index < len(frozen): stream.append((canonicalize_action(frozen[index]), "frozen"))
        if index < len(trained): stream.append((canonicalize_action(trained[index]), "trained"))
    sources = defaultdict(set)
    for action, source in stream:
        sources[action_identity(action)].add(source)
    ordered = []
    for action, _ in stream:
        if action not in ordered:
            ordered.append(action)
        if len(ordered) == budget:
            break
    attribution = {action_identity(action): tuple(sorted(sources[action_identity(action)])) for action in ordered}
    return tuple(ordered), attribution


@dataclass
class EvaluationPolicy(InstrumentedPolicy):
    frozen_model: Any
    frozen_config: Any
    trained_model: Any
    trained_config: Any
    mode: str
    excluded_ids: frozenset[str] = frozenset()
    name: str = field(init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)
    forward_passes: int = field(default=0, init=False)
    inference_calls: int = field(default=0, init=False)
    attribution_by_state: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.mode not in CONFIGURATIONS:
            raise ValueError("unsupported policy configuration")
        self.name = f"tg4e_{self.mode}"

    def reset_metrics(self):
        super().reset_metrics()
        self.forward_passes = self.inference_calls = 0
        self.attribution_by_state = {}

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        del rng
        started = time.perf_counter()
        if remaining_moves <= 0 or max_actions <= 0:
            return ()
        frozen = decode(self.frozen_model, self.frozen_config, residual, 12) if self.mode != "trained_top12" else []
        trained = decode(self.trained_model, self.trained_config, residual, 12) if self.mode != "frozen_top12" else []
        self.forward_passes += bool(frozen) + bool(trained)
        self.inference_calls += bool(frozen) + bool(trained)
        if self.mode == "frozen_top12":
            stream = [(action, "frozen") for action, _ in frozen]
        elif self.mode == "trained_top12":
            stream = [(action, "trained") for action, _ in trained]
        else:
            merged, merged_sources = alternating_merge([row[0] for row in frozen], [row[0] for row in trained], max_actions)
            stream = [(action, source) for action in merged for source in merged_sources[action_identity(action)]]
        ordered, sources = [], defaultdict(set)
        for action, source in stream:
            action = canonicalize_action(action)
            identifier = action_identity(action)
            sources[identifier].add(source)
            if identifier in self.excluded_ids or action in ordered:
                continue
            ordered.append(action)
            if len(ordered) == max_actions:
                break
        # Preserve dual attribution even when a duplicate occurs after the budget boundary.
        for action, source in stream:
            identifier = action_identity(canonicalize_action(action))
            if identifier in {action_identity(item) for item in ordered}:
                sources[identifier].add(source)
        state_id = digest(residual)
        self.attribution_by_state[state_id] = {key: tuple(sorted(value)) for key, value in sources.items()}
        self.metrics.attempted += len(stream)
        self.metrics.valid += len(stream)
        self.metrics.accepted += len(ordered)
        self.metrics.duplicates += len(stream) - len({canonicalize_action(action) for action, _ in stream})
        self.metrics.generation_seconds += time.perf_counter() - started
        return tuple(ProposedAction(action, 1 / len(ordered), self.attribution_by_state[state_id][action_identity(action)])
                     for action in ordered) if ordered else ()


def reference_recall(targets, models) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        residual = tensor_tuple(target["target"])
        factors = [action_from_payload(value) for value in target["canonical_factors"]]
        for step, reference in enumerate(factors):
            for mode in CONFIGURATIONS:
                policy = EvaluationPolicy(models["frozen"][0], models["frozen"][1],
                                          models["trained"][0], models["trained"][1], mode)
                proposals = policy.propose_actions(residual, len(factors) - step, __import__("random").Random(0), 12)
                ids = [action_identity(item.action) for item in proposals]
                reference_id = action_identity(reference)
                rows.append({"target_id": target["target_id"], "constructed_depth": target["construction_length"],
                             "step": step, "configuration": mode, "reference_action_id": reference_id,
                             "recall_at_1": bool(ids and ids[0] == reference_id), "recall_at_12": reference_id in ids,
                             "candidate_count": len(ids)})
            residual = subtract_action(residual, reference)
    return rows


def run_evaluation(targets, models, seeds=(3, 19, 35)) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, attribution = [], []
    for target in targets:
        tensor = tensor_tuple(target["target"])
        for seed in seeds:
            for mode in CONFIGURATIONS:
                policy = EvaluationPolicy(models["frozen"][0], models["frozen"][1],
                                          models["trained"][0], models["trained"][1], mode)
                search = SampledMCTS(tensor, policy, SearchConfig(
                    simulations=200, max_actions=12, max_moves=target["construction_length"], seed=seed,
                    c_puct=1.5, node_limit=200, max_wall_time_seconds=5.0,
                ), HeuristicValueEvaluator()).run()
                row = {"target_id": target["target_id"], "target_digest": target["target_digest"],
                       "constructed_depth": target["construction_length"], "configuration": mode, "seed": seed,
                       "success": search.exact_reconstruction, "termination_reason": search.termination_reason,
                       "simulations_completed": search.simulations_run,
                       "simulations_to_first_success": search.simulations_to_first_success,
                       "expanded_nodes": search.expanded_nodes, "unique_states": search.unique_states,
                       "candidate_evaluations": search.accepted_proposals,
                       "policy_inference_calls": policy.inference_calls, "model_forward_passes": policy.forward_passes,
                       "proposal_generation_seconds": search.proposal_generation_seconds,
                       "search_wall_seconds": search.search_wall_seconds}
                results.append(row)
                if mode == "alternating_hybrid12" and search.solution:
                    residual = tensor
                    for step, action in enumerate(search.solution):
                        state_id = digest(residual)
                        sources = policy.attribution_by_state.get(state_id, {}).get(action_identity(action), ())
                        attribution.append({"target_id": target["target_id"], "seed": seed, "step": step,
                                            "action_id": action_identity(action),
                                            "attribution": "both" if len(sources) == 2 else (sources[0] + "_only" if sources else "unknown"),
                                            "trained_exclusive": sources == ("trained",),
                                            "candidate_in_successful_solution": True,
                                            "causal_necessity_tested": False,
                                            "necessary_for_observed_success": None})
                        residual = subtract_action(residual, action)
    return results, attribution


def aggregate(results: Sequence[dict[str, Any]], recall: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for depth in (1, 2, 3, 4, "all"):
        scoped = list(results) if depth == "all" else [row for row in results if row["constructed_depth"] == depth]
        policies = {}
        for mode in CONFIGURATIONS:
            rows = [row for row in scoped if row["configuration"] == mode]
            successes = [row for row in rows if row["success"]]
            if not rows:
                policies[mode] = {"trials": 0, "unique_targets": 0, "successes": 0, "success_rate": None,
                                  "targets_solved": 0, "mean_simulations_all": None,
                                  "mean_simulations_to_first_success": None, "mean_expansions_all": None,
                                  "mean_forward_passes_all": None, "mean_wall_seconds_all": None}
                continue
            policies[mode] = {"trials": len(rows), "unique_targets": len({row["target_id"] for row in rows}),
                              "successes": len(successes), "success_rate": len(successes) / len(rows),
                              "targets_solved": len({row["target_id"] for row in successes}),
                              "mean_simulations_all": sum(row["simulations_completed"] for row in rows) / len(rows),
                              "mean_simulations_to_first_success": (sum(row["simulations_to_first_success"] for row in successes) / len(successes)) if successes else None,
                              "mean_expansions_all": sum(row["expanded_nodes"] for row in rows) / len(rows),
                              "mean_forward_passes_all": sum(row["model_forward_passes"] for row in rows) / len(rows),
                              "mean_wall_seconds_all": sum(row["search_wall_seconds"] for row in rows) / len(rows)}
        groups[str(depth)] = policies
    paired = {}
    indexed = {(row["target_id"], row["seed"], row["configuration"]): row for row in results}
    for candidate in ("trained_top12", "alternating_hybrid12"):
        counts = Counter()
        for target, seed in sorted({(row["target_id"], row["seed"]) for row in results}):
            baseline = indexed[target, seed, "frozen_top12"]["success"]
            other = indexed[target, seed, candidate]["success"]
            counts[{(True, True): "shared_success", (True, False): "regression", (False, True): "improvement", (False, False): "shared_failure"}[baseline, other]] += 1
        paired[candidate] = dict(counts)
    recall_summary = {}
    for mode in CONFIGURATIONS:
        rows = [row for row in recall if row["configuration"] == mode]
        recall_summary[mode] = {"reference_decisions": len(rows), "top1": sum(row["recall_at_1"] for row in rows),
                                "top12": sum(row["recall_at_12"] for row in rows)}
    return {"by_depth": groups, "paired_vs_frozen": paired, "reference_action_recall": recall_summary}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] = ()):
    with path.open("w", newline="", encoding="utf-8") as stream:
        names = list(rows[0]) if rows else list(fieldnames)
        writer = csv.DictWriter(stream, fieldnames=names)
        if names: writer.writeheader()
        if rows: writer.writerows(rows)


def environment() -> dict[str, Any]:
    torch = __import__("torch")
    return {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "processor": platform.processor() or "unknown", "cuda_available": torch.cuda.is_available(),
            "device": "cpu"}
