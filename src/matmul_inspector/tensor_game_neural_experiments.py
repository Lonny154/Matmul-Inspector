"""Reproducible TG-3C neural-guided MCTS experiments and evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from .tensor_game import matrix_multiplication_tensor, sum_actions, tensor_shape
from .tensor_game_dataset import SyntheticDataset, tensor_from_list
from .tensor_game_experiments import CurriculumProblem, action_to_dict, tensor_digest
from .tensor_game_learning import PolicyValueConfig
from .tensor_game_neural_mcts import (
    HYBRID_POLICY_VERSION,
    NEURAL_POLICY_VERSION,
    NEURAL_VALUE_VERSION,
    NeuralActionPolicy,
    NeuralResidualHybridPolicy,
    NeuralValueEvaluator,
)
from .tensor_game_search import HeuristicValueEvaluator, ResidualInformedPolicy, SampledMCTS, SearchConfig

FORMAT_VERSION = 1
CONFIGURATIONS = (
    "residual_baseline",
    "neural_policy_only",
    "neural_value_only",
    "neural_policy_value",
    "neural_residual_hybrid",
)


@dataclass(frozen=True)
class NeuralMCTSExperimentConfig:
    seeds: tuple[int, ...] = (3, 11)
    simulations: int = 200
    max_actions: int = 12
    c_puct: float = 1.5
    node_limit: int = 200
    max_wall_time_seconds: float = 5.0
    coefficients: tuple[int, ...] = (-1, 0, 1)
    neural_fraction: float = 0.5
    exploration_fraction: float = 0.0
    heldout_per_length: int = 1

    def __post_init__(self) -> None:
        if not self.seeds or any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("at least one integer seed is required")
        if self.heldout_per_length <= 0:
            raise ValueError("heldout_per_length must be positive")
        if not 0 <= self.neural_fraction <= 1 or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("mixture fractions must be in [0, 1]")
        SearchConfig(
            simulations=self.simulations, max_actions=self.max_actions, max_moves=1,
            c_puct=self.c_puct, node_limit=self.node_limit,
            max_wall_time_seconds=self.max_wall_time_seconds,
        )


def dataset_content_digest(dataset: SyntheticDataset) -> str:
    return hashlib.sha256(
        "".join(row["content_digest"] for row in (*dataset.decompositions, *dataset.samples)).encode("ascii")
    ).hexdigest()


def heldout_curriculum(dataset: SyntheticDataset, per_length: int = 1) -> tuple[CurriculumProblem, ...]:
    """Select only persisted test decomposition groups, without exposing factors."""
    selected = []
    counts: dict[int, int] = {}
    for row in sorted(dataset.decompositions, key=lambda item: item["decomposition_id"]):
        if row["split"] != "test":
            continue
        length = row["construction_length"]
        if counts.get(length, 0) >= per_length:
            continue
        selected.append(CurriculumProblem(
            identifier=row["decomposition_id"],
            category=f"heldout_synthetic_length_{length}",
            target=tensor_from_list(row["target"]),
            max_moves=length,
            evidence="persisted test decomposition group; reference factors withheld from search",
        ))
        counts[length] = counts.get(length, 0) + 1
    if not selected:
        raise ValueError("dataset has no held-out test decomposition groups")
    shape = tuple(dataset.manifest["config"]["shape"])
    if shape == (4, 4, 4):
        target = matrix_multiplication_tensor(2, 2, 2)
        selected.extend((
            CurriculumProblem("ood_matmul_2x2_eight", "matrix_multiplication_ood", target, 8, "OOD exact target"),
            CurriculumProblem("ood_matmul_2x2_seven", "matrix_multiplication_ood", target, 7, "OOD exact target"),
        ))
    return tuple(selected)


def _components(name, model, model_config, device, experiment_config):
    shape = model_config.shape
    residual = ResidualInformedPolicy(shape, experiment_config.coefficients)
    neural = NeuralActionPolicy(model, model_config, device)
    heuristic = HeuristicValueEvaluator()
    value = NeuralValueEvaluator(model, model_config, device)
    if name == "residual_baseline":
        return residual, heuristic
    if name == "neural_policy_only":
        return neural, heuristic
    if name == "neural_value_only":
        return residual, value
    if name == "neural_policy_value":
        return neural, value
    if name == "neural_residual_hybrid":
        return NeuralResidualHybridPolicy(
            neural, residual, experiment_config.neural_fraction, experiment_config.exploration_fraction
        ), heuristic
    raise ValueError(f"unknown neural MCTS configuration: {name}")


def _policy_neural_metrics(policy) -> tuple[int, float, list[float]]:
    neural = policy.neural if isinstance(policy, NeuralResidualHybridPolicy) else policy
    return (
        getattr(neural, "inference_calls", 0),
        getattr(neural, "inference_seconds", 0.0),
        list(getattr(neural, "prior_entropies", ())),
    )


def run_trials(
    model: Any,
    model_config: PolicyValueConfig,
    config: NeuralMCTSExperimentConfig,
    problems: Sequence[CurriculumProblem],
    device: Any = "cpu",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for problem in problems:
        if tensor_shape(problem.target) != model_config.shape:
            raise ValueError(f"target {problem.identifier} is incompatible with checkpoint shape")
        for configuration in CONFIGURATIONS:
            for seed in config.seeds:
                policy, evaluator = _components(configuration, model, model_config, device, config)
                search_config = SearchConfig(
                    simulations=config.simulations, max_actions=config.max_actions,
                    max_moves=problem.max_moves, seed=seed, c_puct=config.c_puct,
                    node_limit=config.node_limit, max_wall_time_seconds=config.max_wall_time_seconds,
                )
                result = SampledMCTS(problem.target, policy, search_config, evaluator).run()
                inference_calls, inference_seconds, entropies = _policy_neural_metrics(policy)
                exact = bool(result.solution and sum_actions(result.solution, model_config.shape) == problem.target)
                values = result.value_predictions
                record = {
                    "target_id": problem.identifier,
                    "target_category": problem.category,
                    "target_digest_sha256": tensor_digest(problem.target),
                    "tensor_shape": list(model_config.shape),
                    "max_moves": problem.max_moves,
                    "configuration": configuration,
                    "policy": policy.name,
                    "value_evaluator": evaluator.name,
                    "seed": seed,
                    "success": exact,
                    "exact_reconstruction": exact,
                    "terms": result.scalar_multiplications,
                    "solution": [action_to_dict(action) for action in result.solution] if result.solution else None,
                    "termination_reason": result.termination_reason,
                    "simulations_permitted": config.simulations,
                    "simulations_completed": result.simulations_run,
                    "expanded_nodes": result.expanded_nodes,
                    "unique_states": result.unique_states,
                    "transposition_hits": result.transposition_hits,
                    "maximum_depth": result.maximum_depth,
                    "proposal_attempts": result.proposal_attempts,
                    "valid_proposals": result.valid_proposals,
                    "accepted_proposals": result.accepted_proposals,
                    "rejected_proposals": result.proposal_attempts - result.valid_proposals,
                    "duplicate_proposals": result.duplicate_proposals,
                    "neural_policy_inference_calls": inference_calls,
                    "neural_policy_inference_seconds": inference_seconds,
                    "neural_value_inference_calls": result.value_evaluation_calls if evaluator.name.startswith("neural") else 0,
                    "neural_value_inference_seconds": result.value_inference_seconds,
                    "proposal_generation_seconds": result.proposal_generation_seconds,
                    "search_wall_seconds": result.search_wall_seconds,
                    "simulations_per_second": result.simulations_per_second,
                    "value_prediction_clips": result.value_prediction_clips,
                    "node_value_distribution": {
                        "count": len(values),
                        "minimum": min(values) if values else None,
                        "maximum": max(values) if values else None,
                        "mean": sum(values) / len(values) if values else None,
                    },
                    "mean_policy_prior_entropy": sum(entropies) / len(entropies) if entropies else None,
                }
                records.append(record)
                if progress:
                    progress(record)
    return records


def _markdown(records: Sequence[dict[str, Any]], config: NeuralMCTSExperimentConfig) -> str:
    lines = [
        "# TG-3C neural-guided MCTS comparison", "",
        "All configurations use exact TensorGame transitions and matched search limits. Neural guidance is fixed; no weights are updated.",
        "Failed and time-limited trials are retained. Equal simulation budgets do not imply equal compute cost.", "",
        f"Seeds: `{list(config.seeds)}`. Simulations: `{config.simulations}`. Proposals: `{config.max_actions}`. Node limit: `{config.node_limit}`.", "",
        "| Target | Configuration | Successes | Trials | Mean simulations | Mean search seconds | Mean neural seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target in dict.fromkeys(row["target_id"] for row in records):
        for configuration in CONFIGURATIONS:
            group = [r for r in records if r["target_id"] == target and r["configuration"] == configuration]
            if not group:
                continue
            neural_seconds = sum(r["neural_policy_inference_seconds"] + r["neural_value_inference_seconds"] for r in group)
            lines.append(
                f"| {target} | {configuration} | {sum(r['success'] for r in group)} | {len(group)} | "
                f"{sum(r['simulations_completed'] for r in group) / len(group):.1f} | "
                f"{sum(r['search_wall_seconds'] for r in group) / len(group):.6f} | {neural_seconds / len(group):.6f} |"
            )
    lines.extend(("", "Success means exact reconstruction of the target. A failure is only a bounded-search outcome.", ""))
    return "\n".join(lines)


def save_experiment(
    output: Path,
    config: NeuralMCTSExperimentConfig,
    records: Sequence[dict[str, Any]],
    *,
    dataset: SyntheticDataset,
    checkpoint_manifest: dict[str, Any],
    checkpoint_sha256: str,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "format_version": FORMAT_VERSION,
        "experiment": "tg3c_neural_guided_mcts",
        "experiment_id": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_content_digest": dataset_content_digest(dataset),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_dataset_id": checkpoint_manifest["dataset_id"],
        "checkpoint_dataset_content_digest": checkpoint_manifest["dataset_content_digest"],
        "policy_adapter_version": NEURAL_POLICY_VERSION,
        "value_adapter_version": NEURAL_VALUE_VERSION,
        "hybrid_adapter_version": HYBRID_POLICY_VERSION,
        "config": asdict(config),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor() or "unknown"},
        "timing_semantics": {
            "cpu": "wall time includes tensor construction, model inference, decoding, and search as individually labeled",
            "cuda": "input/output transfers are included in adapter/search wall time; no per-node synchronization is added, so component timings are not standalone kernel timings",
        },
        "trials": list(records),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(records, config), encoding="utf-8")


def load_experiment(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "results.json").read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION or payload.get("experiment") != "tg3c_neural_guided_mcts":
        raise ValueError("unsupported TG-3C experiment evidence")
    return payload
