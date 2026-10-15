"""TG-3D controls and evidence for neural TensorGame value guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

from .tensor_game import matrix_multiplication_tensor, subtract_action, sum_actions
from .tensor_game_dataset import SyntheticDataset, tensor_from_list
from .tensor_game_learning import PolicyValueConfig, create_policy_value_model
from .tensor_game_neural_mcts import ConstantValueEvaluator, NeuralValueEvaluator
from .tensor_game_search import (
    HeuristicValueEvaluator,
    InstrumentedPolicy,
    ProposalMetrics,
    ResidualInformedPolicy,
    SampledMCTS,
    SearchConfig,
)

FORMAT_VERSION = 1
VALUE_CONFIGURATIONS = ("heuristic", "trained_neural", "constant_zero", "untrained_neural")
UNTRAINED_SEED = 20261013


@dataclass(frozen=True)
class ValueStudyConfig:
    seeds: tuple[int, ...] = (3, 11, 19, 27, 35, 43)
    simulations: int = 200
    max_actions: int = 12
    node_limit: int = 200
    wall_time_seconds: float = 5.0
    c_puct: float = 1.5
    coefficients: tuple[int, ...] = (-1, 0, 1)
    move_limits: tuple[int, ...] = (8, 7)
    constant_value: float = 0.0
    trace_limit_per_trial: int = 64
    proposal_trace_limit_per_trial: int = 32

    def __post_init__(self) -> None:
        if self.seeds != (3, 11, 19, 27, 35, 43) or self.move_limits != (8, 7):
            raise ValueError("TG-3D seed and move-limit plans are preregistered")
        if self.trace_limit_per_trial < 0 or self.proposal_trace_limit_per_trial < 0:
            raise ValueError("trace limits must be nonnegative")
        SearchConfig(
            simulations=self.simulations, max_actions=self.max_actions, max_moves=1,
            node_limit=self.node_limit, max_wall_time_seconds=self.wall_time_seconds,
            c_puct=self.c_puct,
        )


@dataclass
class TracedResidualPolicy(InstrumentedPolicy):
    shape: tuple[int, int, int]
    coefficients: tuple[int, ...]
    trace_limit: int = 0
    name: str = field(default="residual_informed", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)
    trace: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.delegate = ResidualInformedPolicy(self.shape, self.coefficients)

    def reset_metrics(self) -> None:
        self.metrics = ProposalMetrics()
        self.delegate.reset_metrics()
        self.trace = []

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        before = asdict(self.delegate.metrics)
        proposals = self.delegate.propose_actions(residual, remaining_moves, rng, max_actions)
        after = asdict(self.delegate.metrics)
        for key in ("attempted", "valid", "accepted", "duplicates", "generation_seconds"):
            setattr(self.metrics, key, getattr(self.metrics, key) + after[key] - before[key])
        if len(self.trace) < self.trace_limit:
            state_digest = hashlib.sha256(json.dumps(residual, separators=(",", ":")).encode()).hexdigest()
            payload = [
                {"u": list(item.action.u), "v": list(item.action.v), "w": list(item.action.w), "prior": item.prior}
                for item in proposals
            ]
            self.trace.append({
                "state_digest": state_digest,
                "remaining_moves": remaining_moves,
                "proposal_digest": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                "accepted": len(proposals),
            })
        return proposals


def create_untrained_model(config: PolicyValueConfig, seed: int = UNTRAINED_SEED):
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = create_policy_value_model(config)
    model.eval()
    return model


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def _evaluator(name, trained_model, untrained_model, model_config, device, config):
    if name == "heuristic":
        return HeuristicValueEvaluator()
    if name == "constant_zero":
        return ConstantValueEvaluator(config.constant_value)
    model = trained_model if name == "trained_neural" else untrained_model
    return NeuralValueEvaluator(model, model_config, device, trace_limit=config.trace_limit_per_trial)


def run_value_study(
    trained_model: Any,
    untrained_model: Any,
    model_config: PolicyValueConfig,
    config: ValueStudyConfig,
    device: Any = "cpu",
    progress=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = matrix_multiplication_tensor(2, 2, 2)
    records = []
    diagnostics = []
    for max_moves in config.move_limits:
        for seed in config.seeds:
            for value_name in VALUE_CONFIGURATIONS:
                policy = TracedResidualPolicy(
                    model_config.shape, config.coefficients, config.proposal_trace_limit_per_trial
                )
                evaluator = _evaluator(
                    value_name, trained_model, untrained_model, model_config, device, config
                )
                search = SampledMCTS(
                    target, policy,
                    SearchConfig(
                        simulations=config.simulations, max_actions=config.max_actions,
                        max_moves=max_moves, seed=seed, c_puct=config.c_puct,
                        node_limit=config.node_limit, max_wall_time_seconds=config.wall_time_seconds,
                    ),
                    evaluator,
                )
                result = search.run()
                verified = bool(result.solution and sum_actions(result.solution, model_config.shape) == target)
                record = {
                    "value_configuration": value_name,
                    "target": "matmul_2x2",
                    "target_digest": hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest(),
                    "max_moves": max_moves,
                    "seed": seed,
                    "success": verified,
                    "exact_reconstruction_verified": verified,
                    "terms": len(result.solution) if result.solution else None,
                    "solution": [
                        {"u": list(a.u), "v": list(a.v), "w": list(a.w)} for a in result.solution
                    ] if result.solution else None,
                    "simulations_permitted": config.simulations,
                    "simulations_completed": result.simulations_run,
                    "simulations_to_first_success": result.simulations_to_first_success,
                    "unique_states": result.unique_states,
                    "expanded_nodes": result.expanded_nodes,
                    "transposition_hits": result.transposition_hits,
                    "maximum_depth": result.maximum_depth,
                    "proposal_attempts": result.proposal_attempts,
                    "valid_proposals": result.valid_proposals,
                    "accepted_proposals": result.accepted_proposals,
                    "duplicate_proposals": result.duplicate_proposals,
                    "proposal_generation_seconds": result.proposal_generation_seconds,
                    "value_queries": result.value_evaluation_calls,
                    "value_clips": result.value_prediction_clips,
                    "value_evaluation_seconds": result.value_inference_seconds,
                    "search_wall_seconds": result.search_wall_seconds,
                    "simulations_per_second": result.simulations_per_second,
                    "termination_reason": result.termination_reason,
                    "proposal_trace": policy.trace,
                }
                records.append(record)
                if isinstance(evaluator, NeuralValueEvaluator):
                    successful_digests = set()
                    residual = target
                    if result.solution:
                        for action in result.solution:
                            successful_digests.add(hashlib.sha256(json.dumps(residual, separators=(",", ":")).encode()).hexdigest())
                            residual = subtract_action(residual, action)
                    trace = []
                    for item in evaluator.trace:
                        trace.append({**item, "on_successful_trajectory": item["state_digest"] in successful_digests})
                    diagnostics.append({
                        "value_configuration": value_name,
                        "max_moves": max_moves,
                        "seed": seed,
                        "queries": evaluator.metrics.calls,
                        "unique_states_evaluated": len(evaluator.unique_state_digests),
                        "clips": evaluator.metrics.clipped,
                        "clip_fraction": evaluator.metrics.clipped / evaluator.metrics.calls if evaluator.metrics.calls else 0.0,
                        "clipping_by_depth": evaluator.clipping_by_depth,
                        "clip_reasons": evaluator.clip_reasons,
                        "raw_predicted_lengths": _summary(evaluator.raw_predictions),
                        "clamped_predicted_lengths": _summary(evaluator.clamped_lengths),
                        "converted_leaf_values": _summary(evaluator.metrics.predictions),
                        "queries_per_simulation": evaluator.metrics.calls / result.simulations_run,
                        "value_evaluation_seconds": evaluator.metrics.inference_seconds,
                        "bounded_trace": trace,
                        "trace_limit": config.trace_limit_per_trial,
                    })
                if progress:
                    progress(record)
    return records, diagnostics


def synthetic_value_diagnostic(dataset: SyntheticDataset, evaluators: dict[str, Any], max_per_length: int = 1):
    selected = []
    counts = {}
    for row in sorted(dataset.decompositions, key=lambda item: item["decomposition_id"]):
        length = row["construction_length"]
        if row["split"] != "test" or counts.get(length, 0) >= max_per_length:
            continue
        selected.append(row)
        counts[length] = counts.get(length, 0) + 1
    results = []
    for row in selected:
        residual = tensor_from_list(row["target"])
        for name, evaluator in evaluators.items():
            evaluator.reset_metrics()
            value = evaluator.evaluate(residual, row["construction_length"], row["construction_length"], 1)
            results.append({
                "decomposition_id": row["decomposition_id"], "construction_length": row["construction_length"],
                "value_configuration": name, "converted_value": value,
                "raw_predicted_length": evaluator.raw_predictions[-1] if isinstance(evaluator, NeuralValueEvaluator) else None,
            })
    return results


def paired_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trained = {(r["max_moves"], r["seed"]): r for r in records if r["value_configuration"] == "trained_neural"}
    result = {}
    for name in VALUE_CONFIGURATIONS:
        group = [r for r in records if r["value_configuration"] == name]
        pairs = [(trained[(r["max_moves"], r["seed"])]["success"], r["success"]) for r in group]
        result[name] = {
            "successes": sum(r["success"] for r in group), "trials": len(group),
            "success_rate": sum(r["success"] for r in group) / len(group),
            "both_success": sum(a and b for a, b in pairs),
            "trained_only": sum(a and not b for a, b in pairs),
            "baseline_only": sum(not a and b for a, b in pairs),
            "neither": sum(not a and not b for a, b in pairs),
            "mean_search_seconds": sum(r["search_wall_seconds"] for r in group) / len(group),
            "median_search_seconds": statistics.median(r["search_wall_seconds"] for r in group),
            "timeouts": sum(r["termination_reason"] == "wall_time_limit_reached" for r in group),
        }
    return result


def save_value_study(output: Path, config: ValueStudyConfig, records, diagnostics, synthetic, provenance):
    output.mkdir(parents=True, exist_ok=False)
    experiment = {
        "format_version": FORMAT_VERSION, "experiment": "tg3d_neural_value_study",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "value_configurations": list(VALUE_CONFIGURATIONS), "untrained_seed": UNTRAINED_SEED,
        "provenance": provenance,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    summary = paired_summary(records)
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps({"trials": records, "paired_summary": summary}, indent=2, sort_keys=True) + "\n")
    (output / "value_diagnostics.json").write_text(json.dumps({"trials": diagnostics, "synthetic": synthetic}, indent=2, sort_keys=True) + "\n")
    lines = ["# TG-3D neural value guidance study", "", "Residual-informed proposals were fixed across all value controls.", "", "| Value control | Successes | Trials | Mean search seconds | Timeouts |", "|---|---:|---:|---:|---:|"]
    for name in VALUE_CONFIGURATIONS:
        row = summary[name]
        lines.append(f"| {name} | {row['successes']} | {row['trials']} | {row['mean_search_seconds']:.6f} | {row['timeouts']} |")
    lines.extend(("", "Success requires independent exact-integer reconstruction. Seven-term failure is only a bounded-search outcome.", ""))
    (output / "report.md").write_text("\n".join(lines))


def load_value_study(output: Path):
    experiment = json.loads((output / "experiment.json").read_text())
    if experiment.get("format_version") != FORMAT_VERSION or experiment.get("experiment") != "tg3d_neural_value_study":
        raise ValueError("unsupported TG-3D evidence")
    return experiment, json.loads((output / "results.json").read_text()), json.loads((output / "value_diagnostics.json").read_text())
