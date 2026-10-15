"""TG-3E value-scale controls and bounded MCTS trajectory evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

from .tensor_game import matrix_multiplication_tensor, sum_actions
from .tensor_game_learning import PolicyValueConfig
from .tensor_game_neural_mcts import ConstantValueEvaluator, NeuralValueEvaluator
from .tensor_game_search import SampledMCTS, SearchConfig
from .tensor_game_value_study import TracedResidualPolicy

FORMAT_VERSION = 1
CONFIGURATIONS = (
    "constant_zero", "constant_half", "constant_max",
    "trained_original", "trained_compressed", "trained_expanded",
)
TRANSFORMS = {
    "trained_original": (1.0, 0.0),
    "trained_compressed": (0.5, 0.0),
    "trained_expanded": (2.0, 0.0),
}


@dataclass(frozen=True)
class ValueScaleConfig:
    seeds: tuple[int, ...] = (3, 11, 19, 27, 35, 43)
    simulations: int = 200
    max_actions: int = 12
    node_limit: int = 200
    wall_time_seconds: float = 5.0
    c_puct: float = 1.5
    coefficients: tuple[int, ...] = (-1, 0, 1)
    move_limits: tuple[int, ...] = (8, 7)
    traced_simulations: int = 20


class AffineNeuralValueEvaluator:
    """Apply a fixed positive affine transform to converted neural leaf values."""

    def __init__(self, model, model_config, device, scale: float, offset: float, name: str):
        if scale <= 0:
            raise ValueError("rank-preserving affine scale must be positive")
        self.base = NeuralValueEvaluator(model, model_config, device, trace_limit=64)
        self.scale = scale
        self.offset = offset
        self.name = name
        self.original_values = []
        self.transformed_values = []

    @property
    def metrics(self):
        return self.base.metrics

    def reset_metrics(self):
        self.base.reset_metrics()
        self.original_values = []
        self.transformed_values = []

    def evaluate(self, residual, remaining_moves, max_moves, target_nonzeros):
        original = self.base.evaluate(residual, remaining_moves, max_moves, target_nonzeros)
        transformed = self.scale * original + self.offset
        self.original_values.append(original)
        self.transformed_values.append(transformed)
        self.base.metrics.predictions[-1] = transformed
        if self.base.trace:
            self.base.trace[-1]["original_converted_leaf_value"] = original
            self.base.trace[-1]["transformed_leaf_value"] = transformed
            self.base.trace[-1]["affine_scale"] = self.scale
            self.base.trace[-1]["affine_offset"] = self.offset
        return transformed


def _digest_solution(solution) -> str | None:
    if not solution:
        return None
    value = [{"u": a.u, "v": a.v, "w": a.w} for a in solution]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _evaluator(name, model, model_config, device, max_moves):
    if name == "constant_zero":
        return ConstantValueEvaluator(0.0)
    if name == "constant_half":
        return ConstantValueEvaluator(max_moves / 2)
    if name == "constant_max":
        return ConstantValueEvaluator(float(max_moves))
    scale, offset = TRANSFORMS[name]
    return AffineNeuralValueEvaluator(model, model_config, device, scale, offset, name)


def run_value_scale_study(model, model_config: PolicyValueConfig, config: ValueScaleConfig, device="cpu", progress=None):
    target = matrix_multiplication_tensor(2, 2, 2)
    records = []
    trajectory = []
    diagnostics = []
    for max_moves in config.move_limits:
        for seed in config.seeds:
            for name in CONFIGURATIONS:
                policy = TracedResidualPolicy(model_config.shape, config.coefficients, trace_limit=32)
                evaluator = _evaluator(name, model, model_config, device, max_moves)
                result = SampledMCTS(
                    target, policy,
                    SearchConfig(
                        simulations=config.simulations, max_actions=config.max_actions,
                        max_moves=max_moves, seed=seed, c_puct=config.c_puct,
                        node_limit=config.node_limit, max_wall_time_seconds=config.wall_time_seconds,
                        trace_simulations=config.traced_simulations,
                    ), evaluator,
                ).run()
                verified = bool(result.solution and sum_actions(result.solution, model_config.shape) == target)
                records.append({
                    "configuration": name, "max_moves": max_moves, "seed": seed,
                    "success": verified, "exact_reconstruction_verified": verified,
                    "terms": len(result.solution) if result.solution else None,
                    "factor_sequence_digest": _digest_solution(result.solution),
                    "solution": [{"u": list(a.u), "v": list(a.v), "w": list(a.w)} for a in result.solution] if result.solution else None,
                    "simulations_permitted": config.simulations, "simulations_completed": result.simulations_run,
                    "simulations_to_first_success": result.simulations_to_first_success,
                    "expanded_nodes": result.expanded_nodes, "unique_states": result.unique_states,
                    "transposition_hits": result.transposition_hits, "maximum_depth": result.maximum_depth,
                    "value_queries": result.value_evaluation_calls, "value_clips": result.value_prediction_clips,
                    "proposal_generation_seconds": result.proposal_generation_seconds,
                    "value_evaluation_seconds": result.value_inference_seconds,
                    "search_wall_seconds": result.search_wall_seconds,
                    "simulations_per_second": result.simulations_per_second,
                    "termination_reason": result.termination_reason,
                })
                trajectory.append({
                    "configuration": name, "max_moves": max_moves, "seed": seed,
                    "proposal_trace": policy.trace,
                    "first_simulations": list(result.simulation_trace),
                })
                if isinstance(evaluator, AffineNeuralValueEvaluator):
                    diagnostics.append({
                        "configuration": name, "max_moves": max_moves, "seed": seed,
                        "scale": evaluator.scale, "offset": evaluator.offset,
                        "ordering_preserved_by_transform": evaluator.scale > 0,
                        "raw_prediction_summary": _summary(evaluator.base.raw_predictions),
                        "original_value_summary": _summary(evaluator.original_values),
                        "transformed_value_summary": _summary(evaluator.transformed_values),
                        "clips": evaluator.metrics.clipped,
                        "clip_reasons": evaluator.base.clip_reasons,
                        "bounded_value_trace": evaluator.base.trace,
                    })
                if progress:
                    progress(records[-1])
    return records, trajectory, diagnostics


def _summary(values: Sequence[float]):
    return {
        "count": len(values), "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def save_value_scale(output: Path, config: ValueScaleConfig, records, trajectory, diagnostics, provenance):
    output.mkdir(parents=True, exist_ok=False)
    experiment = {
        "format_version": FORMAT_VERSION, "experiment": "tg3e_value_scale",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "configurations": list(CONFIGURATIONS), "transforms": TRANSFORMS,
        "permutation_control": {
            "implemented": False,
            "reason": "No preregistered fixed residual-state pool spans the value-dependent trees. Online assignment would introduce evaluation-order bias and would not preserve fixed state-value associations.",
        },
        "provenance": provenance,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    summary = {}
    for name in CONFIGURATIONS:
        group = [r for r in records if r["configuration"] == name]
        summary[name] = {
            "successes": sum(r["success"] for r in group), "trials": len(group),
            "mean_search_seconds": sum(r["search_wall_seconds"] for r in group) / len(group),
            "timeouts": sum(r["termination_reason"] == "wall_time_limit_reached" for r in group),
        }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps({"trials": records, "summary": summary}, indent=2, sort_keys=True) + "\n")
    (output / "trajectory_diagnostics.json").write_text(json.dumps({"trials": trajectory}, indent=2, sort_keys=True) + "\n")
    (output / "value_scale_diagnostics.json").write_text(json.dumps({"trials": diagnostics}, indent=2, sort_keys=True) + "\n")
    lines = ["# TG-3E MCTS value-scale study", "", "| Configuration | Successes | Trials | Mean search seconds | Timeouts |", "|---|---:|---:|---:|---:|"]
    for name in CONFIGURATIONS:
        row = summary[name]
        lines.append(f"| {name} | {row['successes']} | {row['trials']} | {row['mean_search_seconds']:.6f} | {row['timeouts']} |")
    lines.extend(("", "All success claims require independent exact reconstruction. Seven-term failures remain bounded-search outcomes.", ""))
    (output / "report.md").write_text("\n".join(lines))


def load_value_scale(output: Path):
    experiment = json.loads((output / "experiment.json").read_text())
    if experiment.get("format_version") != FORMAT_VERSION or experiment.get("experiment") != "tg3e_value_scale":
        raise ValueError("unsupported TG-3E evidence")
    return experiment, json.loads((output / "results.json").read_text())
