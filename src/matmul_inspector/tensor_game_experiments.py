"""Reproducible TG-2B curriculum experiments and evidence persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from .tensor_game import (
    RankOneAction,
    Tensor,
    matrix_multiplication_tensor,
    outer_product,
    sum_actions,
    tensor_shape,
)
from .tensor_game_search import (
    HybridSampledPolicy,
    ResidualInformedPolicy,
    SampledMCTS,
    SearchConfig,
    SearchResult,
    UniformSampledPolicy,
)

FORMAT_VERSION = 1
STRATEGIES = ("uniform_sampled", "residual_informed", "hybrid")


@dataclass(frozen=True)
class CurriculumProblem:
    identifier: str
    category: str
    target: Tensor
    max_moves: int
    evidence: str


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (3, 11)
    simulations: int = 200
    max_actions: int = 12
    c_puct: float = 1.5
    node_limit: int = 200
    coefficients: tuple[int, ...] = (-1, 0, 1)
    hybrid_residual_probability: float = 0.5
    max_wall_time_seconds: float | None = 2.0

    def __post_init__(self) -> None:
        if not self.seeds or any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("at least one integer seed is required")
        SearchConfig(
            simulations=self.simulations,
            max_actions=self.max_actions,
            max_moves=1,
            c_puct=self.c_puct,
            node_limit=self.node_limit,
            max_wall_time_seconds=self.max_wall_time_seconds,
        )
        if not 0 <= self.hybrid_residual_probability <= 1:
            raise ValueError("hybrid_residual_probability must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            seeds=tuple(data["seeds"]),
            simulations=data["simulations"],
            max_actions=data["max_actions"],
            c_puct=data["c_puct"],
            node_limit=data["node_limit"],
            coefficients=tuple(data["coefficients"]),
            hybrid_residual_probability=data["hybrid_residual_probability"],
            max_wall_time_seconds=data.get("max_wall_time_seconds"),
        )


def curriculum() -> tuple[CurriculumProblem, ...]:
    rank_one_action = RankOneAction((1, -1), (1, 1), (1, -1))
    first = RankOneAction((1, 0), (1, 0), (1, 0))
    second = RankOneAction((0, 1), (0, 1), (0, 1))
    rank_two = sum_actions((first, second), (2, 2, 2))
    return (
        CurriculumProblem(
            "level1_rank_one_2x2x2",
            "synthetic_rank_one",
            outer_product(rank_one_action),
            1,
            "constructed from one nonzero rank-one term",
        ),
        CurriculumProblem(
            "level2_rank_two_2x2x2",
            "synthetic_rank_two",
            rank_two,
            2,
            "two-term upper bound; mode-0 flattening has exact matrix rank 2",
        ),
        CurriculumProblem(
            "level3_matmul_1x2x1",
            "matrix_multiplication",
            matrix_multiplication_tensor(1, 2, 1),
            2,
            "exact 1x2 by 2x1 matrix multiplication tensor",
        ),
        CurriculumProblem(
            "level4_matmul_2x2_eight",
            "matrix_multiplication",
            matrix_multiplication_tensor(2, 2, 2),
            8,
            "independent search with an eight-action limit",
        ),
        CurriculumProblem(
            "level5_matmul_2x2_seven",
            "matrix_multiplication",
            matrix_multiplication_tensor(2, 2, 2),
            7,
            "independent stretch search with a seven-action limit",
        ),
    )


def exact_matrix_rank(rows: Sequence[Sequence[int]]) -> int:
    """Compute matrix rank by exact rational elimination."""
    matrix = [[Fraction(value) for value in row] for row in rows]
    if not matrix:
        return 0
    rank = 0
    for column in range(len(matrix[0])):
        pivot = next((row for row in range(rank, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(len(matrix)):
            if row != rank and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[rank])]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def mode_zero_flattening_rank(tensor: Tensor) -> int:
    _, y, z = tensor_shape(tensor)
    return exact_matrix_rank([[value for row in plane for value in row] for plane in tensor])


def tensor_digest(tensor: Tensor) -> str:
    payload = json.dumps(tensor, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy(name: str, shape: tuple[int, int, int], config: ExperimentConfig):
    if name == "uniform_sampled":
        return UniformSampledPolicy(shape, config.coefficients)
    if name == "residual_informed":
        return ResidualInformedPolicy(shape, config.coefficients)
    if name == "hybrid":
        return HybridSampledPolicy(
            shape,
            residual_probability=config.hybrid_residual_probability,
            coefficients=config.coefficients,
        )
    raise ValueError(f"unknown strategy: {name}")


def action_to_dict(action: RankOneAction) -> dict[str, list[int]]:
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def _result_record(problem: CurriculumProblem, strategy: str, seed: int, result: SearchResult) -> dict[str, Any]:
    solution = [action_to_dict(action) for action in result.solution] if result.solution else None
    return {
        "target_id": problem.identifier,
        "target_category": problem.category,
        "target_digest_sha256": tensor_digest(problem.target),
        "tensor_shape": list(tensor_shape(problem.target)),
        "strategy": strategy,
        "seed": seed,
        "success": result.exact_reconstruction,
        "exact_reconstruction": result.exact_reconstruction,
        "terms": result.scalar_multiplications,
        "solution": solution,
        "termination_reason": result.termination_reason,
        "simulations_permitted": result.config.simulations,
        "simulations_completed": result.simulations_run,
        "node_limit": result.config.node_limit,
        "unique_states": result.unique_states,
        "expanded_nodes": result.expanded_nodes,
        "transposition_hits": result.transposition_hits,
        "maximum_depth": result.maximum_depth,
        "proposal_attempts": result.proposal_attempts,
        "valid_proposals": result.valid_proposals,
        "accepted_proposals": result.accepted_proposals,
        "duplicate_proposals": result.duplicate_proposals,
        "proposal_generation_seconds": result.proposal_generation_seconds,
        "search_wall_seconds": result.search_wall_seconds,
        "simulations_per_second": result.simulations_per_second,
    }


def run_trials(
    config: ExperimentConfig,
    problems: Sequence[CurriculumProblem] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for problem in problems or curriculum():
        shape = tensor_shape(problem.target)
        for strategy in STRATEGIES:
            for seed in config.seeds:
                search_config = SearchConfig(
                    simulations=config.simulations,
                    max_actions=config.max_actions,
                    max_moves=problem.max_moves,
                    seed=seed,
                    c_puct=config.c_puct,
                    node_limit=config.node_limit,
                    max_wall_time_seconds=config.max_wall_time_seconds,
                )
                result = SampledMCTS(problem.target, _policy(strategy, shape, config), search_config).run()
                record = _result_record(problem, strategy, seed, result)
                records.append(record)
                if progress is not None:
                    progress(record)
    return records


def _markdown(records: Sequence[dict[str, Any]], config: ExperimentConfig) -> str:
    lines = [
        "# TG-2B independent discovery comparison",
        "",
        "These CPU results compare matched MCTS budgets. No policy received known standard or Strassen factors.",
        "A stopped or unsuccessful trial is not evidence that a decomposition does not exist.",
        "",
        f"Seeds: `{list(config.seeds)}`. Simulations per trial: `{config.simulations}`. "
        f"Proposals per expansion: `{config.max_actions}`. Node limit: `{config.node_limit}`.",
        "",
        "| Target | Strategy | Successes | Trials | Mean completed simulations | Mean wall time (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    targets = list(dict.fromkeys(record["target_id"] for record in records))
    for target in targets:
        for strategy in STRATEGIES:
            group = [r for r in records if r["target_id"] == target and r["strategy"] == strategy]
            successes = sum(record["success"] for record in group)
            mean_simulations = sum(record["simulations_completed"] for record in group) / len(group)
            mean_wall = sum(record["search_wall_seconds"] for record in group) / len(group)
            lines.append(
                f"| {target} | {strategy} | {successes} | {len(group)} | "
                f"{mean_simulations:.1f} | {mean_wall:.6f} |"
            )
    lines.extend(
        [
            "",
            "Success means exact reconstruction within the configured move limit. Timing is CPU wall time and includes proposal generation and MCTS bookkeeping.",
            "Level 5 success, if any, must be inspected from the exact saved factors before making a rediscovery claim.",
            "",
        ]
    )
    return "\n".join(lines)


def save_experiment(output: Path, config: ExperimentConfig, records: Sequence[dict[str, Any]]) -> None:
    """Create a new evidence directory without overwriting prior results."""
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "tg2b_independent_tensor_decomposition",
        "config": asdict(config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "curriculum": [
            {
                "identifier": item.identifier,
                "category": item.category,
                "shape": list(tensor_shape(item.target)),
                "digest_sha256": tensor_digest(item.target),
                "max_moves": item.max_moves,
                "evidence": item.evidence,
            }
            for item in curriculum()
        ],
        "trials": list(records),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(records, config), encoding="utf-8")


def load_config(path: Path) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported TG-2B experiment format version")
    return ExperimentConfig.from_dict(payload["config"])
