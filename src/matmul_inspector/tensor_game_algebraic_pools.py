"""Exact, bounded algebraic action-pool diagnostics for the Tensor Game."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Any, Iterable, Sequence

from .tensor_game import RankOneAction, Tensor, subtract_action, tensor_shape, validate_action
from .tensor_game_bounded_residual import VECTORS, action_identity, squared_norm
from .tensor_game_hardware_cost import action_cost
from .tensor_game_residual_lookahead import exact_matrix_rank, residual_subspace_features
from .tensor_game_search import canonicalize_action


def tensor_slices(tensor: Tensor, mode: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Extract exact matrices formed by fixing one tensor coordinate."""
    shape = tensor_shape(tensor)
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")
    other = tuple(index for index in range(3) if index != mode)
    result = []
    for fixed in range(shape[mode]):
        matrix = []
        for first in range(shape[other[0]]):
            row = []
            for second in range(shape[other[1]]):
                indices = [0, 0, 0]
                indices[mode] = fixed
                indices[other[0]] = first
                indices[other[1]] = second
                row.append(tensor[indices[0]][indices[1]][indices[2]])
            matrix.append(tuple(row))
        result.append(tuple(matrix))
    return tuple(result)


def combine_slices(tensor: Tensor, mode: int, coefficients: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return an exact linear combination of the slices along one mode."""
    slices = tensor_slices(tensor, mode)
    if len(coefficients) != len(slices):
        raise ValueError("coefficient count must equal the number of slices")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in coefficients):
        raise ValueError("slice coefficients must be integers")
    return tuple(tuple(sum(coefficient * slices[index][row][column]
                           for index, coefficient in enumerate(coefficients))
                       for column in range(len(slices[0][0])))
                 for row in range(len(slices[0])))


def exact_rref(matrix: Sequence[Sequence[int | Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    """Return deterministic reduced row echelon form over the rationals."""
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row != pivot_row and rows[row][column]:
                factor = rows[row][column]
                rows[row] = [value - factor * pivot_value for value, pivot_value in zip(rows[row], rows[pivot_row])]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows if any(row))


def exact_row_space_basis(matrix: Sequence[Sequence[int]]) -> tuple[tuple[Fraction, ...], ...]:
    return exact_rref(matrix)


def exact_column_space_basis(matrix: Sequence[Sequence[int]]) -> tuple[tuple[Fraction, ...], ...]:
    if not matrix:
        return ()
    return exact_rref(tuple(zip(*matrix)))


def exact_null_space_basis(matrix: Sequence[Sequence[int]]) -> tuple[tuple[Fraction, ...], ...]:
    """Return a rational basis for the right null space."""
    if not matrix:
        return ()
    rref = exact_rref(matrix)
    width = len(matrix[0])
    pivots = [next(index for index, value in enumerate(row) if value) for row in rref]
    free = [index for index in range(width) if index not in pivots]
    basis = []
    for free_column in free:
        vector = [Fraction(0) for _ in range(width)]
        vector[free_column] = 1
        for row, pivot in zip(rref, pivots):
            vector[pivot] = -row[free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def vector_in_row_space(vector: Sequence[int], matrix: Sequence[Sequence[int]]) -> bool:
    return exact_matrix_rank(matrix) == exact_matrix_rank(tuple(matrix) + (tuple(vector),))


def _direction(values: Iterable[int]) -> tuple[int, ...] | None:
    values = tuple(values)
    divisor = math.gcd(*(abs(value) for value in values))
    if divisor == 0:
        return None
    normalized = tuple(value // divisor for value in values)
    first = next(value for value in normalized if value)
    return normalized if first > 0 else tuple(-value for value in normalized)


@lru_cache(maxsize=1)
def _pair_direction_map(rows: int, columns: int) -> dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]]]:
    if rows != 4 or columns != 4:
        raise ValueError("the bounded algebraic generator currently supports 4x4 slices")
    result = {}
    for left in VECTORS:
        for right in VECTORS:
            direction = _direction(left_value * right_value for left_value in left for right_value in right)
            pair = min((left, right), (tuple(-x for x in left), tuple(-x for x in right)))
            if direction is not None:
                result.setdefault(direction, pair)
    return result


def factor_rank_one_matrix(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Map a nonzero exact rank-one 4x4 matrix direction to ternary factors."""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("matrix must have shape 4x4")
    if exact_matrix_rank(matrix) != 1:
        return None
    return _pair_direction_map(4, 4).get(_direction(value for row in matrix for value in row))


def _ternary_direction(values: Sequence[int]) -> tuple[int, ...] | None:
    direction = _direction(values)
    if direction is None or any(value not in (-1, 0, 1) for value in direction):
        return None
    return direction


def low_rank_matrix_factor_proposals(matrix: Sequence[Sequence[int]]) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Pair ternary column and row directions observed in a rank-at-most-two matrix."""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("matrix must have shape 4x4")
    rank = exact_matrix_rank(matrix)
    if rank == 0 or rank > 2:
        return ()
    rows = {_ternary_direction(tuple(row)) for row in matrix}
    columns = {_ternary_direction(tuple(matrix[row][column] for row in range(4))) for column in range(4)}
    rows.discard(None); columns.discard(None)
    return tuple(sorted((left, right) for left in columns for right in rows))  # type: ignore[misc]


def _action_from_mode_pair(mode: int, fixed: tuple[int, ...], pair: tuple[tuple[int, ...], tuple[int, ...]]) -> RankOneAction:
    other = tuple(index for index in range(3) if index != mode)
    factors: list[tuple[int, ...] | None] = [None, None, None]
    factors[mode] = fixed
    factors[other[0]], factors[other[1]] = pair
    return canonicalize_action(RankOneAction(factors[0], factors[1], factors[2]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class AlgebraicPoolConfig:
    combination_vectors: int = 80
    missing_factors_per_pair: int = 4
    maximum_pair_proposals: int = 240
    maximum_exact_scores: int = 19_200
    maximum_seconds: float = 10.0

    def validate(self) -> None:
        if not 1 <= self.combination_vectors <= len(VECTORS):
            raise ValueError("combination_vectors must be in [1, 80]")
        if not 1 <= self.missing_factors_per_pair <= len(VECTORS):
            raise ValueError("missing_factors_per_pair must be in [1, 80]")
        if min(self.maximum_pair_proposals, self.maximum_exact_scores) <= 0:
            raise ValueError("pair and score budgets must be positive")
        if not math.isfinite(self.maximum_seconds) or self.maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be finite and positive")


def algebraic_action_candidates(residual: Tensor, config: AlgebraicPoolConfig = AlgebraicPoolConfig()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate exact actions from rank-one ternary combinations of residual slices."""
    config.validate()
    if tensor_shape(residual) != (4, 4, 4):
        raise ValueError("algebraic pool currently supports shape (4, 4, 4)")
    started = time.perf_counter()
    deadline = started + config.maximum_seconds
    pair_provenance: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[dict[str, Any]]] = {}
    combinations_considered = rank_one_combinations = rank_two_combinations = domain_misses = 0
    stop_reason = "completed"
    for mode in range(3):
        for coefficients in VECTORS[:config.combination_vectors]:
            if time.perf_counter() >= deadline:
                stop_reason = "time_budget"; break
            combinations_considered += 1
            matrix = combine_slices(residual, mode, coefficients)
            matrix_rank = exact_matrix_rank(matrix)
            if matrix_rank > 2 or matrix_rank == 0:
                continue
            rank_one_combinations += matrix_rank == 1
            rank_two_combinations += matrix_rank == 2
            pairs = ((factor_rank_one_matrix(matrix),) if matrix_rank == 1 else low_rank_matrix_factor_proposals(matrix))
            pairs = tuple(pair for pair in pairs if pair is not None)
            if not pairs:
                domain_misses += 1
            for pair in pairs:
                pair_provenance.setdefault((mode, pair[0], pair[1]), []).append({
                    "combination": coefficients, "matrix_rank": matrix_rank,
                    "method": "rank_one_slice_combination" if matrix_rank == 1 else "rank_two_observed_row_column_directions",
                })
        if stop_reason != "completed":
            break

    pair_rows = []
    for (mode, left, right), provenance in sorted(pair_provenance.items(), key=lambda item: (
        min(row["matrix_rank"] for row in item[1]),
        min(sum(value != 0 for value in row["combination"]) for row in item[1]), item[0],
    ))[:config.maximum_pair_proposals]:
        pair = (left, right)
        pair_id = hashlib.sha256(json.dumps([mode, pair], separators=(",", ":")).encode()).hexdigest()
        pair_rows.append({"mode": mode, "pair": pair, "pair_id": pair_id, "sources": provenance})
    if len(pair_provenance) > config.maximum_pair_proposals:
        stop_reason = "pair_budget"

    candidates: dict[RankOneAction, dict[str, Any]] = {}
    exact_scores = 0
    for pair_rank, pair_row in enumerate(pair_rows, 1):
        scored = []
        for missing in VECTORS:
            if exact_scores >= config.maximum_exact_scores or time.perf_counter() >= deadline:
                stop_reason = "score_budget" if exact_scores >= config.maximum_exact_scores else "time_budget"
                break
            action = _action_from_mode_pair(pair_row["mode"], missing, pair_row["pair"])
            validate_action(action, (4, 4, 4), frozenset((-1, 0, 1)))
            score = squared_norm(subtract_action(residual, action))
            exact_scores += 1
            scored.append((score, action))
        for missing_rank, (score, action) in enumerate(sorted(set(scored), key=lambda item: (item[0], item[1].u, item[1].v, item[1].w))[:config.missing_factors_per_pair], 1):
            provenance = {"method": "low_rank_slice_subspace", "mode": pair_row["mode"],
                          "slice_sources": [{**source, "combination": list(source["combination"])} for source in pair_row["sources"]],
                          "pair_id": pair_row["pair_id"],
                          "pair_rank": pair_rank, "missing_factor_rank": missing_rank}
            row = candidates.get(action)
            if row is None:
                candidates[action] = {"action": action, "action_id": action_identity(action),
                                      "score": score, "provenance": [provenance]}
            else:
                row["provenance"].append(provenance)
        if stop_reason in ("score_budget", "time_budget"):
            break
    ordered = sorted(candidates.values(), key=lambda row: (
        min(item["missing_factor_rank"] for item in row["provenance"]),
        row["score"], row["action"].u, row["action"].v, row["action"].w,
    ))
    for rank, row in enumerate(ordered, 1):
        row["algebraic_rank"] = rank
    return ordered, {
        "schema": "tensor_game_algebraic_candidate_audit.v1",
        "config": config.__dict__, "slice_combinations_considered": combinations_considered,
        "rank_one_slice_combinations": rank_one_combinations,
        "rank_two_slice_combinations": rank_two_combinations,
        "rank_one_domain_mapping_misses": domain_misses, "pair_proposals": len(pair_rows),
        "distinct_pair_ids": len({row["pair_id"] for row in pair_rows}),
        "exact_scoring_operations": exact_scores, "unique_complete_actions": len(ordered),
        "duplicate_complete_actions": len(pair_rows) * config.missing_factors_per_pair - len(ordered),
        "stop_reason": stop_reason, "runtime_seconds": time.perf_counter() - started,
    }


def algebraic_pool(residual: Tensor, size: int, config: AlgebraicPoolConfig = AlgebraicPoolConfig()) -> tuple[tuple[RankOneAction, ...], dict[str, Any]]:
    if size <= 0:
        raise ValueError("pool size must be positive")
    rows, audit = algebraic_action_candidates(residual, config)
    return tuple(row["action"] for row in rows[:size]), audit


def hybrid_pool(algebraic: Sequence[RankOneAction], residual_ranked: Sequence[RankOneAction], size: int, algebraic_quota: int) -> tuple[RankOneAction, ...]:
    """Reserve an exact algebraic quota, then fill from residual rank order."""
    if not 0 <= algebraic_quota <= size:
        raise ValueError("algebraic quota must be in [0, size]")
    result = list(dict.fromkeys(canonicalize_action(action) for action in algebraic[:algebraic_quota]))
    for action in residual_ranked:
        action = canonicalize_action(action)
        if action not in result:
            result.append(action)
        if len(result) == size:
            break
    if len(result) != size:
        raise ValueError("source pools cannot fill requested hybrid size")
    return tuple(result)


def known_action_recall(pool: Sequence[RankOneAction], known: Sequence[RankOneAction]) -> dict[str, Any]:
    """Measure held-out factor, pair, action, and rank-one tensor equivalence recall."""
    raw_pool = tuple(pool)
    pool = tuple(canonicalize_action(action) for action in pool)
    pool_ids = {action_identity(action) for action in pool}
    from .tensor_game import outer_product
    tensor_set = {outer_product(action) for action in pool}
    normalize = lambda vector: tuple((1 if next(value for value in vector if value) > 0 else -1) * value for value in vector)
    pair_sets = [{(normalize(action.u), normalize(action.v)) for action in pool},
                 {(normalize(action.u), normalize(action.w)) for action in pool},
                 {(normalize(action.v), normalize(action.w)) for action in pool}]
    rows = []
    for index, raw in enumerate(known, 1):
        action = canonicalize_action(raw)
        pair_hits = [pair in pairs for pair, pairs in zip(((normalize(action.u), normalize(action.v)),
                     (normalize(action.u), normalize(action.w)), (normalize(action.v), normalize(action.w))), pair_sets)]
        rows.append({"known_index": index, "action_id": action_identity(action),
                     "exact_coefficient_identity": raw in raw_pool,
                     "coefficient_identity": action_identity(action) in pool_ids,
                     "tensor_equivalence": outer_product(action) in tensor_set,
                     "mode_pair_recall": pair_hits, "any_pair_recall": any(pair_hits)})
    return {"pool_size": len(pool), "pair_recall": sum(row["any_pair_recall"] for row in rows),
            "exact_coefficient_recall": sum(row["exact_coefficient_identity"] for row in rows),
            "complete_action_recall": sum(row["coefficient_identity"] for row in rows),
            "tensor_equivalence_recall": sum(row["tensor_equivalence"] for row in rows), "records": rows}


def candidate_diagnostics(residual: Tensor, actions: Sequence[RankOneAction], limit: int = 32) -> list[dict[str, Any]]:
    """Bound exact residual/subspace analysis to a deterministic action prefix."""
    rows = []
    before = residual_subspace_features(residual)
    for action in actions[:limit]:
        after_tensor = subtract_action(residual, action)
        after = residual_subspace_features(after_tensor)
        rows.append({"action_id": action_identity(action), "score": squared_norm(after_tensor),
                     "support": sum(value != 0 for vector in (action.u, action.v, action.w) for value in vector),
                     "maximum_absolute_coefficient": max(abs(value) for vector in (action.u, action.v, action.w) for value in vector),
                     "preparation_additions": action_cost(action).preparation_additions,
                     "flattening_ranks_before": before["flattening_ranks"], "flattening_ranks_after": after["flattening_ranks"],
                     "slice_ranks_before": before["slice_ranks"], "slice_ranks_after": after["slice_ranks"]})
    return rows
