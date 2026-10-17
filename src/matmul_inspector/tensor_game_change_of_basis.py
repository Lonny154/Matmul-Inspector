"""Exact bounded change-of-basis variants of 2x2 bilinear algorithms."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import random
from typing import Iterable, Sequence

from .tensor_game import RankOneAction, standard_2x2_decomposition, strassen_2x2_decomposition
from .tensor_game_candidate_dataset import _record

Matrix2 = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ChangeOfBasisConfig:
    entry_bound: int = 1
    combination_limit: int = 512
    mode: str = "sample"
    seed: int = 20261016
    maximum_coefficient: int = 8
    maximum_total_support: int = 96
    maximum_pre_additions: int = 64
    maximum_post_additions: int = 32

    def __post_init__(self) -> None:
        if self.entry_bound <= 0 or self.combination_limit <= 0:
            raise ValueError("entry bound and combination limit must be positive")
        if self.mode not in {"enumerate", "sample"}:
            raise ValueError("mode must be 'enumerate' or 'sample'")
        if min(self.maximum_coefficient, self.maximum_total_support, self.maximum_pre_additions, self.maximum_post_additions) <= 0:
            raise ValueError("practical limits must be positive")


def determinant(matrix: Matrix2) -> int:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def inverse_unimodular(matrix: Matrix2) -> Matrix2:
    det = determinant(matrix)
    if det not in (-1, 1):
        raise ValueError("matrix must have determinant +1 or -1")
    a, b = matrix[0]
    c, d = matrix[1]
    return ((d // det, -b // det), (-c // det, a // det))


def multiply(left: Matrix2, right: Matrix2) -> Matrix2:
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(2)) for j in range(2)) for i in range(2))  # type: ignore[return-value]


def transpose(matrix: Matrix2) -> Matrix2:
    return ((matrix[0][0], matrix[1][0]), (matrix[0][1], matrix[1][1]))


def _as_matrix(vector: Sequence[int]) -> Matrix2:
    if len(vector) != 4:
        raise ValueError("factor must contain four row-major coefficients")
    return ((vector[0], vector[1]), (vector[2], vector[3]))


def _flatten(matrix: Matrix2) -> tuple[int, int, int, int]:
    return matrix[0] + matrix[1]


def transform_decomposition(actions: Sequence[RankOneAction], p: Matrix2, q: Matrix2, r: Matrix2) -> tuple[RankOneAction, ...]:
    """Apply A'=P A Q^-1, B'=Q B R^-1, AB=P^-1(A'B')R."""
    pinv, qinv, rinv = map(inverse_unimodular, (p, q, r))
    return tuple(
        RankOneAction(
            _flatten(multiply(multiply(transpose(p), _as_matrix(action.u)), transpose(qinv))),
            _flatten(multiply(multiply(transpose(q), _as_matrix(action.v)), transpose(rinv))),
            _flatten(multiply(multiply(pinv, _as_matrix(action.w)), r)),
        )
        for action in actions
    )


def unimodular_matrices(entry_bound: int) -> tuple[Matrix2, ...]:
    if entry_bound <= 0:
        raise ValueError("entry bound must be positive")
    values = range(-entry_bound, entry_bound + 1)
    matrices = (((a, b), (c, d)) for a, b, c, d in itertools.product(values, repeat=4))
    return tuple(matrix for matrix in matrices if determinant(matrix) in (-1, 1))


def transformation_triples(config: ChangeOfBasisConfig) -> tuple[tuple[int, Matrix2, Matrix2, Matrix2], ...]:
    matrices = unimodular_matrices(config.entry_bound)
    total = len(matrices) ** 3
    count = min(config.combination_limit, total)
    if config.mode == "enumerate":
        iterator = itertools.islice(itertools.product(matrices, repeat=3), count)
        return tuple((index, p, q, r) for index, (p, q, r) in enumerate(iterator))
    rng = random.Random(config.seed)
    indices = rng.sample(range(total), count)
    size = len(matrices)
    triples = []
    for index in indices:
        p_index, remainder = divmod(index, size * size)
        q_index, r_index = divmod(remainder, size)
        triples.append((index, matrices[p_index], matrices[q_index], matrices[r_index]))
    return tuple(triples)


def _matrix_payload(matrix: Matrix2) -> list[list[int]]:
    return [list(row) for row in matrix]


def _total_support(actions: Sequence[RankOneAction]) -> int:
    return sum(value != 0 for action in actions for value in action.u + action.v + action.w)


def generate_change_of_basis_candidates(config: ChangeOfBasisConfig) -> tuple[list[dict], dict]:
    seeds = (("conventional", standard_2x2_decomposition()), ("strassen", strassen_2x2_decomposition()))
    triples = transformation_triples(config)
    accepted: dict[str, dict] = {}
    rejections: Counter[str] = Counter()
    valid_before_limits = 0
    for family, actions in seeds:
        for enumeration_index, p, q, r in triples:
            try:
                transformed = transform_decomposition(actions, p, q, r)
                record = _record(f"{family}-cob-{enumeration_index}", family, transformed, [])
                valid_before_limits += 1
                max_coefficient = max(abs(value) for action in transformed for value in action.u + action.v + action.w)
                support = _total_support(transformed)
                cost = record["static_cost"]
                if max_coefficient > config.maximum_coefficient:
                    rejections["maximum_coefficient"] += 1
                    continue
                if support > config.maximum_total_support:
                    rejections["maximum_total_support"] += 1
                    continue
                if cost["pre_additions"] > config.maximum_pre_additions:
                    rejections["maximum_pre_additions"] += 1
                    continue
                if cost["post_additions"] > config.maximum_post_additions:
                    rejections["maximum_post_additions"] += 1
                    continue
                record["parent_decomposition"] = family
                record["known_equivalence"] = "exact GL(2,Z)^3 change of basis"
                record["change_of_basis"] = {
                    "P": _matrix_payload(p), "Q": _matrix_payload(q), "R": _matrix_payload(r),
                    "determinants": [determinant(p), determinant(q), determinant(r)],
                    "inverses": [_matrix_payload(inverse_unimodular(x)) for x in (p, q, r)],
                    "enumeration_index": enumeration_index,
                    "selection_mode": config.mode,
                    "selection_seed": config.seed if config.mode == "sample" else None,
                }
                record["generator_configuration"] = asdict(config)
                record["total_coefficient_support"] = support
                accepted.setdefault(record["candidate_id"], record)
            except ValueError as error:
                rejections[f"exact_validation: {error}"] += 1
    records = sorted(accepted.values(), key=lambda row: row["candidate_id"])
    arithmetic = lambda row: (
        row["rank"], row["static_cost"]["pre_additions"], row["static_cost"]["post_additions"],
        row["static_cost"]["coefficient_scalings"],
    )
    summary = {
        "unimodular_matrix_count": len(unimodular_matrices(config.entry_bound)),
        "transformation_triples_selected": len(triples),
        "attempted_transformations": len(seeds) * len(triples),
        "exact_valid_before_limits": valid_before_limits,
        "rejected_transformations": sum(rejections.values()),
        "rejection_reasons": dict(sorted(rejections.items())),
        "unique_exact_representations": len(records),
        "duplicate_exact_representations": valid_before_limits - sum(rejections.values()) - len(records),
        "limited_canonical_classes": len({row["canonical_id"] for row in records}),
        "structural_signatures": len({row["structural_signature_id"] for row in records}),
        "arithmetic_cost_points": len({arithmetic(row) for row in records}),
        "family_distribution": dict(sorted(Counter(row["family_id"] for row in records).items())),
        "pre_addition_distribution": dict(sorted(Counter(row["static_cost"]["pre_additions"] for row in records).items())),
        "post_addition_distribution": dict(sorted(Counter(row["static_cost"]["post_additions"] for row in records).items())),
        "coefficient_scaling_distribution": dict(sorted(Counter(row["static_cost"]["coefficient_scalings"] for row in records).items())),
        "total_support_distribution": dict(sorted(Counter(row["total_coefficient_support"] for row in records).items())),
        "maximum_coefficient_distribution": dict(sorted(Counter(row["structural_signature"]["max_abs_coefficient"] for row in records).items())),
    }
    return records, summary


def select_cost_representatives(records: Iterable[dict], limit: int = 12) -> list[dict]:
    """Select stable representatives covering families and arithmetic costs."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    ordered = sorted(records, key=lambda row: (
        row["static_cost"]["pre_additions"] + row["static_cost"]["post_additions"] + row["static_cost"]["coefficient_scalings"],
        row["total_coefficient_support"], row["candidate_id"],
    ))
    selected, seen = [], set()
    for row in ordered:
        point = (row["family_id"], row["static_cost"]["pre_additions"], row["static_cost"]["post_additions"], row["static_cost"]["coefficient_scalings"], row["structural_signature"]["max_abs_coefficient"])
        if point in seen:
            continue
        seen.add(point); selected.append(row)
        if len(selected) == limit:
            break
    return selected
