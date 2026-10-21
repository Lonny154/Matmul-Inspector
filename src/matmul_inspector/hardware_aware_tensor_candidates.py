"""CPU-only exact rank-8 basis variants and static multi-objective analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from .tensor_game import RankOneAction, standard_2x2_decomposition, strassen_2x2_decomposition
from .tensor_game_candidate_dataset import _record, canonical_actions, validate_decomposition
from .tensor_game_change_of_basis import (
    ChangeOfBasisConfig, Matrix2, determinant, inverse_unimodular,
    transform_decomposition, transformation_triples,
)

# All four dimensions are minimized. Rank is fixed at eight and is not an objective.
PARETO_DIMENSIONS = ("pre_additions", "post_additions", "coefficient_scalings", "total_support")
DESCRIPTOR_KEYS = PARETO_DIMENSIONS + ("nonzero_coefficients", "coefficient_l1", "max_abs_coefficient")


@dataclass(frozen=True)
class Rank8Config:
    seed: int = 20261018
    max_attempts: int = 64  # Includes the conventional anchor.
    max_accepted: int = 64  # Unique canonical decompositions, including anchor.
    entry_bound: int = 1
    maximum_coefficient: int = 8
    maximum_total_support: int = 96
    maximum_pre_additions: int = 64
    maximum_post_additions: int = 32
    representative_count: int = 16
    mode: str = "sample"
    include_rank7_reference: bool = True

    def __post_init__(self) -> None:
        if min(self.max_attempts, self.max_accepted, self.entry_bound,
               self.maximum_coefficient, self.maximum_total_support,
               self.maximum_pre_additions, self.maximum_post_additions,
               self.representative_count) <= 0:
            raise ValueError("all limits must be positive")
        if self.mode not in {"sample", "enumerate"}:
            raise ValueError("mode must be sample or enumerate")


def descriptor_vector(row: dict) -> dict[str, int]:
    """Keep the established decomposition cost and coefficient counts separate."""
    coefficients = [value for action in row["actions"] for factor in ("u", "v", "w") for value in action[factor]]
    support = sum(value != 0 for value in coefficients)
    return {
        "pre_additions": row["static_cost"]["pre_additions"],
        "post_additions": row["static_cost"]["post_additions"],
        "coefficient_scalings": row["static_cost"]["coefficient_scalings"],
        "total_support": support,
        "nonzero_coefficients": support,
        "coefficient_l1": sum(abs(value) for value in coefficients),
        "max_abs_coefficient": max(map(abs, coefficients)),
    }


def _matrix_payload(matrix: Matrix2) -> list[list[int]]:
    return [list(part) for part in matrix]


def validated_candidate(actions: Sequence[RankOneAction], provenance: dict, config: Rank8Config,
                        *, enforce_limits: bool = True) -> dict:
    """Independently reconstruct all tensor entries before canonical admission."""
    if len(actions) != 8:
        raise ValueError("candidate must have exactly eight terms")
    validate_decomposition(actions)
    canonical = canonical_actions(actions)
    row = _record("conventional", "conventional", canonical, [provenance])
    vector = descriptor_vector(row)
    for name, value, limit in (
        ("maximum_coefficient", vector["max_abs_coefficient"], config.maximum_coefficient),
        ("maximum_total_support", vector["total_support"], config.maximum_total_support),
        ("maximum_pre_additions", vector["pre_additions"], config.maximum_pre_additions),
        ("maximum_post_additions", vector["post_additions"], config.maximum_post_additions),
    ):
        if enforce_limits and value > limit:
            raise ValueError(name)
    row.update({
        "canonical_candidate_hash": row["canonical_id"],
        "exact_reconstruction_status": "PASS",
        "descriptor_vector": vector,
        "source_transformation": provenance,
        "generation_paths": [provenance],
        "equivalence_status": "known conventional rank-8 GL(2,Z)^3 representation",
        "coefficient_bounds": {"configured_max_abs": config.maximum_coefficient, "observed_max_abs": vector["max_abs_coefficient"]},
    })
    return row


def dominates(left: dict[str, int], right: dict[str, int], dimensions: Sequence[str] = PARETO_DIMENSIONS) -> bool:
    return all(left[key] <= right[key] for key in dimensions) and any(left[key] < right[key] for key in dimensions)


def annotate_pareto(records: list[dict]) -> list[dict]:
    """Stable O(n^2) frontier; tied descriptor vectors are all nondominated."""
    ordered = sorted(records, key=lambda row: row["canonical_candidate_hash"])
    for row in ordered:
        count = sum(dominates(other["descriptor_vector"], row["descriptor_vector"])
                    for other in ordered if other is not row)
        row["pareto_dominator_count"] = count
        row["pareto_dominated"] = count > 0
    return [row for row in ordered if not row["pareto_dominated"]]


def descriptor_ranges(rows: Sequence[dict]) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {}
    return {name: {"minimum": min(values), "median": median(values), "maximum": max(values)}
            for name in DESCRIPTOR_KEYS
            for values in ([row["descriptor_vector"][name] for row in rows],)}


def structural_groups(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["structural_signature_id"]].append(row)
    return [{"structural_signature_id": key, "structural_signature": group[0]["structural_signature"],
             "canonical_decomposition_count": len(group), "descriptor_ranges": descriptor_ranges(group),
             "canonical_candidate_hashes": sorted(row["canonical_candidate_hash"] for row in group)}
            for key, group in sorted(groups.items())]


def select_representatives(frontier: Sequence[dict], baseline_hash: str, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    ordered = sorted(frontier, key=lambda row: row["canonical_candidate_hash"])
    anchor = next((row for row in ordered if row["canonical_candidate_hash"] == baseline_hash), None)
    if anchor is None:
        raise ValueError("conventional baseline is missing from Pareto frontier")
    selected: dict[str, dict] = {}

    def add(row: dict, reason: str) -> None:
        key = row["canonical_candidate_hash"]
        if key in selected:
            if reason not in selected[key]["selection_reasons"]:
                selected[key]["selection_reasons"].append(reason)
        elif len(selected) < limit:
            selected[key] = {"canonical_candidate_hash": key, "structural_signature_id": row["structural_signature_id"],
                             "descriptor_vector": row["descriptor_vector"], "selection_reasons": [reason]}

    add(anchor, "conventional baseline")
    for name in DESCRIPTOR_KEYS:
        add(min(ordered, key=lambda row: (row["descriptor_vector"][name], row["canonical_candidate_hash"])), f"minimum {name}")
        add(max(ordered, key=lambda row: (row["descriptor_vector"][name], row["canonical_candidate_hash"])), f"maximum {name} on frontier")
    for row in ordered:
        if row["structural_signature_id"] not in {item["structural_signature_id"] for item in selected.values()}:
            add(row, "distinct structural signature")
    for row in ordered:
        add(row, "additional Pareto candidate")
    return list(selected.values())


def run_study(config: Rank8Config) -> dict:
    baseline_actions = standard_2x2_decomposition()
    baseline = validated_candidate(baseline_actions, {"type": "literal_conventional_anchor"},
                                   config, enforce_limits=False)
    baseline["equivalent_generation_path_count"] = 1
    accepted = {baseline["canonical_candidate_hash"]: baseline}
    exact_valid = 1
    attempts = 1
    duplicates = 0
    rejections: Counter[str] = Counter()
    matrix_config = ChangeOfBasisConfig(entry_bound=config.entry_bound,
                                         combination_limit=max(1, config.max_attempts - 1),
                                         mode=config.mode, seed=config.seed)
    if config.max_attempts > 1:
        triples = transformation_triples(matrix_config)
        for index, p, q, r in triples:
            if attempts >= config.max_attempts or len(accepted) >= config.max_accepted:
                break
            attempts += 1
            provenance = {"type": "exact_unimodular_change_of_basis", "P": _matrix_payload(p),
                          "Q": _matrix_payload(q), "R": _matrix_payload(r),
                          "determinants": [determinant(x) for x in (p, q, r)],
                          "inverses": [_matrix_payload(inverse_unimodular(x)) for x in (p, q, r)],
                          "enumeration_index": index, "selection_mode": config.mode,
                          "selection_seed": config.seed if config.mode == "sample" else None}
            try:
                actions = transform_decomposition(baseline_actions, p, q, r)
                validate_decomposition(actions)
                exact_valid += 1
                row = validated_candidate(actions, provenance, config)
            except ValueError as exc:
                rejections[str(exc)] += 1
                continue
            key = row["canonical_candidate_hash"]
            if key in accepted:
                duplicates += 1
                accepted[key]["equivalent_generation_path_count"] += 1
                accepted[key]["generation_paths"].append(provenance)
            else:
                row["equivalent_generation_path_count"] = 1
                accepted[key] = row
    rows = sorted(accepted.values(), key=lambda row: row["canonical_candidate_hash"])
    frontier = annotate_pareto(rows)
    groups = structural_groups(rows)
    representatives = select_representatives(frontier, baseline["canonical_candidate_hash"], config.representative_count)
    frontier_signatures = len({row["structural_signature_id"] for row in frontier})
    # A tiny pilot cannot characterize diversity. This gate never predicts speed.
    frontier_cost_points = len({tuple(row["descriptor_vector"][key] for key in PARETO_DIMENSIONS)
                                for row in frontier})
    if attempts < 128:
        decision = "INCONCLUSIVE"
    elif len(frontier) >= 10 and frontier_signatures >= 3 and frontier_cost_points >= 2:
        decision = "PROCEED_GPU_STUDY"
    else:
        decision = "LOW_DIVERSITY"
    summary = {
        "generation_attempts": attempts, "exact_valid_candidates": exact_valid,
        "rejected_candidates": sum(rejections.values()), "rejection_reasons": dict(sorted(rejections.items())),
        "duplicate_candidates": duplicates, "unique_canonical_rank8_candidates": len(rows),
        "unique_structural_signatures": len(groups), "pareto_front_size": len(frontier),
        "unique_frontier_signatures": frontier_signatures, "representative_count": len(representatives),
        "frontier_cost_points": frontier_cost_points,
        "descriptor_ranges_full": descriptor_ranges(rows), "descriptor_ranges_frontier": descriptor_ranges(frontier),
        "conventional_baseline_descriptor_vector": baseline["descriptor_vector"],
        "transformation_acceptance_rates": {"exact_valid_per_attempt": exact_valid / attempts,
                                            "unique_per_attempt": len(rows) / attempts},
        "decision": decision,
    }
    rank7 = None
    if config.include_rank7_reference:
        rank7 = _record("known-strassen-reference", "strassen", canonical_actions(strassen_2x2_decomposition()), [])
        rank7.update({"descriptor_vector": descriptor_vector(rank7), "reference_status": "known rank-7 control; excluded from rank-8 frontier"})
    return {"config": asdict(config), "baseline": baseline, "candidates": rows,
            "structural_signatures": groups, "pareto_frontier": frontier,
            "representatives": representatives, "rank7_reference": rank7, "aggregate_summary": summary}
