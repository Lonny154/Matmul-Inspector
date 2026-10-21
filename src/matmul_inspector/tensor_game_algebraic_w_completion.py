"""Offline output-vector completion for frozen algebraic Tensor Game pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Any, Literal, Sequence

from .tensor_game import RankOneAction, Tensor, outer_product, tensor_shape, validate_action
from .tensor_game_algebraic_pools import combine_slices, low_rank_matrix_factor_proposals
from .tensor_game_bounded_residual import COEFFICIENTS, VECTORS, action_identity, exact_score_from_contraction, squared_norm
from .tensor_game_hardware_cost import action_cost
from .tensor_game_quota_w import top_k_w_vectors
from .tensor_game_residual_lookahead import exact_matrix_rank
from .tensor_game_search import canonicalize_action

CompletionStrategy = Literal["residual", "structural", "slice_projection", "hybrid"]


def _stable_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _pair_key(u: Sequence[int], v: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def normalize(vector: Sequence[int]) -> tuple[int, ...]:
        sign = 1 if next(value for value in vector if value) > 0 else -1
        return tuple(sign * value for value in vector)
    return normalize(u), normalize(v)


def _pair_id(mode: int, u: Sequence[int], v: Sequence[int]) -> str:
    u_key, v_key = _pair_key(u, v)
    return hashlib.sha256(_stable_bytes({"source_mode": mode, "u": list(u_key), "v": list(v_key)})).hexdigest()


def _contraction(residual: Tensor, u: Sequence[int], v: Sequence[int]) -> tuple[int, ...]:
    return tuple(sum(residual[i][j][k] * u[i] * v[j] for i in range(4) for j in range(4)) for k in range(4))


def frozen_algebraic_uv_pairs(residual: Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce all prior mode-aware pairs, frozen as coefficient-pair inputs."""
    if tensor_shape(residual) != (4, 4, 4):
        raise ValueError("frozen algebraic pairs require shape (4, 4, 4)")
    pair_sources: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[dict[str, Any]]] = {}
    rank_histogram = Counter()
    for mode in range(3):
        for coefficients in VECTORS:
            matrix = combine_slices(residual, mode, coefficients)
            rank = exact_matrix_rank(matrix)
            rank_histogram[(mode, rank)] += 1
            if rank != 2:
                continue
            for pair in low_rank_matrix_factor_proposals(matrix):
                pair_sources.setdefault((mode, pair[0], pair[1]), []).append({
                    "method": "rank_two_observed_row_column_directions",
                    "source_mode": mode, "combination": list(coefficients), "matrix_rank": rank,
                })
    ordered = sorted(pair_sources.items(), key=lambda item: (
        min(source["matrix_rank"] for source in item[1]),
        min(sum(value != 0 for value in source["combination"]) for source in item[1]),
        item[0],
    ))
    rows = []
    common = math.factorial(16)
    for rank, ((mode, u, v), provenance) in enumerate(ordered, 1):
        q = sum(value * value for value in u) * sum(value * value for value in v)
        contraction = _contraction(residual, u, v)
        rows.append({
            "pair_id": _pair_id(mode, u, v), "original_candidate_order": rank, "source_mode": mode,
            "u": list(u), "v": list(v), "q": q, "contraction": list(contraction),
            "heuristic_bound_numerator": (common // q) * sum(value * value for value in contraction),
            "u_support": sum(value != 0 for value in u), "v_support": sum(value != 0 for value in v),
            "u_negative": sum(value < 0 for value in u), "v_negative": sum(value < 0 for value in v),
            "provenance": provenance,
        })
    payload = {"schema": "tensor_game_frozen_algebraic_uv_pairs.v1", "pairs": rows}
    return rows, {"pair_count": len(rows), "source_modes": [0, 1, 2],
                  "combination_rank_histogram": [{"mode": mode, "rank": rank, "count": count}
                                                   for (mode, rank), count in sorted(rank_histogram.items())],
                  "content_sha256": hashlib.sha256(_stable_bytes(payload)).hexdigest()}


def _sign_pattern(vector: Sequence[int]) -> tuple[int, ...]:
    return tuple(0 if value == 0 else (1 if value > 0 else -1) for value in vector)


def _hamming(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(left, right))


def _scored_w_rows(contraction: Sequence[int], q: int) -> list[dict[str, Any]]:
    rows = []
    for w in VECTORS:
        dot = sum(value * coefficient for value, coefficient in zip(contraction, w))
        support = sum(value != 0 for value in w)
        rows.append({"w": w, "score_delta": exact_score_from_contraction(0, contraction, q, w),
                     "dot": dot, "support": support, "sign_pattern": _sign_pattern(w),
                     "projection_strength": Fraction(dot * dot, support)})
    return rows


def complete_w_vectors(contraction: Sequence[int], q: int, count: int, strategy: CompletionStrategy) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select bounded legal output vectors using one deterministic strategy."""
    if len(contraction) != 4 or q <= 0:
        raise ValueError("completion requires a four-coordinate contraction and positive q")
    if count <= 0 or count > 80:
        raise ValueError("count must be in [1, 80]")
    if strategy not in ("residual", "structural", "slice_projection", "hybrid"):
        raise ValueError("unknown completion strategy")
    started = time.perf_counter_ns()
    scoring_started = time.perf_counter_ns()
    scored = _scored_w_rows(contraction, q)
    scoring_ms = (time.perf_counter_ns() - scoring_started) / 1e6
    top_k_started = time.perf_counter_ns()
    residual_rows, top_k_audit = top_k_w_vectors(contraction, q, count)
    top_k_ms = (time.perf_counter_ns() - top_k_started) / 1e6
    by_w = {row["w"]: row for row in scored}
    residual = [by_w[row["w"]] | {"source": "exact_residual_top_k", "residual_rank": row["rank"]} for row in residual_rows]

    def structural_selection(limit: int) -> list[dict[str, Any]]:
        selected = [residual[0] | {"source": "structural_residual_anchor"}]
        remaining = [row for row in scored if row["w"] != selected[0]["w"]]
        while len(selected) < limit:
            used_supports = {row["support"] for row in selected}
            used_signs = {row["sign_pattern"] for row in selected}
            choice = min(remaining, key=lambda row: (
                row["support"] in used_supports, row["sign_pattern"] in used_signs,
                -min(_hamming(row["w"], item["w"]) for item in selected),
                row["score_delta"], row["w"],
            ))
            remaining.remove(choice); selected.append(choice | {"source": "structural_maximin"})
        return selected

    slice_rows = sorted(scored, key=lambda row: (
        row["dot"] <= 0, -row["projection_strength"], -row["dot"], row["support"], row["w"],
    ))
    slice_selected = [row | {"source": "exact_slice_projection"} for row in slice_rows[:count]]
    if strategy == "residual":
        selected = residual
    elif strategy == "structural":
        selected = structural_selection(count)
    elif strategy == "slice_projection":
        selected = slice_selected
    else:
        residual_quota = (count + 1) // 2
        selected = list(residual[:residual_quota])
        for row in structural_selection(count):
            if row["w"] not in {item["w"] for item in selected}:
                selected.append(row | {"source": "hybrid_structural"})
            if len(selected) == count:
                break
    for selection_rank, row in enumerate(selected, 1):
        row["selection_rank"] = selection_rank
    return selected, {
        "strategy": strategy, "requested": count, "returned": len(selected),
        "valid_w_domain": 80, "exhaustive_exact_scores_for_matched_audit": 80,
        "top_k_complete_vectors_scored": top_k_audit["complete_vectors_scored"],
        "matched_exhaustive_scoring_ms": scoring_ms, "existing_top_k_ms": top_k_ms,
        "runtime_ms": (time.perf_counter_ns() - started) / 1e6,
    }


@dataclass(frozen=True)
class CompletionConfig:
    strategy: CompletionStrategy
    w_per_pair: int
    final_pool_size: int
    maximum_pairs: int = 64
    maximum_complete_scores: int = 5120

    def validate(self) -> None:
        if self.strategy not in ("residual", "structural", "slice_projection", "hybrid"):
            raise ValueError("unknown completion strategy")
        if not 1 <= self.w_per_pair <= 80 or self.final_pool_size <= 0 or self.maximum_pairs <= 0:
            raise ValueError("completion budgets must be positive and w_per_pair at most 80")
        if self.maximum_complete_scores < self.maximum_pairs * 80:
            raise ValueError("score budget must cover the matched 80-vector audit for every pair")


def complete_frozen_pairs(residual: Tensor, pairs: Sequence[dict[str, Any]], config: CompletionConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Complete a frozen ordered pair set and retain a pair-balanced action pool."""
    config.validate()
    started = time.perf_counter_ns()
    residual_norm = squared_norm(residual)
    instances = []
    completion_ms = exact_scoring_ms = existing_top_k_ms = 0.0
    for pair in pairs[:config.maximum_pairs]:
        phase = time.perf_counter_ns()
        selected, audit = complete_w_vectors(pair["contraction"], pair["q"], config.w_per_pair, config.strategy)
        completion_ms += (time.perf_counter_ns() - phase) / 1e6
        exact_scoring_ms += audit["matched_exhaustive_scoring_ms"]
        existing_top_k_ms += audit["existing_top_k_ms"]
        for item in selected:
            action = canonicalize_action(RankOneAction(pair["u"], pair["v"], item["w"]))
            validate_action(action, (4, 4, 4), frozenset(COEFFICIENTS))
            instances.append({"action": action, "action_id": action_identity(action),
                              "pair_id": pair["pair_id"], "pair_order": pair["original_candidate_order"],
                              "w": list(item["w"]), "w_selection_rank": item["selection_rank"],
                              "residual_w_rank": 1 + sum((row["score_delta"], row["w"]) < (item["score_delta"], item["w"])
                                                          for row in _scored_w_rows(pair["contraction"], pair["q"])),
                              "score": residual_norm + item["score_delta"], "source": item["source"],
                              "w_support": item["support"], "w_sign_pattern": list(item["sign_pattern"]),
                              "projection_strength": [item["projection_strength"].numerator, item["projection_strength"].denominator],
                              "preparation_additions": action_cost(action).preparation_additions})
    dedup_started = time.perf_counter_ns()
    unique = {}
    for row in instances:
        previous = unique.get(row["action_id"])
        key = (row["score"], row["action_id"], row["pair_order"], row["w_selection_rank"])
        if previous is None or key < previous[0]:
            unique[row["action_id"]] = (key, row)
    dedup_ms = (time.perf_counter_ns() - dedup_started) / 1e6
    ranking_started = time.perf_counter_ns()
    ordered = [item[1] for item in sorted(unique.values(), key=lambda item: item[0])]
    retained = ordered[:config.final_pool_size]
    for rank, row in enumerate(retained, 1):
        row["final_rank"] = rank
        row["action"] = {"u": list(row["action"].u), "v": list(row["action"].v), "w": list(row["action"].w)}
    ranking_ms = (time.perf_counter_ns() - ranking_started) / 1e6
    return retained, {
        "config": config.__dict__, "pairs_consumed": min(len(pairs), config.maximum_pairs),
        "valid_w_vectors_scored": min(len(pairs), config.maximum_pairs) * 80,
        "selected_pair_w_instances": len(instances), "duplicate_instances": len(instances) - len(unique),
        "unique_complete_actions": len(unique), "retained_actions": len(retained),
        "completion_ms": completion_ms, "matched_exact_scoring_ms": exact_scoring_ms,
        "existing_top_k_ms": existing_top_k_ms, "deduplication_ms": dedup_ms, "ranking_ms": ranking_ms,
        "total_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def completion_recall(rows: Sequence[dict[str, Any]], known: Sequence[RankOneAction], frozen_pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure pair/action/tensor coverage without influencing construction."""
    actions = tuple(RankOneAction(row["action"]["u"], row["action"]["v"], row["action"]["w"]) for row in rows)
    ids = {action_identity(action) for action in actions}
    tensors = {outer_product(action) for action in actions}
    pair_keys = {_pair_key(row["u"], row["v"]) for row in frozen_pairs}
    records = []
    for index, raw in enumerate(known, 1):
        canonical = canonicalize_action(raw)
        pair_present = _pair_key(canonical.u, canonical.v) in pair_keys
        matches = [row for row in rows if row["action_id"] == action_identity(canonical)]
        possible_w_ranks = []
        for pair in frozen_pairs:
            if _pair_key(pair["u"], pair["v"]) != _pair_key(canonical.u, canonical.v):
                continue
            scored = _scored_w_rows(pair["contraction"], pair["q"])
            hit = next((item for item in scored if outer_product(RankOneAction(pair["u"], pair["v"], item["w"])) == outer_product(canonical)), None)
            if hit is not None:
                possible_w_ranks.append(1 + sum((item["score_delta"], item["w"]) < (hit["score_delta"], hit["w"]) for item in scored))
        records.append({"known_index": index, "pair_present": pair_present,
                        "complete_action_present": action_identity(canonical) in ids,
                        "tensor_equivalent": outer_product(canonical) in tensors,
                        "residual_w_rank": min(possible_w_ranks, default=None),
                        "final_action_rank": min((row["final_rank"] for row in matches), default=None),
                        "earliest_exclusion_stage": "retained" if matches else ("w_selection_or_final_pool" if pair_present else "frozen_pair_set")})
    return {"pair_recall": sum(row["pair_present"] for row in records),
            "complete_action_recall": sum(row["complete_action_present"] for row in records),
            "tensor_equivalence_recall": sum(row["tensor_equivalent"] for row in records), "records": records}
