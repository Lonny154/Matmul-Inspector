"""Bounded exact completion and one algebraic recombination diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
import random
import time
from typing import Sequence

from .tensor_game import RankOneAction, Tensor, outer_product, sum_actions, tensor_shape
from .tensor_game_experience import action_identity
from .tensor_game_meet_in_middle import add_tensors, partial_sum_identity, subtract_tensors
from .tensor_game_tempered_search import (
    PairRepairTable, build_pair_repair_table, canonical_state, pool_identity,
    residual_metrics, state_identity,
)


@dataclass(frozen=True)
class CompletionConfig:
    depths: tuple[int, ...] = (2, 3, 4)
    maximum_pool_size: int = 128
    maximum_pair_records: int = 8128
    maximum_unique_pair_sums: int = 8128
    maximum_complement_probes: int = 10_000
    maximum_verified_per_core: int = 1
    maximum_seconds: float = 10.0
    maximum_details: int = 4
    allow_repeated_actions: bool = False
    energy_mode: str = "nnz_l1"
    mismatch_weight: int = 4

    def validate(self) -> None:
        if not self.depths or any(depth not in (2, 3, 4) for depth in self.depths):
            raise ValueError("depths must be a nonempty subset of 2, 3, 4")
        if len(set(self.depths)) != len(self.depths):
            raise ValueError("completion depths must be unique")
        if min(self.maximum_pool_size, self.maximum_pair_records,
               self.maximum_unique_pair_sums, self.maximum_complement_probes,
               self.maximum_verified_per_core) <= 0 or self.maximum_details < 0:
            raise ValueError("completion limits must be positive and details nonnegative")
        if not math.isfinite(self.maximum_seconds) or self.maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be finite and positive")
        if self.energy_mode not in ("l1", "nnz_l1") or self.mismatch_weight < 0:
            raise ValueError("invalid energy settings")


@dataclass(frozen=True)
class CompletionLookup:
    pool_hash: str
    oracle_assisted: bool
    status: str
    pair_records: int
    unique_pair_sums: int
    build_seconds: float
    pair_sums_hashed: int
    pair_table: PairRepairTable


def build_completion_lookup(pool: Sequence[RankOneAction], config: CompletionConfig,
                            *, oracle_assisted: bool) -> CompletionLookup:
    """Reuse the existing exact pair table, with explicit size and hash limits."""
    config.validate()
    started = time.perf_counter()
    table = build_pair_repair_table(pool, allow_repeated_actions=config.allow_repeated_actions,
                                    maximum_pool_size=config.maximum_pool_size,
                                    maximum_pair_records=config.maximum_pair_records,
                                    maximum_seconds=config.maximum_seconds)
    status = "READY" if table.status == "READY" else "PAIR_TABLE_LIMIT"
    unique = len(table.table)
    if unique > config.maximum_unique_pair_sums:
        status = "PAIR_TABLE_LIMIT"
        table = PairRepairTable("BUDGET_EXCEEDED_UNIQUE_SUMS", table.records, {})
    return CompletionLookup(pool_identity(pool), oracle_assisted, status, table.records,
                            unique, time.perf_counter() - started, table.records, table)


def _check_lookup(pool: Sequence[RankOneAction], lookup: CompletionLookup,
                  oracle_assisted: bool) -> None:
    if pool_identity(pool) != lookup.pool_hash:
        raise ValueError("pair table pool hash does not match the active pool")
    if lookup.oracle_assisted and not oracle_assisted:
        raise ValueError("oracle pair table is forbidden in discovery mode")


def complete_core(target: Tensor, frozen: Sequence[int], missing: int,
                  pool: Sequence[RankOneAction], lookup: CompletionLookup,
                  config: CompletionConfig, *, oracle_assisted: bool,
                  maximum_probes: int | None = None, deadline: float | None = None) -> dict:
    """Find exact 2, 3, or 4 term complements for one frozen core."""
    _check_lookup(pool, lookup, oracle_assisted)
    if missing not in (2, 3, 4) or len(frozen) + missing != 7:
        raise ValueError("core and completion must total rank seven")
    if any(index < 0 or index >= len(pool) for index in frozen):
        raise ValueError("frozen core contains an invalid action index")
    if not config.allow_repeated_actions and len(set(frozen)) != len(frozen):
        raise ValueError("frozen core violates the repeated-action policy")
    if lookup.status != "READY":
        return {"status": "PAIR_TABLE_LIMIT", "probes": 0, "raw_matches": 0,
                "domain_valid_matches": 0, "verified_matches": 0,
                "exact_states": [], "first_exact": None, "work": {}}
    started = time.perf_counter()
    remaining = subtract_tensors(target, sum_actions(tuple(pool[i] for i in frozen), tensor_shape(target)))
    tensors = tuple(outer_product(action) for action in pool) if missing == 3 else ()
    pair_rows = tuple(row for bucket in lookup.pair_table.table.values() for row in bucket) if missing == 4 else ()
    probes = raw = valid = verified = subtractions = equalities = reconstructions = 0
    validation_seconds = 0.0
    unique: dict[str, tuple[int, ...]] = {}
    stopped = None
    limit = config.maximum_complement_probes if maximum_probes is None else maximum_probes
    if limit < 0:
        raise ValueError("maximum_probes must be nonnegative")
    if deadline is None:
        deadline = started + config.maximum_seconds
    stream = (((), remaining),) if missing == 2 else (
        ((i,), subtract_tensors(remaining, tensors[i])) for i in range(len(pool))
    ) if missing == 3 else (
        ((left, right), subtract_tensors(remaining, pair_sum)) for pair_sum, left, right in pair_rows
    )
    for prefix, required in stream:
        if probes >= limit:
            stopped = "LOOKUP_LIMIT"; break
        if time.perf_counter() >= deadline:
            stopped = "TIME_BUDGET"; break
        probes += 1
        subtractions += missing != 2
        bucket = lookup.pair_table.table.get(partial_sum_identity(required), ())
        for exact_sum, left, right in bucket:
            equalities += 1
            if exact_sum != required:
                continue
            raw += 1
            completed = canonical_state((*frozen, *prefix, left, right))
            if not config.allow_repeated_actions and len(set(completed)) != 7:
                continue
            valid += 1
            reconstructions += 1
            validation_started = time.perf_counter()
            reconstructed = sum_actions(tuple(pool[i] for i in completed), tensor_shape(target))
            validation_seconds += time.perf_counter() - validation_started
            if reconstructed != target:
                continue
            verified += 1
            unique.setdefault(state_identity(completed, pool), completed)
            if verified >= config.maximum_verified_per_core:
                stopped = "COMPLETE"; break
        if stopped:
            break
    status = stopped or ("COMPLETE" if unique else "NO_COMPLETION")
    elapsed = time.perf_counter() - started
    return {"status": status, "probes": probes, "raw_matches": raw,
            "domain_valid_matches": valid, "verified_matches": verified,
            "exact_states": list(unique.values()),
            "first_exact": next(iter(unique), None),
            "work": {"tensor_additions": 0, "tensor_subtractions": subtractions + 1,
                     "pair_sums_hashed": 0, "exact_tensor_equalities": equalities,
                     "candidate_reconstructions": reconstructions,
                     "probe_seconds": elapsed - validation_seconds,
                     "validation_seconds": validation_seconds,
                     "wall_clock_seconds": elapsed}}


def audit_completion_depth(target: Tensor, candidate: Sequence[int], pool: Sequence[RankOneAction],
                           lookup: CompletionLookup, config: CompletionConfig,
                           *, oracle_assisted: bool, source: dict | None = None) -> dict:
    """Audit all 21, 35, and 35 rank-seven frozen cores under strict budgets."""
    config.validate()
    _check_lookup(pool, lookup, oracle_assisted)
    if len(candidate) != 7 or any(index < 0 or index >= len(pool) for index in candidate):
        raise ValueError("candidate must be a complete rank-seven state in the pool")
    candidate = canonical_state(candidate)
    if not config.allow_repeated_actions and len(set(candidate)) != 7:
        raise ValueError("candidate violates the repeated-action policy")
    metrics = asdict(residual_metrics(target, tuple(pool[i] for i in candidate),
                                      config.energy_mode, config.mismatch_weight))
    started = time.perf_counter()
    depths = {}
    exact: dict[str, tuple[int, ...]] = {}
    for missing in config.depths:
        cores = list(combinations(range(7), 7 - missing))
        records = []
        depth_exact: dict[str, tuple[int, ...]] = {}
        probes_used = 0
        depth_deadline = time.perf_counter() + config.maximum_seconds
        for positions in cores:
            frozen = tuple(candidate[i] for i in positions)
            result = complete_core(target, frozen, missing, pool, lookup, config,
                oracle_assisted=oracle_assisted,
                maximum_probes=max(0, config.maximum_complement_probes - probes_used),
                deadline=depth_deadline)
            probes_used += result["probes"]
            for state in result["exact_states"]:
                exact.setdefault(state_identity(state, pool), state)
                depth_exact.setdefault(state_identity(state, pool), state)
            records.append({"frozen_positions": positions, "frozen_action_ids":
                            [action_identity(pool[i]) for i in frozen],
                            "status": result["status"], "probes": result["probes"],
                            "raw_matches": result["raw_matches"],
                            "domain_valid_matches": result["domain_valid_matches"],
                            "verified_matches": result["verified_matches"],
                            "work": result["work"]})
        limited = sum(row["status"] in ("PAIR_TABLE_LIMIT", "LOOKUP_LIMIT", "TIME_BUDGET") for row in records)
        depths[str(missing)] = {
            "missing_terms": missing, "frozen_terms": 7 - missing,
            "frozen_cores_tested": len(records),
            "fully_tested": sum(row["status"] == "NO_COMPLETION" for row in records),
            "budget_limited": limited,
            "cores_with_raw_matches": sum(row["raw_matches"] > 0 for row in records),
            "cores_with_domain_valid_matches": sum(row["domain_valid_matches"] > 0 for row in records),
            "cores_with_verified_matches": sum(row["verified_matches"] > 0 for row in records),
            "raw_matches": sum(row["raw_matches"] for row in records),
            "domain_valid_matches": sum(row["domain_valid_matches"] for row in records),
            "verified_matches": sum(row["verified_matches"] for row in records),
            "unique_exact_completions": len(depth_exact),
            "first_exact": next(iter(depth_exact), None), "lookup_count": probes_used,
            "pair_table_build_seconds": lookup.build_seconds,
            "wall_clock_seconds": sum(row["work"].get("wall_clock_seconds", 0) for row in records),
            "statuses": {status: sum(row["status"] == status for row in records)
                         for status in ("COMPLETE", "NO_COMPLETION", "PAIR_TABLE_LIMIT", "LOOKUP_LIMIT", "TIME_BUDGET")},
            "work": {key: sum(row["work"].get(key, 0) for row in records)
                     for key in ("tensor_additions", "tensor_subtractions", "pair_sums_hashed",
                                 "exact_tensor_equalities", "candidate_reconstructions",
                                 "probe_seconds", "validation_seconds")},
            "core_records": records,
        }
    return {"schema": "tensor_game_completion_depth_audit.v1",
            "candidate_hash": state_identity(candidate, pool), "state": candidate,
            "action_ids": [action_identity(pool[i]) for i in candidate],
            "metrics": metrics, "pool_hash": lookup.pool_hash,
            "oracle_assisted": oracle_assisted, "source": source or {},
            "depths": depths, "unique_exact_completions": len(exact),
            "exact_candidates": [{"candidate_hash": digest, "state": state,
                                  "action_ids": [action_identity(pool[i]) for i in state],
                                  "exact_reconstruction": True,
                                  "oracle_assisted": oracle_assisted}
                                 for digest, state in list(exact.items())[:config.maximum_details]],
            "wall_clock_seconds": time.perf_counter() - started,
            "lookup_status": lookup.status}


def recombine_parents(target: Tensor, parents: Sequence[dict], pool: Sequence[RankOneAction],
                      lookup: CompletionLookup, config: CompletionConfig, *, oracle_assisted: bool,
                      top_k: int = 8, maximum_parent_pairs: int = 4,
                      maximum_partial_children_per_pair: int = 4, seed: int = 17) -> list[dict]:
    """Rank legal 2+2 parent unions by exact residual, then complete three slots."""
    _check_lookup(pool, lookup, oracle_assisted)
    if min(top_k, maximum_parent_pairs, maximum_partial_children_per_pair) <= 0:
        raise ValueError("recombination limits must be positive")
    unique = {row["candidate_hash"]: row for row in sorted(parents,
              key=lambda row: (row["metrics"]["energy"], row["candidate_hash"]))}
    ranked = list(unique.values())[:top_k]
    rng = random.Random(seed)
    parent_pairs = list(combinations(ranked, 2))
    rng.shuffle(parent_pairs)
    results = []
    for first, second in parent_pairs[:maximum_parent_pairs]:
        pair_started = time.perf_counter()
        partials = {}
        for left in combinations(first["state"], 2):
            for right in combinations(second["state"], 2):
                child = canonical_state((*left, *right))
                if not config.allow_repeated_actions and len(set(child)) != 4:
                    continue
                if child in partials:
                    continue
                metrics = asdict(residual_metrics(target, tuple(pool[i] for i in child),
                                                  config.energy_mode, config.mismatch_weight))
                partials[child] = metrics
        children = sorted(partials, key=lambda state: (partials[state]["energy"],
                          partials[state]["nonzero_entries"], state))[:maximum_partial_children_per_pair]
        child_rows = []
        for child in children:
            completion = complete_core(target, child, 3, pool, lookup, config,
                                       oracle_assisted=oracle_assisted)
            child_rows.append({"selected_term_ids": [action_identity(pool[i]) for i in child],
                               "partial_state": child,
                               "partial_tensor_hash": partial_sum_identity(sum_actions(tuple(pool[i] for i in child),
                                                                                         tensor_shape(target))),
                               "pre_completion_metrics": partials[child], "completion_depth": 3,
                               "completion_status": completion["status"],
                               "completion_attempts": 1, "completion_probes": completion["probes"],
                               "verified_matches": completion["verified_matches"],
                               "exact_states": completion["exact_states"],
                               "first_exact": completion["first_exact"],
                               "work": completion["work"]})
        results.append({"parent_hashes": [first["candidate_hash"], second["candidate_hash"]],
                        "parent_energies": [first["metrics"]["energy"], second["metrics"]["energy"]],
                        "subsets_scored": len(partials), "partial_children": child_rows,
                        "oracle_assisted": oracle_assisted, "pool_hash": lookup.pool_hash,
                        "wall_clock_seconds": time.perf_counter() - pair_started})
    return results
