"""Matched, descriptive ablations for the bounded tempered search prototype."""

from __future__ import annotations

from dataclasses import replace
import statistics
from typing import Sequence

from .tensor_game import RankOneAction, Tensor
from .tensor_game_tempered_search import (
    TemperedSearchConfig, audit_two_term_repairability, build_pair_repair_table,
    initial_states_for_seed, parallel_tempering_search, state_identity,
)

VARIANTS = (
    ("independent_no_repair", False, False),
    ("tempered_no_repair", True, False),
    ("independent_with_repair", False, True),
    ("tempered_with_repair", True, True),
)
COMPARISONS = (
    ("tempered_no_repair", "independent_no_repair"),
    ("tempered_with_repair", "independent_with_repair"),
    ("independent_with_repair", "independent_no_repair"),
    ("tempered_with_repair", "tempered_no_repair"),
)


def run_matched_study(target: Tensor, pool: Sequence[RankOneAction], base: TemperedSearchConfig,
                      seeds: Sequence[int], *, audit_enabled: bool, oracle_assisted: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate each seed's initial states once and run all four variants."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    base.validate(len(pool))
    audit_table = (build_pair_repair_table(pool, allow_repeated_actions=base.allow_repeated_actions,
                   maximum_pool_size=base.repair_maximum_pool_size,
                   maximum_pair_records=base.repair_maximum_pair_records,
                   maximum_seconds=base.repair_maximum_seconds))
    initial_rows: list[dict] = []
    runs: list[dict] = []
    audits: list[dict] = []
    for seed in seeds:
        paired_config = replace(base, seed=seed)
        states = initial_states_for_seed(len(pool), paired_config)
        hashes = [state_identity(state, pool) for state in states]
        initial_rows.append({"seed": seed, "states": states, "state_hashes": hashes})
        for name, exchange, repair in VARIANTS:
            config = replace(paired_config, exchange_enabled=exchange, repair_enabled=repair,
                             stop_on_exact=False)
            result = parallel_tempering_search(target, pool, config, states, audit_table)
            row = {"seed": seed, "variant": name, **result}
            if audit_enabled:
                audit = audit_two_term_repairability(target, result["best_state"], pool,
                            audit_table, base.allow_repeated_actions, oracle_assisted=oracle_assisted)  # type: ignore[arg-type]
                row["repairability_audit"] = {key: value for key, value in audit.items() if key != "details"}
                audits.append({"seed": seed, "variant": name, **audit})
            runs.append(row)
    return initial_rows, runs, audits


def aggregate_matched_runs(runs: Sequence[dict]) -> dict:
    """Summarize matched seed differences without inferential claims."""
    if not runs:
        raise ValueError("runs must be nonempty")
    by_variant = {}
    seed_sets = []
    for name, _, _ in VARIANTS:
        rows = {row["seed"]: row for row in runs if row["variant"] == name}
        if len(rows) != len([row for row in runs if row["variant"] == name]):
            raise ValueError("duplicate variant/seed run")
        by_variant[name] = rows
        seed_sets.append(set(rows))
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("all variants must have matching seeds")
    for seed in seed_sets[0]:
        hashes = {tuple(by_variant[name][seed]["initial_state_hashes"]) for name in by_variant}
        if len(hashes) != 1:
            raise ValueError("paired variants have different initial states")
        proposals = {by_variant[name][seed]["counters"]["local_proposals"] for name in by_variant}
        if len(proposals) != 1:
            raise ValueError("paired variants have different local proposal counts")
    groups = {}
    for name, rows_by_seed in by_variant.items():
        rows = list(rows_by_seed.values())
        counters = [row["counters"] for row in rows]
        total = lambda field: sum(c[field] for c in counters)
        proposals = total("local_proposals")
        accepts = total("local_accepts")
        exchanges = total("exchange_attempts")
        repairs = total("repair_attempts")
        groups[name] = {
            "runs": len(rows), "exact_successes": sum(row["status"] == "EXACT" for row in rows),
            "exact_success_rate": sum(row["status"] == "EXACT" for row in rows) / len(rows),
            "median_best_nnz": statistics.median(row["best_metrics"]["nonzero_entries"] for row in rows),
            "mean_best_nnz": statistics.mean(row["best_metrics"]["nonzero_entries"] for row in rows),
            "median_best_l1": statistics.median(row["best_metrics"]["l1_norm"] for row in rows),
            "mean_best_l1": statistics.mean(row["best_metrics"]["l1_norm"] for row in rows),
            "median_best_energy": statistics.median(row["best_metrics"]["energy"] for row in rows),
            "mean_best_energy": statistics.mean(row["best_metrics"]["energy"] for row in rows),
            "median_time_to_best_seconds": statistics.median(row["time_to_best_seconds"] for row in rows),
            "median_wall_clock_seconds": statistics.median(row["wall_clock_seconds"] for row in rows),
            "local_acceptance_rate": accepts / proposals if proposals else None,
            "worsening_acceptance_rate": total("worsening_accepts") / proposals if proposals else None,
            "exchange_acceptance_rate": total("exchange_accepts") / exchanges if exchanges else None,
            "repair_success_rate": total("repair_successes") / repairs if repairs else None,
            "repairable_best_fraction": (sum(row["repairability_audit"]["repairable"] for row in rows) / len(rows)
                                         if all("repairability_audit" in row for row in rows) else None),
            "median_unique_states_visited": statistics.median(row["unique_canonical_states_visited"] for row in rows),
            "total_local_proposals": proposals, "total_local_residual_evaluations": total("local_residual_evaluations"),
        }
    paired = {}
    for candidate, baseline in COMPARISONS:
        detail = []
        for seed in sorted(seed_sets[0]):
            left, right = by_variant[candidate][seed], by_variant[baseline][seed]
            detail.append({"seed": seed, "initial_state_hashes": left["initial_state_hashes"],
                           "energy_difference": left["best_metrics"]["energy"] - right["best_metrics"]["energy"],
                           "nnz_difference": left["best_metrics"]["nonzero_entries"] - right["best_metrics"]["nonzero_entries"],
                           "l1_difference": left["best_metrics"]["l1_norm"] - right["best_metrics"]["l1_norm"],
                           "exact_success_difference": int(left["status"] == "EXACT") - int(right["status"] == "EXACT"),
                           "wall_seconds_difference": left["wall_clock_seconds"] - right["wall_clock_seconds"]})
        paired[f"{candidate}_vs_{baseline}"] = {
            "per_seed": detail,
            "better_energy": sum(row["energy_difference"] < 0 for row in detail),
            "equal_energy": sum(row["energy_difference"] == 0 for row in detail),
            "worse_energy": sum(row["energy_difference"] > 0 for row in detail),
            "median_energy_difference": statistics.median(row["energy_difference"] for row in detail),
        }
    return {"schema": "tensor_game_tempered_study_aggregate.v1", "seeds": sorted(seed_sets[0]),
            "variants": groups, "paired_comparisons": paired,
            "interpretation": "Descriptive paired pilot; no statistical significance claim."}
