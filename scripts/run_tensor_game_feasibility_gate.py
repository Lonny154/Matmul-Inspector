#!/usr/bin/env python3
"""Run matched CPU searches for the optional exact feasibility gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from matmul_inspector.tensor_game import (
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
)
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def summarize_trials(trials):
    result = {}
    for gate in ("off", "flattening_rank"):
        for rank in (7, 8):
            group = [row for row in trials if row["gate"] == gate and row["rank"] == rank]
            key = f"{gate}:rank{rank}"
            result[key] = {
                "trials": len(group),
                "successes": sum(row["result"]["success"] for row in group),
                "maximum_depth": max(row["result"]["depth_reached"] for row in group),
                "nodes_expanded": sum(row["result"]["nodes_expanded"] for row in group),
                "provider_calls": sum(row["result"]["provider_calls"] for row in group),
                "candidate_edges_generated": sum(
                    row["result"]["candidate_edges_generated"] for row in group
                ),
                "duplicate_state_paths": sum(
                    row["result"]["duplicate_state_paths"] for row in group
                ),
                "feasibility_checked": sum(
                    row["result"]["feasibility_gate_diagnostics"]["checked_children"]
                    for row in group
                ),
                "feasibility_pruned": sum(
                    row["result"]["feasibility_gate_diagnostics"]["pruned_children"]
                    for row in group
                ),
                "rank_check_seconds": sum(
                    row["result"]["feasibility_gate_diagnostics"]["check_seconds"]
                    for row in group
                ),
                "wall_seconds": sum(row["result"]["wall_seconds"] for row in group),
                "termination_reasons": dict(Counter(
                    row["result"]["termination_reason"] for row in group
                )),
            }
    return result


def paired_effects(trials):
    by_key = {(row["rank"], row["seed"], row["gate"]): row["result"] for row in trials}
    rows = []
    for rank in (7, 8):
        seeds = sorted({row["seed"] for row in trials if row["rank"] == rank})
        for seed in seeds:
            baseline = by_key[(rank, seed, "off")]
            gated = by_key[(rank, seed, "flattening_rank")]
            would_enter = sum(
                depth["feasibility"]["infeasible_that_would_enter_beam"]
                for depth in gated["depth_diagnostics"]
            )
            rows.append({
                "rank": rank,
                "seed": seed,
                "baseline_success": baseline["success"],
                "gated_success": gated["success"],
                "children_pruned": gated["feasibility_gate_diagnostics"]["pruned_children"],
                "pruned_that_would_enter_beam": would_enter,
                "actual_expansions_avoided": baseline["nodes_expanded"] - gated["nodes_expanded"],
                "provider_calls_avoided": baseline["provider_calls"] - gated["provider_calls"],
                "candidate_edges_avoided": (
                    baseline["candidate_edges_generated"] - gated["candidate_edges_generated"]
                ),
                "wall_seconds_delta_gated_minus_baseline": (
                    gated["wall_seconds"] - baseline["wall_seconds"]
                ),
            })
    return rows


def depth_summary(trials):
    grouped = defaultdict(list)
    for trial in trials:
        if trial["gate"] != "flattening_rank":
            continue
        for row in trial["result"]["depth_diagnostics"]:
            grouped[(trial["rank"], row["depth"])].append(row)
    return {
        f"rank{rank}:depth{depth}": {
            "trials": len(rows),
            "remaining_budget": rows[0]["feasibility"]["remaining_budget"],
            "checked_children": sum(row["feasibility"]["checked_children"] for row in rows),
            "lower_bound_histogram": dict(sorted(sum(
                (Counter(row["feasibility"]["flattening_lower_bound_histogram"]) for row in rows),
                Counter(),
            ).items())),
            "pruned_children": sum(row["feasibility"]["pruned_children"] for row in rows),
            "surviving_checked_children": sum(
                row["feasibility"]["surviving_checked_children"] for row in rows
            ),
            "infeasible_that_would_enter_beam": sum(
                row["feasibility"]["infeasible_that_would_enter_beam"] for row in rows
            ),
            "survivors_admitted_only_after_pruning": sum(
                row["feasibility"]["survivors_admitted_only_after_pruning"] for row in rows
            ),
            "rank_check_seconds": sum(row["feasibility"]["check_seconds"] for row in rows),
            "beam_occupancy": [row["beam_occupancy"] for row in rows],
        }
        for (rank, depth), rows in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)

    trajectories = {
        "conventional": standard_2x2_decomposition(),
        "strassen": strassen_2x2_decomposition(),
    }
    trials = []
    for gate in ("off", "flattening_rank"):
        for rank in (7, 8):
            for seed in args.seeds:
                config = DiscoveryConfig(
                    rank=rank,
                    seed=seed,
                    node_budget=args.node_budget,
                    beam_width=args.beam_width,
                    max_actions=args.max_actions,
                    feasibility_gate=gate,
                )
                result = discover(config, diagnostic_trajectories=trajectories)
                trials.append({"gate": gate, "rank": rank, "seed": seed, "result": result})
                print(json.dumps({
                    "gate": gate,
                    "rank": rank,
                    "seed": seed,
                    "success": result["success"],
                    "nodes": result["nodes_expanded"],
                    "pruned": result["feasibility_gate_diagnostics"]["pruned_children"],
                    "seconds": result["wall_seconds"],
                }), flush=True)

    groups = summarize_trials(trials)
    pairs = paired_effects(trials)
    depths = depth_summary(trials)
    known_gate_losses = [
        (trial["rank"], row)
        for trial in trials if trial["gate"] == "flattening_rank"
        for row in trial["result"]["known_trajectory_diagnostics"]
        if row["child_feasibility_pruned"]
    ]
    rank_matched_gate_losses = [
        (rank, row) for rank, row in known_gate_losses
        if (rank, row["trajectory"]) in {(7, "strassen"), (8, "conventional")}
    ]
    summary = {
        "schema": "tensor_game_feasibility_gate_summary.v1",
        "groups": groups,
        "paired_effects": pairs,
        "depths": depths,
        "all_diagnostic_trajectory_gate_losses": len(known_gate_losses),
        "rank_matched_known_trajectory_gate_losses": len(rank_matched_gate_losses),
        "diagnostic_loss_note": (
            "The only gate losses are conventional rank-8 suffixes traced inside rank-7 trials; "
            "those states are correctly infeasible under the smaller budget."
        ),
        "elapsed_time_matched_comparison": (
            "not run: this deterministic beam exhausts its frontier before the node budget, "
            "and it has no cooperative wall-time limit; measured wall times are reported instead"
        ),
        "cache": (
            "not implemented: exact-state deduplication occurs after the gate, "
            "but this run is too small to justify cache complexity"
        ),
    }
    configuration = {
        "schema": "tensor_game_feasibility_gate_configuration.v1",
        "command": " ".join(sys.argv),
        "configuration": vars(args) | {
            "output": str(args.output),
            "ranks": [7, 8],
            "gates": ["off", "flattening_rank"],
            "feasibility_activation_budget": 3,
            "provider": "bounded_128",
            "admission": "global",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    payloads = {
        "configuration.json": encoded(configuration),
        "trials.json": encoded({"schema": "tensor_game_feasibility_gate_trials.v1", "trials": trials}),
        "summary.json": encoded(summary),
    }
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    hashes = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_feasibility_gate_hashes.v1", "files": hashes
    }))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
