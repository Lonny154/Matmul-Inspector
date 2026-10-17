#!/usr/bin/env python3
"""Run interleaved feasibility-gated algebraic exploration experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from matmul_inspector.tensor_game import standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_quota_w import QuotaMultiWProvider
from run_tensor_game_quota_provider import BaselineAuditProvider


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def experiment_specs():
    return (
        {
            "name": "bounded_128_global",
            "provider": "bounded_128",
            "admission": "global",
            "baseline_slots": 12,
            "alternative_slots": 0,
        },
        {
            "name": "quota_residual_11_1_global",
            "provider": "quota_residual_11_1",
            "admission": "global",
            "baseline_slots": 11,
            "alternative_slots": 1,
        },
        {
            "name": "quota_residual_11_1_provenance",
            "provider": "quota_residual_11_1",
            "admission": "provenance_quota",
            "baseline_slots": 11,
            "alternative_slots": 1,
        },
    )


def make_provider(spec):
    if spec["provider"] == "bounded_128":
        return BaselineAuditProvider()
    return QuotaMultiWProvider(
        protected_baseline=11,
        alternative_slots=1,
        pair_width=128,
        w_candidates_per_pair=2,
        strategy="residual",
        total_returned=12,
        trace_limit=0,
    )


def run_trial(spec, gate, rank, seed, node_budget, budget_mode, execution_index):
    provider = make_provider(spec)
    config = DiscoveryConfig(
        rank=rank,
        seed=seed,
        node_budget=node_budget,
        beam_width=12,
        max_actions=12,
        admission_policy=spec["admission"],
        baseline_capacity=(11 if spec["admission"] == "provenance_quota" else None),
        alternative_capacity=(1 if spec["admission"] == "provenance_quota" else 0),
        feasibility_gate=gate,
        measure_phase_times=True,
    )
    result = discover(
        config,
        provider,
        diagnostic_trajectories={
            "conventional": standard_2x2_decomposition(),
            "strassen": strassen_2x2_decomposition(),
        },
    )
    return {
        "budget_mode": budget_mode,
        "experiment": spec["name"],
        "provider_configuration": spec,
        "gate": gate,
        "rank": rank,
        "seed": seed,
        "execution_index": execution_index,
        "result": result,
        "provider_diagnostics": provider.diagnostic_summary(),
    }


def schedule_trials(specs, seeds, node_budget, compute_node_budget):
    schedule = []
    pair_index = 0
    for rank in (7, 8):
        for seed_index, seed in enumerate(seeds):
            for spec_index, spec in enumerate(specs):
                gates = ("off", "flattening_rank")
                if (seed_index + rank + spec_index) % 2:
                    gates = tuple(reversed(gates))
                for gate in gates:
                    schedule.append(("node_matched", spec, gate, rank, seed, node_budget, pair_index))
                pair_index += 1
    # This is a bounded evaluation-cost approximation, not an exact CPU match.
    for rank in (7, 8):
        for spec in specs[1:]:
            for gate in ("off", "flattening_rank"):
                schedule.append((
                    "evaluation_cap_approximation",
                    spec,
                    gate,
                    rank,
                    seeds[0],
                    compute_node_budget,
                    pair_index,
                ))
            pair_index += 1
    return schedule


def group_summary(trials):
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial["budget_mode"], trial["experiment"], trial["gate"], trial["rank"])].append(trial)
    summary = {}
    for key, group in sorted(grouped.items()):
        depths = [depth for trial in group for depth in trial["result"]["depth_diagnostics"]]
        generated = Counter()
        admitted = Counter()
        pruned = Counter()
        for depth in depths:
            generated.update(depth["generated"])
            admitted.update(depth["admitted"])
            pruned.update(depth["feasibility"]["pruned_by_provenance"])
        alternative_depths = [
            depth["depth"] for depth in depths if depth["admitted"].get("alternative", 0)
        ]
        trajectory_losses = Counter()
        for trial in group:
            for trajectory in ("conventional", "strassen"):
                rows = [row for row in trial["result"]["known_trajectory_diagnostics"]
                        if row["trajectory"] == trajectory and row["loss_stage"]]
                label = "none" if not rows else f"depth_{rows[0]['depth']}:{rows[0]['loss_stage']}"
                trajectory_losses[f"{trajectory}:{label}"] += 1
        name = f"{key[0]}:{key[1]}:{key[2]}:rank{key[3]}"
        summary[name] = {
            "trials": len(group),
            "successes": sum(trial["result"]["success"] for trial in group),
            "maximum_depth": max(trial["result"]["depth_reached"] for trial in group),
            "nodes_expanded": sum(trial["result"]["nodes_expanded"] for trial in group),
            "provider_calls": sum(trial["result"]["provider_calls"] for trial in group),
            "candidate_edges_generated": sum(
                trial["result"]["candidate_edges_generated"] for trial in group
            ),
            "duplicate_state_paths": sum(
                trial["result"]["duplicate_state_paths"] for trial in group
            ),
            "duplicate_edge_fraction": (
                sum(trial["result"]["duplicate_state_paths"] for trial in group)
                / max(1, sum(trial["result"]["candidate_edges_generated"] for trial in group))
            ),
            "provider_generation_seconds": sum(
                trial["provider_diagnostics"]["generation_seconds"] for trial in group
            ),
            "complete_actions_scored": sum(
                trial["provider_diagnostics"]["complete_actions_scored"] for trial in group
            ),
            "feasibility_checked": sum(
                trial["result"]["feasibility_gate_diagnostics"]["checked_children"]
                for trial in group
            ),
            "feasibility_pruned": sum(
                trial["result"]["feasibility_gate_diagnostics"]["pruned_children"]
                for trial in group
            ),
            "feasibility_seconds": sum(
                trial["result"]["feasibility_gate_diagnostics"]["check_seconds"]
                for trial in group
            ),
            "beam_admission_seconds": sum(depth["beam_admission_seconds"] for depth in depths),
            "infeasible_that_would_enter_beam": sum(
                depth["feasibility"]["infeasible_that_would_enter_beam"] for depth in depths
            ),
            "generated_by_provenance": dict(sorted(generated.items())),
            "admitted_by_provenance": dict(sorted(admitted.items())),
            "pruned_by_provenance": dict(sorted(pruned.items())),
            "maximum_alternative_edge_depth": max(alternative_depths, default=None),
            "wall_seconds": sum(trial["result"]["wall_seconds"] for trial in group),
            "termination_reasons": dict(Counter(
                trial["result"]["termination_reason"] for trial in group
            )),
            "first_known_trajectory_losses": dict(sorted(trajectory_losses.items())),
        }
    return summary


def paired_summary(trials):
    by_key = {
        (trial["budget_mode"], trial["experiment"], trial["rank"], trial["seed"], trial["gate"]): trial
        for trial in trials
    }
    pairs = []
    base_keys = sorted({key[:-1] for key in by_key})
    for key in base_keys:
        off = by_key[key + ("off",)]
        on = by_key[key + ("flattening_rank",)]
        off_result = off["result"]
        on_result = on["result"]
        pairs.append({
            "budget_mode": key[0],
            "experiment": key[1],
            "rank": key[2],
            "seed": key[3],
            "execution_order": [
                trial["gate"] for trial in sorted((off, on), key=lambda trial: trial["execution_index"])
            ],
            "off_success": off_result["success"],
            "on_success": on_result["success"],
            "wall_seconds_delta_on_minus_off": on_result["wall_seconds"] - off_result["wall_seconds"],
            "provider_generation_seconds_delta_on_minus_off": (
                on["provider_diagnostics"]["generation_seconds"]
                - off["provider_diagnostics"]["generation_seconds"]
            ),
            "nodes_avoided": off_result["nodes_expanded"] - on_result["nodes_expanded"],
            "provider_calls_avoided": off_result["provider_calls"] - on_result["provider_calls"],
            "candidate_edges_avoided": (
                off_result["candidate_edges_generated"] - on_result["candidate_edges_generated"]
            ),
            "children_pruned": on_result["feasibility_gate_diagnostics"]["pruned_children"],
            "pruned_that_would_enter_beam": sum(
                depth["feasibility"]["infeasible_that_would_enter_beam"]
                for depth in on_result["depth_diagnostics"]
            ),
        })
    return pairs


def paired_runtime_summary(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        if row["budget_mode"] == "node_matched":
            grouped[(row["experiment"], row["rank"])].append(row)
    return {
        f"{experiment}:rank{rank}": {
            "pairs": len(rows),
            "mean_wall_seconds_delta_on_minus_off": statistics.mean(
                row["wall_seconds_delta_on_minus_off"] for row in rows
            ),
            "median_wall_seconds_delta_on_minus_off": statistics.median(
                row["wall_seconds_delta_on_minus_off"] for row in rows
            ),
            "mean_provider_seconds_delta_on_minus_off": statistics.mean(
                row["provider_generation_seconds_delta_on_minus_off"] for row in rows
            ),
            "total_nodes_avoided": sum(row["nodes_avoided"] for row in rows),
            "total_provider_calls_avoided": sum(row["provider_calls_avoided"] for row in rows),
            "total_candidate_edges_avoided": sum(row["candidate_edges_avoided"] for row in rows),
        }
        for (experiment, rank), rows in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--compute-node-budget", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)

    specs = experiment_specs()
    schedule = schedule_trials(specs, args.seeds, args.node_budget, args.compute_node_budget)
    trials = []
    for execution_index, (budget_mode, spec, gate, rank, seed, node_budget, pair_index) in enumerate(schedule):
        trial = run_trial(spec, gate, rank, seed, node_budget, budget_mode, execution_index)
        trial["pair_index"] = pair_index
        trials.append(trial)
        result = trial["result"]
        print(json.dumps({
            "budget": budget_mode,
            "experiment": spec["name"],
            "gate": gate,
            "rank": rank,
            "seed": seed,
            "success": result["success"],
            "nodes": result["nodes_expanded"],
            "seconds": result["wall_seconds"],
        }), flush=True)

    pairs = paired_summary(trials)
    summary = {
        "schema": "tensor_game_feasible_exploration_summary.v1",
        "groups": group_summary(trials),
        "paired_trials": pairs,
        "paired_runtime": paired_runtime_summary(pairs),
        "compute_match_semantics": (
            "The 64-node secondary quota runs approximate a complete-action evaluation cap because "
            "the provider has fixed per-call budgets. Gate pruning can reduce calls, so this is not "
            "exact equal compute."
        ),
        "provenance_semantics": (
            "Provenance labels incoming edges, not permanent state ancestry; maximum alternative depth "
            "means the deepest admitted alternative-origin edge."
        ),
    }
    configuration = {
        "schema": "tensor_game_feasible_exploration_configuration.v1",
        "command": " ".join(sys.argv),
        "configuration": vars(args) | {
            "output": str(args.output),
            "ranks": [7, 8],
            "beam_width": 12,
            "max_actions": 12,
            "specifications": specs,
            "schedule": [{
                "budget_mode": row[0], "experiment": row[1]["name"], "gate": row[2],
                "rank": row[3], "seed": row[4], "node_budget": row[5], "pair_index": row[6],
            } for row in schedule],
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    payloads = {
        "configuration.json": encoded(configuration),
        "trials.json": encoded({"schema": "tensor_game_feasible_exploration_trials.v1", "trials": trials}),
        "summary.json": encoded(summary),
    }
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    hashes = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_feasible_exploration_hashes.v1", "files": hashes
    }))
    print(json.dumps({
        "groups": summary["groups"], "paired_runtime": summary["paired_runtime"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
