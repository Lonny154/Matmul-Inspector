#!/usr/bin/env python3
"""Run the controlled CPU provenance-aware Tensor Game beam study."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from matmul_inspector.tensor_game import standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_quota_w import QuotaMultiWProvider
from run_tensor_game_quota_provider import BaselineAuditProvider


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def experiment_configurations():
    yield {"name": "baseline_global_12_0", "provider": "bounded_128", "strategy": None,
           "admission": "global", "baseline_slots": 12, "alternative_slots": 0}
    for strategy in ("residual", "diversity"):
        yield {"name": f"quota_{strategy}_global_11_1", "provider": "quota", "strategy": strategy,
               "admission": "global", "baseline_slots": 11, "alternative_slots": 1}
        for baseline, alternative in ((11, 1), (10, 2), (8, 4)):
            yield {"name": f"quota_{strategy}_provenance_{baseline}_{alternative}", "provider": "quota",
                   "strategy": strategy, "admission": "provenance_quota",
                   "baseline_slots": baseline, "alternative_slots": alternative}


def make_provider(spec, trace_limit):
    if spec["provider"] == "bounded_128":
        return BaselineAuditProvider()
    return QuotaMultiWProvider(
        protected_baseline=spec["baseline_slots"],
        alternative_slots=spec["alternative_slots"],
        strategy=spec["strategy"],
        trace_limit=trace_limit,
    )


def summarize(rows):
    result = {}
    for key in sorted({(row["budget_mode"], row["experiment"], row["config"]["rank"]) for row in rows}):
        group = [row for row in rows if (row["budget_mode"], row["experiment"], row["config"]["rank"]) == key]
        depths = [depth for row in group for depth in row["depth_diagnostics"]]
        first_losses = Counter()
        for row in group:
            for trajectory in ("conventional", "strassen"):
                losses = [item for item in row["known_trajectory_diagnostics"]
                          if item["trajectory"] == trajectory and item["loss_stage"]]
                label = "none" if not losses else f"depth_{losses[0]['depth']}:{losses[0]['loss_stage']}"
                first_losses[f"{trajectory}:{label}"] += 1
        result[f"{key[0]}:{key[1]}:rank{key[2]}"] = {
            "trials": len(group),
            "successes": sum(row["success"] for row in group),
            "nodes_expanded": sum(row["nodes_expanded"] for row in group),
            "provider_calls": sum(row["provider_diagnostics"]["provider_calls"] for row in group),
            "complete_actions_scored": sum(row["provider_diagnostics"]["complete_actions_scored"] for row in group),
            "wall_seconds": sum(row["wall_seconds"] for row in group),
            "maximum_depth_reached": max(row["depth_reached"] for row in group),
            "candidate_edges_generated": sum(row["candidate_edges_generated"] for row in group),
            "duplicate_state_paths": sum(row["duplicate_state_paths"] for row in group),
            "admitted_baseline": sum(depth["admitted"].get("baseline", 0) for depth in depths),
            "admitted_alternative": sum(depth["admitted"].get("alternative", 0) for depth in depths),
            "termination_reasons": dict(Counter(row["termination_reason"] for row in group)),
            "first_known_trajectory_losses": dict(sorted(first_losses.items())),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--ranks", type=int, nargs="+", default=(7, 8))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--compute-matched-node-budget", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--trace-limit", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    if args.beam_width != 12:
        parser.error("this frozen study requires --beam-width 12")
    args.output.mkdir(parents=True)

    specs = list(experiment_configurations())
    scheduled = [("node_matched", spec, args.node_budget) for spec in specs]
    scheduled += [("evaluation_approximately_matched", spec, args.compute_matched_node_budget)
                  for spec in specs if spec["alternative_slots"] == 1]
    trajectories = {"conventional": standard_2x2_decomposition(), "strassen": strassen_2x2_decomposition()}
    trials = []
    for budget_mode, spec, node_budget in scheduled:
        for rank in args.ranks:
            for seed in args.seeds:
                provider = make_provider(spec, args.trace_limit)
                config = DiscoveryConfig(
                    rank, seed, node_budget, args.beam_width, args.max_actions,
                    admission_policy=spec["admission"],
                    baseline_capacity=spec["baseline_slots"] if spec["admission"] == "provenance_quota" else None,
                    alternative_capacity=spec["alternative_slots"] if spec["admission"] == "provenance_quota" else 0,
                )
                result = discover(config, provider, diagnostic_trajectories=trajectories)
                result.update({
                    "budget_mode": budget_mode,
                    "experiment": spec["name"],
                    "provider_configuration": spec,
                    "provider_diagnostics": provider.diagnostic_summary(),
                    "provider_trace": getattr(provider, "trace", []),
                })
                trials.append(result)
                print(json.dumps({"budget": budget_mode, "experiment": spec["name"], "rank": rank,
                                  "seed": seed, "success": result["success"],
                                  "depth": result["depth_reached"], "seconds": result["wall_seconds"]}), flush=True)

    configuration = {
        "command": " ".join(sys.argv), "seeds": args.seeds, "ranks": args.ranks,
        "node_budget": args.node_budget, "compute_matched_node_budget": args.compute_matched_node_budget,
        "beam_width": args.beam_width, "max_actions": args.max_actions,
        "configurations": specs,
    }
    payloads = {
        "trials.json": encoded({
            "schema": "tensor_game_provenance_beam_trials.v1",
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "configuration": configuration, "trials": trials,
        }),
        "summary.json": encoded({"schema": "tensor_game_provenance_beam_summary.v1", "search": summarize(trials)}),
        "configuration.json": encoded(configuration),
    }
    for name, payload in payloads.items():
        (args.output / name).write_bytes(payload)
    hashes = {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
              for name, payload in payloads.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded(hashes))
    print(json.dumps(json.loads(payloads["summary.json"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
