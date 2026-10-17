#!/usr/bin/env python3
"""Run quota-preserving multi-w recall and CPU search diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from matmul_inspector.tensor_game import matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_bounded_residual import (
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
)
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_quota_w import QuotaMultiWProvider, quota_multi_w_candidates
from matmul_inspector.tensor_game_proposal_coverage import tensor_identity


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def strassen_states():
    actions = strassen_2x2_decomposition()
    specifications = (("forward", tuple(range(7)), None), ("reverse", tuple(reversed(range(7))), {1, 3, 6}))
    rows = []
    for name, order, selected_depths in specifications:
        residual = matrix_multiplication_tensor(2, 2, 2)
        for depth, term_index in enumerate(order):
            action = actions[term_index]
            if selected_depths is None or depth in selected_depths:
                rows.append({
                    "label": f"strassen/{name}/depth-{depth}",
                    "residual": residual,
                    "expected": action,
                })
            residual = subtract_action(residual, action)
    return rows


class BaselineAuditProvider:
    name = "bounded_128"

    def __init__(self):
        self.calls = 0
        self.complete_actions_scored = 0
        self.returned_actions = 0
        self.generation_seconds = 0.0

    def __call__(self, residual, limit):
        started = time.perf_counter()
        rows, audit = bounded_residual_candidates_memory_efficient(residual, 128, limit)
        self.calls += 1
        self.complete_actions_scored += audit["fully_evaluated_pairs"]
        self.returned_actions += len(rows)
        self.generation_seconds += time.perf_counter() - started
        return tuple(action_from_payload(row["action"]) for row in rows)

    def diagnostic_summary(self):
        return {
            "provider_calls": self.calls,
            "complete_actions_scored": self.complete_actions_scored,
            "returned_actions": self.returned_actions,
            "mean_branching_factor": self.returned_actions / self.calls if self.calls else 0.0,
            "generation_seconds": self.generation_seconds,
        }


def recall_records():
    configurations = [
        ("baseline_12_0", "residual", 12, 0),
        ("quota_residual_11_1", "residual", 11, 1),
        ("quota_residual_10_2", "residual", 10, 2),
        ("quota_residual_8_4", "residual", 8, 4),
        ("quota_diversity_11_1", "diversity", 11, 1),
        ("quota_diversity_10_2", "diversity", 10, 2),
        ("quota_diversity_8_4", "diversity", 8, 4),
    ]
    records = []
    for name, strategy, protected, slots in configurations:
        for state in strassen_states():
            rows, audit = quota_multi_w_candidates(
                state["residual"],
                protected_baseline=protected,
                alternative_slots=slots,
                pair_width=128,
                w_candidates_per_pair=2,
                strategy=strategy,
                total_returned=12,
            )
            expected_id = action_identity(state["expected"])
            returned_ids = [row["action_id"] for row in rows]
            generated_ids = set(returned_ids) | set(audit["generated_alternative_action_ids"])
            records.append({
                "configuration": name,
                "strategy": strategy,
                "protected_baseline": protected,
                "alternative_slots": slots,
                "state_label": state["label"],
                "state_identity": tensor_identity(state["residual"]),
                "expected_action_id": expected_id,
                "expected_action_reachable_top_k": expected_id in generated_ids,
                "expected_action_returned": expected_id in returned_ids,
                "expected_return_rank": returned_ids.index(expected_id) + 1 if expected_id in returned_ids else None,
                "excluded_by_reserved_slots": expected_id in generated_ids and expected_id not in returned_ids,
                "audit": audit,
                "returned_candidates": [{
                    "action_id": row["action_id"],
                    "source": row["source"],
                    "residual_reduction": row["residual_reduction"],
                    "preparation_additions": row["preparation_additions"],
                    "w_support": row["w_support"],
                    "total_support": sum(
                        value != 0 for key in ("u", "v", "w") for value in row["action"][key]
                    ),
                } for row in rows],
            })
    return records


def make_provider(name, trace_limit):
    if name == "bounded_128":
        return BaselineAuditProvider()
    strategy = name.removeprefix("quota_")
    return QuotaMultiWProvider(
        protected_baseline=11,
        alternative_slots=1,
        pair_width=128,
        w_candidates_per_pair=2,
        strategy=strategy,
        total_returned=12,
        trace_limit=trace_limit,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--ranks", type=int, nargs="+", default=(7, 8))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--compute-matched-node-budget", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--trace-limit", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)

    recall = recall_records()
    trials = []
    experiment_groups = [
        ("node_matched", "bounded_128", args.node_budget),
        ("node_matched", "quota_residual", args.node_budget),
        ("node_matched", "quota_diversity", args.node_budget),
        ("evaluation_approximately_matched", "quota_residual", args.compute_matched_node_budget),
        ("evaluation_approximately_matched", "quota_diversity", args.compute_matched_node_budget),
    ]
    for budget_mode, provider_name, nodes in experiment_groups:
        for rank in args.ranks:
            for seed in args.seeds:
                provider = make_provider(provider_name, args.trace_limit)
                result = discover(
                    DiscoveryConfig(rank, seed, nodes, args.beam_width, args.max_actions, "baseline", 1),
                    provider,
                )
                result.update({
                    "budget_mode": budget_mode,
                    "provider": provider_name,
                    "provider_configuration": (
                        {"pair_width": 128, "deterministic_w": True}
                        if provider_name == "bounded_128"
                        else {
                            "pair_width": 128,
                            "w_candidates_per_pair": 2,
                            "protected_baseline": 11,
                            "alternative_slots": 1,
                            "strategy": provider_name.removeprefix("quota_"),
                        }
                    ),
                    "provider_diagnostics": provider.diagnostic_summary(),
                    "provider_trace": getattr(provider, "trace", []),
                })
                trials.append(result)
                print(json.dumps({
                    "budget_mode": budget_mode,
                    "provider": provider_name,
                    "rank": rank,
                    "seed": seed,
                    "success": result["success"],
                    "nodes": result["nodes_expanded"],
                    "seconds": result["wall_seconds"],
                }), flush=True)

    recall_summary = {}
    candidate_costs = {}
    for name in sorted({row["configuration"] for row in recall}):
        group = [row for row in recall if row["configuration"] == name]
        recall_summary[name] = {
            "states": len(group),
            "exact_action_reachable_top_k": sum(row["expected_action_reachable_top_k"] for row in group),
            "exact_action_returned": sum(row["expected_action_returned"] for row in group),
            "excluded_by_reserved_slots": sum(row["excluded_by_reserved_slots"] for row in group),
            "mean_complete_vectors_scored": sum(row["audit"]["top_k_complete_vectors_scored"] for row in group) / len(group),
            "mean_provider_ms": sum(row["audit"]["total_ms"] for row in group) / len(group),
        }
        costs = [cost for row in group for cost in row["returned_candidates"]]
        candidate_costs[name] = {
            "candidate_instances": len(costs),
            "preparation_additions": dict(sorted(Counter(row["preparation_additions"] for row in costs).items())),
            "residual_reductions": dict(sorted(Counter(row["residual_reduction"] for row in costs).items())),
            "w_support": dict(sorted(Counter(row["w_support"] for row in costs).items())),
            "total_support": dict(sorted(Counter(row["total_support"] for row in costs).items())),
        }
    search_summary = {}
    for budget_mode, provider_name, _ in experiment_groups:
        for rank in args.ranks:
            group = [row for row in trials if row["budget_mode"] == budget_mode and row["provider"] == provider_name and row["config"]["rank"] == rank]
            search_summary[f"{budget_mode}:{provider_name}:rank{rank}"] = {
                "trials": len(group),
                "successes": sum(row["success"] for row in group),
                "nodes_expanded": sum(row["nodes_expanded"] for row in group),
                "provider_calls": sum(row["provider_diagnostics"]["provider_calls"] for row in group),
                "complete_actions_scored": sum(row["provider_diagnostics"]["complete_actions_scored"] for row in group),
                "provider_seconds": sum(row["provider_diagnostics"]["generation_seconds"] for row in group),
                "wall_seconds": sum(row["wall_seconds"] for row in group),
                "candidate_edges_generated": sum(row["candidate_edges_generated"] for row in group),
                "duplicate_state_paths": sum(row["duplicate_state_paths"] for row in group),
                "maximum_depth_reached": max(row["depth_reached"] for row in group),
                "termination_reasons": dict(Counter(row["termination_reason"] for row in group)),
            }

    previous_summary_path = Path("artifacts/tensor-game/multi-w-provider-v1/summary.json")
    previous_summary_bytes = previous_summary_path.read_bytes()
    comparison = {
        "source": str(previous_summary_path),
        "sha256": hashlib.sha256(previous_summary_bytes).hexdigest(),
        "node_matched_search": json.loads(previous_summary_bytes)["search"],
    }
    payloads = {
        "recall.json": encoded({"schema": "tensor_game_quota_recall.v1", "records": recall}),
        "trials.json": encoded({
            "schema": "tensor_game_quota_search.v1",
            "configuration": vars(args) | {"output": str(args.output)},
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "trials": trials,
        }),
        "summary.json": encoded({
            "schema": "tensor_game_quota_summary.v1",
            "recall": recall_summary,
            "search": search_summary,
            "candidate_cost_distributions": candidate_costs,
            "previous_exhaustive_multi_w_comparison": comparison,
        }),
        "configuration.json": encoded({"command": " ".join(sys.argv), "seeds": args.seeds, "ranks": args.ranks}),
    }
    for name, payload in payloads.items():
        (args.output / name).write_bytes(payload)
    hashes = {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in payloads.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded(hashes))
    print(json.dumps(json.loads(payloads["summary.json"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
