#!/usr/bin/env python3
"""Run the bounded multi-w recall and CPU search experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_bounded_residual import (
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
    squared_norm,
)
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_hardware_cost import action_cost
from matmul_inspector.tensor_game_multi_w import MultiWProvider, multi_w_candidates
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
                    "expected_child": subtract_action(residual, action),
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
            "retained_pair_w_instances": self.complete_actions_scored,
            "returned_actions": self.returned_actions,
            "mean_branching_factor": self.returned_actions / self.calls if self.calls else 0.0,
            "generation_seconds": self.generation_seconds,
        }


def recall_rows(states, provider_name, strategy=None, w_count=1, complete_budget=None):
    records = []
    for state in states:
        if provider_name == "bounded_128":
            rows_for_generation, audit = bounded_residual_candidates_memory_efficient(state["residual"], 128, 128)
            rows = rows_for_generation[:12]
        else:
            rows_for_generation, audit = multi_w_candidates(
                state["residual"],
                pair_width=128,
                w_candidates_per_pair=w_count,
                strategy=strategy,
                complete_action_budget=complete_budget,
                limit=complete_budget,
            )
            rows = rows_for_generation[:12]
        expected_id = action_identity(state["expected"])
        residual_norm = squared_norm(state["residual"])
        for row in rows_for_generation:
            row.setdefault("residual_reduction", residual_norm - row["score"])
        generated_ids = [row["action_id"] for row in rows_for_generation]
        returned_ids = [row["action_id"] for row in rows]
        equivalent_ids = []
        for row in rows:
            action = action_from_payload(row["action"])
            if subtract_action(state["residual"], action) == state["expected_child"]:
                equivalent_ids.append(row["action_id"])
        expected_row = next((row for row in rows_for_generation if row["action_id"] == expected_id), None)
        records.append({
            "provider": provider_name,
            "strategy": strategy,
            "w_candidates_per_pair": w_count,
            "complete_action_budget": complete_budget,
            "state_label": state["label"],
            "state_identity": tensor_identity(state["residual"]),
            "expected_action_id": expected_id,
            "expected_action_generated": expected_id in generated_ids,
            "expected_action_retained_within_budget": expected_id in generated_ids,
            "expected_action_returned_top12": expected_id in returned_ids,
            "equivalent_child_returned": bool(equivalent_ids),
            "equivalent_child_action_ids": equivalent_ids,
            "complete_action_rank": generated_ids.index(expected_id) + 1 if expected_id in generated_ids else None,
            "expected_exact_residual_reduction": (
                expected_row.get("residual_reduction") if expected_row else None
            ),
            "expected_source": expected_row.get("source") if expected_row else None,
            "provider_audit": audit,
            "returned_costs": [
                {
                    "action_id": row["action_id"],
                    "preparation_additions": action_cost(action_from_payload(row["action"])).preparation_additions,
                    "residual_reduction": row.get("residual_reduction"),
                    "w_support": sum(value != 0 for value in row["action"]["w"]),
                    "total_support": sum(value != 0 for key in ("u", "v", "w") for value in row["action"][key]),
                }
                for row in rows
            ],
        })
    return records


def provider_for(name, trace_limit):
    if name == "bounded_128":
        return BaselineAuditProvider()
    strategy = name.removeprefix("multi_w_")
    return MultiWProvider(
        pair_width=128,
        w_candidates_per_pair=4,
        strategy=strategy,
        complete_action_budget=512,
        candidate_pool=128,
        trace_limit=trace_limit,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--ranks", type=int, nargs="+", default=(7, 8))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--trace-limit", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)

    states = strassen_states()
    recall = recall_rows(states, "bounded_128")
    for strategy in ("residual", "diversity"):
        for count in (2, 4, 8):
            recall += recall_rows(states, f"multi_w_{strategy}", strategy, count, 128 * count)

    trials = []
    providers = ("bounded_128", "multi_w_residual", "multi_w_diversity")
    budget_modes = (("node_matched", args.node_budget), ("complete_action_matched", None))
    for budget_mode, fixed_nodes in budget_modes:
        for provider_name in providers:
            nodes = fixed_nodes
            if nodes is None:
                nodes = args.node_budget if provider_name == "bounded_128" else max(1, args.node_budget // 80)
            for rank in args.ranks:
                for seed in args.seeds:
                    provider = provider_for(provider_name, args.trace_limit)
                    config = DiscoveryConfig(rank, seed, nodes, args.beam_width, args.max_actions, "baseline", 1)
                    result = discover(config, provider)
                    result.update({
                        "provider": provider_name,
                        "provider_configuration": (
                            {"pair_width": 128, "w_candidates_per_pair": 1, "strategy": "deterministic"}
                            if provider_name == "bounded_128"
                            else {
                                "pair_width": 128,
                                "w_candidates_per_pair": 4,
                                "strategy": provider_name.removeprefix("multi_w_"),
                                "complete_action_budget": 512,
                            }
                        ),
                        "budget_mode": budget_mode,
                        "provider_diagnostics": provider.diagnostic_summary(),
                        "provider_trace": getattr(provider, "trace", []),
                    })
                    trials.append(result)
                    print(json.dumps({
                        "provider": provider_name,
                        "budget_mode": budget_mode,
                        "rank": rank,
                        "seed": seed,
                        "success": result["success"],
                        "nodes": result["nodes_expanded"],
                        "seconds": result["wall_seconds"],
                    }), flush=True)

    recall_summary = {}
    for key in sorted({(row["provider"], row["strategy"], row["w_candidates_per_pair"]) for row in recall}):
        group = [row for row in recall if (row["provider"], row["strategy"], row["w_candidates_per_pair"]) == key]
        recall_summary[":".join(map(str, key))] = {
            "states": len(group),
            "exact_action_generated": sum(row["expected_action_generated"] for row in group),
            "exact_action_returned_top12": sum(row["expected_action_returned_top12"] for row in group),
            "equivalent_child_returned": sum(row["equivalent_child_returned"] for row in group),
        }
    search_summary = {}
    for budget_mode, _ in budget_modes:
        for provider_name in providers:
            for rank in args.ranks:
                group = [row for row in trials if row["budget_mode"] == budget_mode and row["provider"] == provider_name and row["config"]["rank"] == rank]
                search_summary[f"{budget_mode}:{provider_name}:rank{rank}"] = {
                    "trials": len(group),
                    "successes": sum(row["success"] for row in group),
                    "nodes_expanded": sum(row["nodes_expanded"] for row in group),
                    "complete_actions_scored": sum(row["provider_diagnostics"]["complete_actions_scored"] for row in group),
                    "candidate_edges_generated": sum(row["candidate_edges_generated"] for row in group),
                    "duplicate_state_paths": sum(row["duplicate_state_paths"] for row in group),
                    "wall_seconds": sum(row["wall_seconds"] for row in group),
                    "termination_reasons": dict(Counter(row["termination_reason"] for row in group)),
                }
    candidate_costs = {}
    for provider_name in providers:
        rows = [cost for record in recall if record["provider"] == provider_name and record["w_candidates_per_pair"] in (1, 4) for cost in record["returned_costs"]]
        candidate_costs[provider_name] = {
            "candidate_instances": len(rows),
            "preparation_additions": dict(sorted(Counter(row["preparation_additions"] for row in rows).items())),
            "residual_reductions": dict(sorted(Counter(row["residual_reduction"] for row in rows).items())),
            "w_support": dict(sorted(Counter(row["w_support"] for row in rows).items())),
            "total_support": dict(sorted(Counter(row["total_support"] for row in rows).items())),
        }

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payloads = {
        "recall.json": encoded({"schema": "tensor_game_multi_w_recall.v1", "records": recall}),
        "trials.json": encoded({
            "schema": "tensor_game_multi_w_search.v1",
            "configuration": vars(args) | {"output": str(args.output)},
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "git_commit": git_commit,
            "trials": trials,
        }),
        "summary.json": encoded({
            "schema": "tensor_game_multi_w_summary.v1",
            "recall": recall_summary,
            "search": search_summary,
            "candidate_cost_distributions": candidate_costs,
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
