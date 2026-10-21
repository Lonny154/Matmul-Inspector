#!/usr/bin/env python3
"""Run structural-pair coverage and matched CPU search experiments."""

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
from matmul_inspector.tensor_game_bounded_residual import action_identity
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_structural_pairs import (
    StructuralPairProvider,
    pair_structural_features,
    select_structural_pairs,
    structural_pair_candidates,
    structural_signature,
)
from matmul_inspector.tensor_game_w_coverage import sign_equivalent_representations
from run_tensor_game_quota_provider import BaselineAuditProvider


CONFIGURATIONS = ((128, 0), (120, 8), (112, 16), (96, 32))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def strassen_states():
    actions = strassen_2x2_decomposition()
    specifications = (("forward", tuple(range(7)), None), ("reverse", tuple(reversed(range(7))), {1, 3, 6}))
    rows = []
    for name, order, depths in specifications:
        residual = matrix_multiplication_tensor(2, 2, 2)
        for depth, term in enumerate(order):
            if depths is None or depth in depths:
                rows.append((f"strassen/{name}/depth-{depth}", residual, actions[term]))
            residual = subtract_action(residual, actions[term])
    return rows


def payload(row):
    result = dict(row)
    if "action" in result:
        action = result["action"]
        result["action"] = {"u": list(action.u), "v": list(action.v), "w": list(action.w)}
    return result


def coverage_records():
    records = []
    for protected, structural in CONFIGURATIONS:
        for label, residual, expected in strassen_states():
            started = time.perf_counter()
            pairs, audit = select_structural_pairs(
                residual, protected_residual_pairs=protected, structural_pairs=structural
            )
            pair_seconds = time.perf_counter() - started
            pair_keys = {(pair.u, pair.v) for pair in pairs}
            expected_pairs = {(item.u, item.v) for item in sign_equivalent_representations(expected)}
            pair_recalled = bool(pair_keys & expected_pairs)
            generated, generation_audit = structural_pair_candidates(
                residual, protected_residual_pairs=protected, structural_pairs=structural,
                w_mode="top_k", w_candidates_per_pair=2, limit=1000,
            )
            expected_id = action_identity(expected)
            generated_ids = [row["action_id"] for row in generated]
            returned_ids = generated_ids[:12]
            features = [pair_structural_features(pair) for pair in pairs]
            returned = generated[:12]
            records.append({
                "configuration": f"{protected}_{structural}", "protected_pairs": protected,
                "structural_pairs": structural, "state_label": label,
                "expected_action_id": expected_id, "pair_recalled": pair_recalled,
                "complete_action_generated": expected_id in generated_ids,
                "complete_action_returned": expected_id in returned_ids,
                "complete_action_rank": generated_ids.index(expected_id) + 1 if expected_id in generated_ids else None,
                "selected_original_ranks": audit["selected_original_ranks"],
                "distinct_structural_signatures": len({repr(structural_signature(item)) for item in features}),
                "preparation_additions": dict(sorted(Counter(item["preparation_additions"] for item in features).items())),
                "contraction_support": dict(sorted(Counter(item["contraction_support"] for item in features).items())),
                "pair_selection_seconds": pair_seconds,
                "pair_audit": audit,
                "generation_audit": generation_audit,
                "returned_negative_reduction": sum(row["residual_reduction"] < 0 for row in returned),
                "returned_reduction_histogram": dict(sorted(Counter(row["residual_reduction"] for row in returned).items())),
                "returned_actions": [payload(row) for row in returned],
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--ranks", type=int, nargs="+", default=(7, 8))
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)

    coverage = coverage_records()
    baseline_pairs = {row["state_label"]: row["pair_recalled"] for row in coverage if row["configuration"] == "128_0"}
    coverage_summary = {}
    for protected, structural in CONFIGURATIONS:
        name = f"{protected}_{structural}"
        group = [row for row in coverage if row["configuration"] == name]
        initial = next(row for row in group if row["state_label"] == "strassen/forward/depth-0")
        coverage_summary[name] = {
            "states": len(group), "pair_recall": sum(row["pair_recalled"] for row in group),
            "new_pair_recall": sum(row["pair_recalled"] and not baseline_pairs[row["state_label"]] for row in group),
            "complete_action_generated": sum(row["complete_action_generated"] for row in group),
            "complete_action_returned": sum(row["complete_action_returned"] for row in group),
            "mean_distinct_signatures": sum(row["distinct_structural_signatures"] for row in group) / len(group),
            "pair_selection_seconds": sum(row["pair_selection_seconds"] for row in group),
            "initial_pair_recalled": initial["pair_recalled"],
            "initial_complete_action_returned": initial["complete_action_returned"],
            "initial_selected_original_rank_max": max(initial["selected_original_ranks"]),
            "returned_negative_reduction_actions": sum(row["returned_negative_reduction"] for row in group),
        }

    search_specs = (
        ("baseline", None),
        ("structural_96_32_deterministic", StructuralPairProvider),
        ("structural_96_32_top_k", StructuralPairProvider),
    )
    trials = []
    for name, factory in search_specs:
        for rank in args.ranks:
            for seed in args.seeds:
                if factory is None:
                    provider = BaselineAuditProvider()
                else:
                    provider = factory(96, 32, "top_k" if name.endswith("top_k") else "deterministic", 2, 1)
                result = discover(
                    DiscoveryConfig(rank, seed, args.node_budget, args.beam_width, args.max_actions), provider
                )
                result.update({"provider": name, "provider_diagnostics": provider.diagnostic_summary()})
                trials.append(result)
                print(json.dumps({"provider": name, "rank": rank, "seed": seed,
                                  "success": result["success"], "depth": result["depth_reached"],
                                  "seconds": result["wall_seconds"]}), flush=True)
    search_summary = {}
    for name, _ in search_specs:
        for rank in args.ranks:
            group = [row for row in trials if row["provider"] == name and row["config"]["rank"] == rank]
            search_summary[f"{name}:rank{rank}"] = {
                "trials": len(group), "successes": sum(row["success"] for row in group),
                "maximum_depth": max(row["depth_reached"] for row in group),
                "nodes_expanded": sum(row["nodes_expanded"] for row in group),
                "provider_calls": sum(row["provider_diagnostics"]["provider_calls"] for row in group),
                "complete_actions_scored": sum(row["provider_diagnostics"]["complete_actions_scored"] for row in group),
                "wall_seconds": sum(row["wall_seconds"] for row in group),
                "negative_returned_actions": sum(row["provider_diagnostics"].get("negative_reduction_actions", 0) for row in group),
                "termination_reasons": dict(Counter(row["termination_reason"] for row in group)),
            }
    configuration = vars(args) | {"output": str(args.output), "pair_configurations": CONFIGURATIONS}
    payloads = {
        "coverage.json": encoded({"schema": "tensor_game_structural_pairs_coverage.v1", "records": coverage}),
        "trials.json": encoded({"schema": "tensor_game_structural_pairs_trials.v1", "trials": trials}),
        "summary.json": encoded({"schema": "tensor_game_structural_pairs_summary.v1",
                                 "coverage": coverage_summary, "search": search_summary}),
        "configuration.json": encoded({
            "command": " ".join(sys.argv), "configuration": configuration,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        }),
    }
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    hashes = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded(hashes))
    print(json.dumps(json.loads(payloads["summary.json"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
