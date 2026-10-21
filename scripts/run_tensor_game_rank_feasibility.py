#!/usr/bin/env python3
"""Run the bounded CPU-only Tensor Game rank-feasibility study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    zero_tensor,
)
from matmul_inspector.tensor_game_rank_feasibility import (
    feasibility_record,
    provider_child_feasibility,
    sample_frozen_rank7_failures,
    tensor_digest,
    trajectory_feasibility,
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def tensor_tuple(payload):
    return tuple(tuple(tuple(row) for row in plane) for plane in payload)


def deterministic_orders(length):
    forward = tuple(range(length))
    reverse = tuple(reversed(forward))
    even_then_odd = tuple(range(0, length, 2)) + tuple(range(1, length, 2))
    return (("forward", forward), ("reverse", reverse), ("even_then_odd", even_then_odd))


def grouped_counts(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        str(value): {
            "states": len(group),
            "provably_infeasible": sum(row["provably_infeasible"] for row in group),
            "fraction_provably_infeasible": (
                sum(row["provably_infeasible"] for row in group) / len(group)
            ),
            "lower_bound_histogram": dict(sorted(Counter(row["lower_bound"] for row in group).items())),
        }
        for value, group in sorted(groups.items())
    }


def equal_norm_witnesses(rows):
    by_norm = defaultdict(list)
    for row in rows:
        by_norm[row["squared_norm"]].append(row)
    witnesses = []
    for norm, group in sorted(by_norm.items()):
        bounds = sorted({row["lower_bound"] for row in group})
        if len(bounds) > 1:
            examples = []
            for bound in bounds:
                row = next(item for item in group if item["lower_bound"] == bound)
                examples.append({
                    "residual_digest": row["residual_digest"],
                    "lower_bound": bound,
                    "remaining_budget": row["remaining_budget"],
                    "source": row.get("source", "provider_child"),
                })
            witnesses.append({"squared_norm": norm, "bounds": bounds, "examples": examples})
    return witnesses


def sibling_bound_witnesses(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["parent_digest"], row["squared_norm"], row["remaining_budget"])].append(row)
    witnesses = []
    for (parent_digest, norm, budget), group in sorted(groups.items()):
        bounds = sorted({row["lower_bound"] for row in group})
        if len(bounds) > 1:
            witnesses.append({
                "parent_digest": parent_digest,
                "squared_norm": norm,
                "remaining_budget": budget,
                "bounds": bounds,
                "children": [{
                    "residual_digest": row["residual_digest"],
                    "provider_rank": row["provider_rank"],
                    "lower_bound": row["lower_bound"],
                    "provably_infeasible": row["provably_infeasible"],
                } for row in group],
            })
    return witnesses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--samples-per-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    target = matrix_multiplication_tensor(2, 2, 2)

    trajectory_rows = []
    families = (
        ("conventional", standard_2x2_decomposition()),
        ("strassen", strassen_2x2_decomposition()),
    )
    for family, actions in families:
        for order_name, order in deterministic_orders(len(actions)):
            rows = trajectory_feasibility(target, actions, order, family=family)
            for row in rows:
                row["order_name"] = order_name
            trajectory_rows.extend(rows)
    if any(row["provably_infeasible"] for row in trajectory_rows):
        raise AssertionError("a known valid completion was incorrectly pruned")

    zero = feasibility_record(zero_tensor((4, 4, 4)), 0, known_completion=True)
    original = feasibility_record(target, 7, known_completion=True)
    failed_rows, failure_audit = sample_frozen_rank7_failures(
        target,
        node_budget=args.node_budget,
        beam_width=args.beam_width,
        max_actions=args.max_actions,
        samples_per_depth=args.samples_per_depth,
    )
    if time.perf_counter() - started > args.cpu_seconds:
        raise TimeoutError("explicit CPU runtime limit exceeded after failure-state sampling")

    child_rows = []
    child_state_audits = []
    parent_states = [{
        "source": "original_tensor",
        "depth": 0,
        "remaining_budget": 7,
        "residual": target,
        "residual_digest": tensor_digest(target),
    }]
    parent_states.extend({
        "source": row["source"],
        "depth": row["depth"],
        "remaining_budget": row["remaining_budget"],
        "residual": tensor_tuple(row["residual"]),
        "residual_digest": row["residual_digest"],
    } for row in failed_rows if row["remaining_budget"] > 0)
    for parent in parent_states:
        if time.perf_counter() - started > args.cpu_seconds:
            raise TimeoutError("explicit CPU runtime limit exceeded during child analysis")
        children, audit = provider_child_feasibility(
            parent["residual"], parent["remaining_budget"], max_actions=args.max_actions
        )
        for child in children:
            child.update({
                "source": "bounded_128_child",
                "parent_source": parent["source"],
                "parent_digest": parent["residual_digest"],
                "parent_depth": parent["depth"],
                "depth": parent["depth"] + 1,
            })
        child_rows.extend(children)
        child_state_audits.append({
            "parent_digest": parent["residual_digest"],
            "parent_depth": parent["depth"],
            "children": len(children),
            "provably_infeasible": sum(row["provably_infeasible"] for row in children),
            "provider_audit": audit,
        })

    all_diagnostic_rows = failed_rows + child_rows
    witnesses = equal_norm_witnesses(all_diagnostic_rows)
    sibling_witnesses = sibling_bound_witnesses(child_rows)
    child_rank_ns = sum(row["rank_check_ns"] for row in child_rows)
    summary = {
        "schema": "tensor_game_rank_feasibility_summary.v1",
        "original_tensor": original,
        "zero_tensor": zero,
        "known_trajectory_states": len(trajectory_rows),
        "known_trajectory_incorrect_prunes": sum(row["provably_infeasible"] for row in trajectory_rows),
        "trajectory_lower_bounds": {
            f"{family}:{order_name}": [
                row["lower_bound"] for row in trajectory_rows
                if row["family"] == family and row["order_name"] == order_name
            ]
            for family, actions in families
            for order_name, _ in deterministic_orders(len(actions))
        },
        "failed_search_sample": {
            "states": len(failed_rows),
            "provably_infeasible": sum(row["provably_infeasible"] for row in failed_rows),
            "fraction_provably_infeasible": (
                sum(row["provably_infeasible"] for row in failed_rows) / len(failed_rows)
                if failed_rows else 0.0
            ),
            "by_depth": grouped_counts(failed_rows, "depth"),
            "by_norm": grouped_counts(failed_rows, "squared_norm"),
        },
        "provider_children": {
            "states": len(child_rows),
            "provably_infeasible": sum(row["provably_infeasible"] for row in child_rows),
            "fraction_provably_infeasible": (
                sum(row["provably_infeasible"] for row in child_rows) / len(child_rows)
                if child_rows else 0.0
            ),
            "by_depth": grouped_counts(child_rows, "depth"),
            "by_parent_norm": grouped_counts(child_rows, "parent_squared_norm"),
            "rank_check_total_ms": child_rank_ns / 1e6,
            "rank_check_mean_us": child_rank_ns / max(1, len(child_rows)) / 1e3,
            "provider_generation_total_ms": sum(
                row["provider_audit"]["total_ms"] for row in child_state_audits
            ),
            "pruned_mean_provider_rank": (
                sum(row["provider_rank"] for row in child_rows if row["provably_infeasible"])
                / max(1, sum(row["provably_infeasible"] for row in child_rows))
            ),
            "retained_mean_provider_rank": (
                sum(row["provider_rank"] for row in child_rows if not row["provably_infeasible"])
                / max(1, sum(not row["provably_infeasible"] for row in child_rows))
            ),
        },
        "equal_norm_different_bound_witnesses": len(witnesses),
        "same_parent_equal_norm_different_bound_witnesses": len(sibling_witnesses),
        "total_runtime_seconds": time.perf_counter() - started,
        "historical_state_availability": (
            "Prior search artifacts did not serialize exact residual tensors; "
            "all failed states here are newly generated."
        ),
    }
    configuration = {
        "schema": "tensor_game_rank_feasibility_configuration.v1",
        "command": " ".join(sys.argv),
        "configuration": vars(args) | {
            "output": str(args.output),
            "target": "2x2_matrix_multiplication_tensor",
            "tensor_shape": [4, 4, 4],
            "rank_budget": 7,
            "pair_budget": 128,
            "coefficient_domain": [-1, 0, 1],
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "seed_note": "The frozen baseline provider and beam tie order are deterministic; seed is recorded but unused.",
    }
    payloads = {
        "configuration.json": encoded(configuration),
        "known_trajectories.json": encoded({
            "schema": "tensor_game_rank_feasibility_trajectories.v1", "states": trajectory_rows
        }),
        "failed_states.json": encoded({
            "schema": "tensor_game_rank_feasibility_failures.v1",
            "search_audit": failure_audit,
            "states": failed_rows,
        }),
        "child_states.json": encoded({
            "schema": "tensor_game_rank_feasibility_children.v1",
            "parent_audits": child_state_audits,
            "states": child_rows,
        }),
        "equal_norm_witnesses.json": encoded({
            "schema": "tensor_game_rank_feasibility_equal_norm.v1",
            "witnesses": witnesses,
            "same_parent_child_witnesses": sibling_witnesses,
        }),
        "summary.json": encoded(summary),
    }
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    hashes = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_rank_feasibility_hashes.v1", "files": hashes
    }))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
