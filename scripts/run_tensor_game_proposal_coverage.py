#!/usr/bin/env python3
"""Run the CPU-only Tensor Game proposal-coverage diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates_memory_efficient
from matmul_inspector.tensor_game_proposal_coverage import DEFAULT_WIDTHS, evaluate_state, stable_json


def trajectory_states(name, actions, order, depths=None):
    residual = matrix_multiplication_tensor(2, 2, 2)
    selected = range(len(order)) if depths is None else depths
    selected = set(selected)
    rows = []
    for depth, term_index in enumerate(order):
        if depth in selected:
            rows.append((f"{name}/depth-{depth}", residual, actions[term_index], list(order), depth))
        residual = subtract_action(residual, actions[term_index])
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=DEFAULT_WIDTHS)
    parser.add_argument("--include-pair-records", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    conventional = standard_2x2_decomposition()
    strassen = strassen_2x2_decomposition()
    states = []
    states += trajectory_states("conventional/forward", conventional, tuple(range(8)))
    states += trajectory_states("conventional/reverse", conventional, tuple(reversed(range(8))), (1, 4, 7))
    states += trajectory_states("strassen/forward", strassen, tuple(range(7)))
    states += trajectory_states("strassen/reverse", strassen, tuple(reversed(range(7))), (1, 3, 6))

    tracemalloc.start()
    wall_started = time.perf_counter()
    compact_states = []
    candidate_path = args.output / "candidates.jsonl"
    pair_path = args.output / "pair_records.jsonl" if args.include_pair_records else None
    pair_handle = pair_path.open("w") if pair_path else None
    with candidate_path.open("w") as candidates:
        for label, residual, expected, order, depth in states:
            result = evaluate_state(residual, widths=args.widths, expected_action=expected)
            _, provider_audit = bounded_residual_candidates_memory_efficient(residual, 128, 128)
            for row in result.pop("actions"):
                candidates.write(json.dumps({"state_label": label, **row}, sort_keys=True, separators=(",", ":")) + "\n")
            pair_rows = result.pop("pairs")
            if pair_handle:
                for row in pair_rows:
                    pair_handle.write(json.dumps({"state_label": label, **row}, sort_keys=True, separators=(",", ":")) + "\n")
            result.update({
                "state_label": label,
                "family": label.split("/")[0],
                "term_order": order,
                "trajectory_depth": depth,
                "provider_bounded_128_ms": provider_audit["total_ms"],
            })
            compact_states.append(result)
    if pair_handle:
        pair_handle.close()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    known = [row["known_action"] for row in compact_states]
    strassen_known = [row["known_action"] for row in compact_states if row["family"] == "strassen"]
    by_width = {}
    for width in args.widths:
        rows = [next(item for item in state["counterfactual_widths"] if item["width"] == width) for state in compact_states]
        by_width[str(width)] = {
            "known_action_recall": sum(item["known_action_recalled"] for item in rows) / len(rows),
            "strassen_action_recall": sum(item["known_action_recalled"] for item, state in zip(rows, compact_states) if state["family"] == "strassen") / len(strassen_known),
            "mean_candidate_count": sum(item["candidate_count"] for item in rows) / len(rows),
            "mean_nonzero_preparation_fraction": sum(item["nonzero_preparation_fraction"] for item in rows) / len(rows),
            "mean_full_pareto_recall": sum(item["full_pareto_recall"] for item in rows) / len(rows),
            "exact_scoring_pair_workload_per_state": width,
        }
    summary = {
        "schema": "tensor-game-proposal-coverage-v1",
        "configuration": {"widths": list(args.widths), "coefficient_domain": [-1, 0, 1], "bounded_width": 128, "pair_records_saved": args.include_pair_records},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "state_count": len(compact_states),
        "enumerated_contractions": sum(row["enumerated_uv_pairs"] for row in compact_states),
        "evaluated_unique_complete_actions": sum(row["evaluated_complete_actions"] for row in compact_states),
        "known_action_classifications": dict(sorted(__import__("collections").Counter(row["classification"] for row in known).items())),
        "strassen_known_actions": len(strassen_known),
        "strassen_coefficient_domain_representable": sum(row["complete_action_in_coefficient_domain"] for row in strassen_known),
        "strassen_provider_w_reachable": sum(row["complete_action_reachable"] for row in strassen_known),
        "strassen_recalled_at_128": sum(row["retained_by_bounded_128"] for row in strassen_known),
        "counterfactual_widths": by_width,
        "wall_seconds": time.perf_counter() - wall_started,
        "peak_python_bytes": peak,
        "historical_failed_states": "not included: prior artifacts do not serialize exact residual tensors",
    }
    (args.output / "states.json").write_text(stable_json({"states": compact_states}))
    (args.output / "summary.json").write_text(stable_json(summary))
    config = {"command": " ".join(sys.argv), "state_labels": [row["state_label"] for row in compact_states]}
    (args.output / "configuration.json").write_text(stable_json(config))
    hashes = {}
    for path in sorted(args.output.iterdir()):
        if path.name != "artifact_hashes.json":
            hashes[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (args.output / "artifact_hashes.json").write_text(stable_json(hashes))
    print(stable_json(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
