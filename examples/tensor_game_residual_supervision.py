#!/usr/bin/env python3
"""Run the offline TG-4H residual-supervision feasibility study."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

from matmul_inspector.tensor_game import subtract_action, zero_tensor
from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_ablation import load_ablation_checkpoint
from matmul_inspector.tensor_game_residual_supervision import (
    COEFFICIENTS,
    SHAPE,
    TEACHER_VERSION,
    action_round_trip,
    assert_heldout_excluded,
    canonical_json_digest,
    enumerate_residual_teacher,
    greedy_teacher_continuation,
    residual_score_distribution,
    select_states,
    sha256,
    support_overlap,
    validate_prototype_record,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    return json.loads(path.read_text())


def bool_rank(ids: list[str], identifier: str) -> int | None:
    return ids.index(identifier) + 1 if identifier in ids else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-repeats", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.runtime_repeats < 2:
        raise ValueError("at least two runtime repeats are required")

    training_manifest = load_json(args.training / "manifest.json")
    decisions = load_json(args.training / "decisions.json")
    calibration = load_json(args.calibration)
    heldout = load_json(args.heldout)
    states = select_states(training_manifest, calibration, decisions)
    assert_heldout_excluded(states, heldout)

    torch = __import__("torch")
    frozen_model, frozen_config, _ = load_checkpoint(args.frozen_checkpoint, "cpu")
    trained_model, trained_config, _ = load_ablation_checkpoint(
        args.trained_checkpoint, device="cpu"
    )
    args.output.mkdir(parents=True)

    input_hashes = {
        "training_manifest": sha256(args.training / "manifest.json"),
        "training_decisions": sha256(args.training / "decisions.json"),
        "calibration_targets": sha256(args.calibration),
        "heldout_exclusion_manifest": sha256(args.heldout),
        "frozen_checkpoint": sha256(args.frozen_checkpoint / "best_model_state.pt"),
        "trained_checkpoint": sha256(args.trained_checkpoint / "best_model_state.pt"),
    }
    provenance = {
        "teacher_version": TEACHER_VERSION,
        "selection": "first two target_ids per source and constructed depth before teacher scoring",
        "sources": {"training": 8, "calibration": 8},
        "runtime_repeats": args.runtime_repeats,
        "shape": list(SHAPE),
        "coefficients": list(COEFFICIENTS),
        "teacher_limit": 12,
        "input_hashes": input_hashes,
    }

    representation_rows = []
    comparison_rows = []
    multistep_rows = []
    runtime_rows = []
    policy_runtime_rows = []
    labels = []
    teacher_stats = []
    for state in states:
        residual = state["residual"]
        reference = action_from_payload(state["reference_actions"][0])
        reference_id = action_identity(reference)
        teacher, audit = enumerate_residual_teacher(residual, 12)
        teacher_stats.append(audit)
        probabilities = residual_score_distribution(teacher, temperature=1.0)
        frozen = decode(frozen_model, frozen_config, residual, 256)
        trained = decode(trained_model, trained_config, residual, 256)
        frozen_ids = [action_identity(action) for action, _ in frozen]
        trained_ids = [action_identity(action) for action, _ in trained]
        for provider, model, config in (
            ("frozen_top12", frozen_model, frozen_config),
            ("trained_top12", trained_model, trained_config),
        ):
            decode(model, config, residual, 12)
            for repeat in range(args.runtime_repeats):
                started = time.perf_counter()
                proposals = decode(model, config, residual, 12)
                policy_runtime_rows.append(
                    {
                        "state_id": state["state_id"],
                        "target_id": state["target_id"],
                        "source": state["source"],
                        "constructed_depth": state["constructed_depth"],
                        "provider": provider,
                        "repeat": repeat,
                        "returned_candidates": len(proposals),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
        mcts = state["mcts_decision"]["sampled_actions"] if state["mcts_decision"] else []
        overlap = support_overlap((row["action_id"] for row in teacher), mcts)
        mcts_by_id = {row["action_id"]: row for row in mcts}
        teacher_ids = [row["action_id"] for row in teacher]
        comparison_rows.append(
            {
                "state_id": state["state_id"],
                "target_id": state["target_id"],
                "source": state["source"],
                "constructed_depth": state["constructed_depth"],
                "teacher_mcts_overlap": overlap["overlap_count"],
                "teacher_actions_absent_from_mcts": overlap["teacher_absent_from_mcts"],
                "mcts_support_size": overlap["mcts_count"],
                "mcts_probability_on_teacher": overlap["mcts_probability_on_teacher"],
                "teacher_reference_rank": bool_rank(teacher_ids, reference_id),
                "mcts_reference_present": reference_id in mcts_by_id,
                "mcts_reference_probability": mcts_by_id.get(reference_id, {}).get(
                    "search_probability", 0.0
                ),
                "frozen_reference_rank_256": bool_rank(frozen_ids, reference_id),
                "trained_reference_rank_256": bool_rank(trained_ids, reference_id),
            }
        )
        for row, probability in zip(teacher, probabilities):
            action = action_from_payload(row["action"])
            representation_rows.append(
                {
                    "state_id": state["state_id"],
                    "target_id": state["target_id"],
                    "source": state["source"],
                    "constructed_depth": state["constructed_depth"],
                    "action_id": row["action_id"],
                    "teacher_rank": row["rank"],
                    "teacher_score": row["score"],
                    "legal": True,
                    "representable": action_round_trip(action),
                    "round_trip_exact": action_round_trip(action),
                    "frozen_rank_256": bool_rank(frozen_ids, row["action_id"]),
                    "trained_rank_256": bool_rank(trained_ids, row["action_id"]),
                    "in_mcts_support": row["action_id"] in mcts_by_id,
                    "mcts_target_probability": mcts_by_id.get(row["action_id"], {}).get(
                        "search_probability", 0.0
                    ),
                    "illustrative_teacher_probability_t1": probability,
                }
            )
        for first in teacher[:3]:
            action = action_from_payload(first["action"])
            next_residual = subtract_action(residual, action)
            if next_residual == zero_tensor(SHAPE):
                continuation = {
                    "success": True,
                    "steps": 0,
                    "final_residual_norm_squared": 0,
                }
            else:
                continuation = greedy_teacher_continuation(
                    next_residual, state["constructed_depth"] - 1
                )
            multistep_rows.append(
                {
                    "state_id": state["state_id"],
                    "target_id": state["target_id"],
                    "source": state["source"],
                    "constructed_depth": state["constructed_depth"],
                    "first_action_rank": first["rank"],
                    "first_action_id": first["action_id"],
                    "first_action_score": first["score"],
                    "first_action_is_reference": first["action_id"] == reference_id,
                    "bounded_greedy_continuation_success": continuation["success"],
                    "continuation_steps": continuation["steps"],
                    "final_residual_norm_squared": continuation[
                        "final_residual_norm_squared"
                    ],
                    "best_score_tie_count": audit["minimum_score_ties"],
                }
            )
        # Warm up separately, then retain every repeated observation.
        enumerate_residual_teacher(residual, 12)
        for repeat in range(args.runtime_repeats):
            _, timing = enumerate_residual_teacher(residual, 12)
            runtime_rows.append(
                {
                    "state_id": state["state_id"],
                    "target_id": state["target_id"],
                    "source": state["source"],
                    "constructed_depth": state["constructed_depth"],
                    "repeat": repeat,
                    **timing,
                }
            )
        label = {
            "state_id": state["state_id"],
            "target_id": state["target_id"],
            "target_digest": state["target_digest"],
            "source": state["source"],
            "constructed_depth": state["constructed_depth"],
            "residual": residual,
            "teacher_actions": [
                {**row, "illustrative_probability_temperature_1": probability}
                for row, probability in zip(teacher, probabilities)
            ],
            "mcts_support": mcts,
            "known_successful_continuation": state["reference_actions"],
            "encoding_round_trip_verified": all(
                action_round_trip(action_from_payload(row["action"])) for row in teacher
            ),
            "provenance": provenance,
        }
        validate_prototype_record(
            label, {record["target_digest"] for record in heldout["targets"]}
        )
        labels.append(label)

    write_csv(args.output / "representation_audit.csv", representation_rows)
    write_csv(args.output / "supervision_comparison.csv", comparison_rows)
    write_csv(args.output / "multistep_analysis.csv", multistep_rows)
    write_csv(args.output / "teacher_runtime.csv", runtime_rows)
    write_csv(args.output / "policy_runtime.csv", policy_runtime_rows)
    with (args.output / "prototype_labels.jsonl").open("w") as stream:
        for label in labels:
            stream.write(json.dumps(label, sort_keys=True, separators=(",", ":")) + "\n")

    runtime_by_depth = {}
    for depth in range(1, 5):
        values = [
            row["elapsed_seconds"]
            for row in runtime_rows
            if row["constructed_depth"] == depth
        ]
        runtime_by_depth[str(depth)] = {
            "observations": len(values),
            "minimum_seconds": min(values),
            "median_seconds": statistics.median(values),
            "mean_seconds": statistics.mean(values),
            "maximum_seconds": max(values),
            "stdev_seconds": statistics.stdev(values),
        }
    teacher_characterization = {
        "teacher_version": TEACHER_VERSION,
        "enumeration": "all 80 nonzero ternary u vectors crossed with all 80 nonzero ternary v vectors",
        "enumerated_uv_pairs_per_state": 6400,
        "w_derivation": "exact residual contraction divided by ||u||^2||v||^2, quantized to {-1,0,1}",
        "score": "exact squared norm of residual minus canonical rank-one action",
        "ordering": "ascending score, then lexicographic canonical u, v, w",
        "exhaustive_scope": "exhaustive over u,v with deterministic derived w; not all 80^3 rank-one triples",
        "exact_residual_required": True,
        "ties_supported": True,
        "complexity": "O(80^2 * (4^3 + canonicalization)) for fixed 4x4x4 shape",
        "runtime": runtime_by_depth,
        "candidate_counts": {
            "minimum_unique": min(row["unique_legal_candidates"] for row in teacher_stats),
            "mean_unique": statistics.mean(
                row["unique_legal_candidates"] for row in teacher_stats
            ),
            "maximum_unique": max(row["unique_legal_candidates"] for row in teacher_stats),
        },
        "memory": {
            "implementation_behavior": "materializes one Python dictionary entry per unique canonical candidate before sorting",
            "analytical_payload_lower_bound_bytes_per_candidate": 104,
            "lower_bound_basis": "12 signed 64-bit coefficients plus one 64-bit score; excludes Python object and hash-table overhead",
        },
    }
    (args.output / "teacher_characterization.json").write_text(
        json.dumps(teacher_characterization, indent=2, sort_keys=True) + "\n"
    )

    policy_runtime = {}
    for provider in ("frozen_top12", "trained_top12"):
        values = [
            row["elapsed_seconds"]
            for row in policy_runtime_rows
            if row["provider"] == provider
        ]
        policy_runtime[provider] = {
            "observations": len(values),
            "minimum_seconds": min(values),
            "median_seconds": statistics.median(values),
            "mean_seconds": statistics.mean(values),
            "maximum_seconds": max(values),
            "stdev_seconds": statistics.stdev(values),
        }

    summary = {
        "git": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_clean": not bool(
                subprocess.check_output(["git", "status", "--short"], text=True).strip()
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "provenance": provenance,
        "selected_states": len(states),
        "prototype_records": len(labels),
        "prototype_sha256": sha256(args.output / "prototype_labels.jsonl"),
        "prototype_canonical_digest": canonical_json_digest(labels),
        "heldout_exclusion_verified": True,
        "representation": {
            "teacher_actions": len(representation_rows),
            "legal": sum(row["legal"] for row in representation_rows),
            "representable": sum(row["representable"] for row in representation_rows),
            "frozen_accessible_at_256": sum(
                row["frozen_rank_256"] is not None for row in representation_rows
            ),
            "trained_accessible_at_256": sum(
                row["trained_rank_256"] is not None for row in representation_rows
            ),
        },
        "supervision": {
            "states_with_mcts_support": sum(
                row["mcts_support_size"] > 0 for row in comparison_rows
            ),
            "teacher_actions": len(representation_rows),
            "teacher_mcts_overlap": sum(
                row["teacher_mcts_overlap"] for row in comparison_rows
            ),
            "teacher_actions_absent_from_mcts": sum(
                row["teacher_actions_absent_from_mcts"]
                for row in comparison_rows
                if row["mcts_support_size"] > 0
            ),
            "teacher_reference_recall": sum(
                row["teacher_reference_rank"] is not None for row in comparison_rows
            ),
            "mcts_reference_recall": sum(
                row["mcts_reference_present"] for row in comparison_rows
            ),
        },
        "multistep": {
            "tested_first_actions": len(multistep_rows),
            "successful_bounded_greedy_continuations": sum(
                row["bounded_greedy_continuation_success"] for row in multistep_rows
            ),
            "top1_states": len(states),
            "top1_successes": sum(
                row["bounded_greedy_continuation_success"]
                for row in multistep_rows
                if row["first_action_rank"] == 1
            ),
            "top1_reference_agreements": sum(
                row["first_action_is_reference"]
                for row in multistep_rows
                if row["first_action_rank"] == 1
            ),
        },
        "teacher_runtime": runtime_by_depth,
        "policy_runtime": policy_runtime,
        "input_hashes": input_hashes,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
