#!/usr/bin/env python3
"""Run the experiment-only TG-4D candidate coverage investigation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game_candidate_coverage import (
    aggregate_proposals, aggregate_recall, audit_training_coverage,
    recall_sweep, run_proposal_experiments, write_csv,
)
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_ablation import load_ablation_checkpoint


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(summary: dict) -> str:
    recall = summary["recall"]
    proposals = summary["proposal_experiments"]
    coverage = summary["training_coverage"]
    return f"""# TG-4D candidate recall and policy coverage investigation

## Result

Candidate coverage is the immediate bottleneck on the four TG-4C critical
targets. The frozen policy recalls all four exact actions at top-12. The trained
policy recalls none through the bounded top-256 audit. Restoring frozen-policy
coverage recovers all 24 fixed-seed discoveries without changing model weights.

The trained policy shows no measurable advantage on this narrow one-step set.
The equal-budget hybrid matches frozen top-12 success and search expansions, but
requires both model inferences. The expanded union also matches success while
doubling candidates from 12 to 24. Runtime values are descriptive CPU timings,
not a performance claim.

## Candidate recall

| Policy | Top-1 | Top-12 | Top-32 | Top-64 | Top-128 | Top-256 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | {recall['frozen']['unique_target_recall']['1']}/4 | {recall['frozen']['unique_target_recall']['12']}/4 | {recall['frozen']['unique_target_recall']['32']}/4 | {recall['frozen']['unique_target_recall']['64']}/4 | {recall['frozen']['unique_target_recall']['128']}/4 | {recall['frozen']['unique_target_recall']['256']}/4 |
| Trained | {recall['trained']['unique_target_recall']['1']}/4 | {recall['trained']['unique_target_recall']['12']}/4 | {recall['trained']['unique_target_recall']['32']}/4 | {recall['trained']['unique_target_recall']['64']}/4 | {recall['trained']['unique_target_recall']['128']}/4 | {recall['trained']['unique_target_recall']['256']}/4 |

Repeated-seed denominators are 24 runs. The unique-target denominator is four.
An absent action at top-256 is reported as `>256` or unknown, not assigned an
exhaustive rank. Each cutoff invokes the existing decoder with that cutoff because
its internal beam width depends on the requested candidate count.

## Proposal experiments

| Configuration | Comparison | Budget | Recall | Discoveries | Mean candidates | Mean expansions | Mean search wall |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen top-12 | Equal budget | 12 | {proposals['frozen_top12']['exact_action_recall']}/24 | {proposals['frozen_top12']['exact_discoveries']}/24 | {proposals['frozen_top12']['mean_unique_candidates']:.1f} | {proposals['frozen_top12']['mean_search_expansions']:.1f} | {proposals['frozen_top12']['mean_search_wall_seconds']:.6f} s |
| Trained top-12 | Equal budget | 12 | {proposals['trained_top12']['exact_action_recall']}/24 | {proposals['trained_top12']['exact_discoveries']}/24 | {proposals['trained_top12']['mean_unique_candidates']:.1f} | {proposals['trained_top12']['mean_search_expansions']:.1f} | {proposals['trained_top12']['mean_search_wall_seconds']:.6f} s |
| Alternating hybrid | Equal budget | 12 | {proposals['alternating_hybrid12']['exact_action_recall']}/24 | {proposals['alternating_hybrid12']['exact_discoveries']}/24 | {proposals['alternating_hybrid12']['mean_unique_candidates']:.1f} | {proposals['alternating_hybrid12']['mean_search_expansions']:.1f} | {proposals['alternating_hybrid12']['mean_search_wall_seconds']:.6f} s |
| Top-12 union | Expanded budget | 24 | {proposals['union_top12']['exact_action_recall']}/24 | {proposals['union_top12']['exact_discoveries']}/24 | {proposals['union_top12']['mean_unique_candidates']:.1f} | {proposals['union_top12']['mean_search_expansions']:.1f} | {proposals['union_top12']['mean_search_wall_seconds']:.6f} s |

The hybrid alternates frozen then trained candidates by source rank, applies
canonical deduplication, and stops at 12. The union concatenates both top-12
lists, deduplicates, and retains an expanded budget of 24. These are diagnostic
policies only. Production search behavior is unchanged.

## Training-data coverage

The training partition contains {coverage['training_decisions']} decisions and
{coverage['distinct_training_support_actions']} distinct canonical support actions.
Its median action occurs twice, {coverage['training_action_occurrence_distribution']['singletons']}
actions occur once, and the maximum frequency is
{coverage['training_action_occurrence_distribution']['maximum']}.

Three of the four critical actions never occur in a training support and receive
zero direct target mass. The fourth occurs four times with total target mass
0.263158. This is weak direct supervision. It supports rare-action
underrepresentation, but it does not prove causation. These matches compare exact
canonical actions across different residual states. They do not show that the
critical held-out state or an equivalent state was present during training.

Support-conditional normalization assigns no gradient to an action absent from a
recorded support. That is a property of the objective. The audit does not isolate
normalization from dataset collection as the cause of the regression.

## Hypotheses

- **H1 confirmed for the critical set:** restoring exact-action access recovers 24/24 lost successes.
- **H2 unsupported on this set:** the trained half adds no success, expansion, or candidate-count advantage to the equal-budget hybrid.
- **H3 supported but not causal:** three actions have no direct training support; the fourth has low effective target mass.
- **H4 confirmed:** trained-budget increases through 256 recover none of the four actions.

## Limitations

This diagnostic intentionally uses the four one-step targets selected by TG-4C.
It is not a fresh held-out generalization estimate. Search has only one expansion,
so it cannot show hybrid benefits on longer trajectories. The six seeds repeat
deterministic proposal sets and vary search only. Sequential CPU timings include
Python and inference overhead and are provided only to describe observed cost.

## Recommended TG-4E experiment

Preregister a fresh target set containing one- through four-step problems. Compare
frozen top-12 against the deterministic equal-budget hybrid with identical MCTS,
value guidance, targets, and seeds. Persist per-decision oracle-action recall,
rank, proposal-source attribution, simulations to first success, expansions, and
inference time. This tests whether coverage protection helps beyond the four
diagnostic one-step targets and whether trained candidates add value later in a
trajectory. Do not tune the hybrid on the existing TG-4C test targets.

## Reproduction

```sh
PYTHONPATH=src python examples/tensor_game_candidate_coverage.py \\
  --tg4c artifacts/tensor-game/tg4c-policy-regression-v1/analysis.json \\
  --trials artifacts/tensor-game/tg4b2c-heldout-v1/trials \\
  --experience artifacts/tensor-game/tg4b2a-diverse-full-v1 \\
  --baseline-checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \\
  --trained-checkpoint artifacts/tensor-game/tg4b2b-offline-ablation-v1/candidates/policy_only_natural \\
  --output artifacts/tensor-game/tg4d-candidate-coverage-v1
```

Git commit inspected: `{summary['provenance']['git_commit']}`. Exact checkpoint
and input hashes are recorded in `summary.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg4c", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tg4c = json.loads(args.tg4c.read_text())
    trials = [json.loads(path.read_text()) for path in args.trials.glob("*.json")]
    successful = {row["target_group_id"]: row for row in trials
                  if row["configuration_id"] == "frozen_policy_heuristic_value" and row["exact_reconstruction"]}
    critical = {}
    for target, diagnostic in tg4c["fixed_state_model_diagnostics"].items():
        decision = successful[target]["result"]["decisions"][0]
        critical[target] = {
            "gold_action": diagnostic["gold_action"], "residual": decision["residual"],
            "target": tuple(tuple(tuple(x for x in row) for row in plane)
                            for plane in successful[target]["result"]["episode"]["initial_target"]),
        }
    frozen, frozen_config, _ = load_checkpoint(args.baseline_checkpoint, "cpu")
    trained, trained_config, _ = load_ablation_checkpoint(args.trained_checkpoint, device="cpu")
    models = {"frozen": (frozen, frozen_config), "trained": (trained, trained_config)}
    recall = recall_sweep(critical, models)
    proposals = run_proposal_experiments(critical, models)
    coverage = audit_training_coverage(args.experience, critical)
    summary = {
        "schema_version": 1, "experiment": "tg4d_candidate_recall_and_policy_coverage",
        "scope": {"critical_unique_targets": 4, "seeds": [3, 11, 19, 27, 35, 43],
                  "candidate_cutoffs": [1, 12, 32, 64, 128, 256]},
        "recall": aggregate_recall(recall), "proposal_experiments": aggregate_proposals(proposals),
        "training_coverage": coverage,
        "configuration": {"simulations": 200, "c_puct": 1.5, "node_limit": 200,
                          "max_moves": 1, "value": "nonzero_heuristic", "timeout_seconds": 5.0,
                          "hybrid_rule": "alternate frozen then trained by source rank; canonical deduplication; stop at 12",
                          "union_rule": "frozen top-12 followed by trained top-12; canonical deduplication; expanded budget 24"},
        "provenance": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "tg4c_sha256": digest(args.tg4c),
                       "baseline_checkpoint_sha256": digest(args.baseline_checkpoint / "best_model_state.pt"),
                       "trained_checkpoint_sha256": digest(args.trained_checkpoint / "best_model_state.pt"),
                       "decisions_sha256": digest(args.experience / "decisions.json")},
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "training_coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(report(summary))
    write_csv(args.output / "candidate_recall.csv", recall)
    write_csv(args.output / "proposal_union.csv", proposals)
    write_csv(args.output / "training_coverage.csv", coverage["critical_actions"])
    print(args.output)


if __name__ == "__main__":
    main()
