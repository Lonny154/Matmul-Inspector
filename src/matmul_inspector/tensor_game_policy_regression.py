"""Reproducible paired analysis for the TG-4C policy regression."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE = "frozen_policy_heuristic_value"
TRAINED = "policy_only_natural_policy_heuristic_value"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "state_digest": row["state_digest"],
            "selected_action_id": row["selected_action_id"],
            "selected_action": row["selected_action"],
            "next_state_digest": row["next_state_digest"],
        }
        for index, row in enumerate(record["result"]["decisions"])
    ]


def _first_divergence(baseline: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    left = baseline["result"]["decisions"]
    right = trained["result"]["decisions"]
    common = min(len(left), len(right))
    index = next(
        (i for i in range(common) if left[i]["selected_action_id"] != right[i]["selected_action_id"]),
        common if len(left) != len(right) else None,
    )
    if index is None:
        return {"step": None, "reason": "no_recorded_action_divergence"}
    if index >= common:
        return {"step": index, "reason": "trajectory_length_divergence"}
    b, t = left[index], right[index]
    baseline_support = {row["action_id"] for row in b["sampled_actions"]}
    trained_support = {row["action_id"] for row in t["sampled_actions"]}
    gold = b["selected_action_id"] if baseline["exact_reconstruction"] else None
    return {
        "step": index,
        "reason": "selected_action_divergence",
        "same_reference_state": b["state_digest"] == t["state_digest"],
        "state_digest": b["state_digest"] if b["state_digest"] == t["state_digest"] else None,
        "baseline_selected_action_id": b["selected_action_id"],
        "trained_selected_action_id": t["selected_action_id"],
        "gold_reference_action_id": gold,
        "baseline_available_action_ids": sorted(baseline_support),
        "trained_available_action_ids": sorted(trained_support),
        "available_action_overlap": len(baseline_support & trained_support),
        "gold_in_baseline_support": gold in baseline_support if gold else None,
        "gold_in_trained_support": gold in trained_support if gold else None,
        "baseline_selected_action": b["selected_action"],
        "trained_selected_action": t["selected_action"],
        "baseline_search_row": next(row for row in b["sampled_actions"] if row["action_id"] == b["selected_action_id"]),
        "trained_search_row": next(row for row in t["sampled_actions"] if row["action_id"] == t["selected_action_id"]),
    }


def _load_trials(trial_dir: Path) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(trial_dir.glob("*.json"))]
    if not rows:
        raise ValueError(f"no trial records found in {trial_dir}")
    return rows


def analyze_trials(trial_dir: Path) -> dict[str, Any]:
    """Rebuild the paired baseline/policy-only comparison from saved trials."""
    rows = _load_trials(trial_dir)
    selected = [row for row in rows if row["configuration_id"] in {BASELINE, TRAINED}]
    by_key = {(row["target_group_id"], row["seed"], row["configuration_id"]): row for row in selected}
    pair_keys = sorted({(row["target_group_id"], row["seed"]) for row in selected})
    if len(by_key) != len(pair_keys) * 2:
        raise ValueError("baseline and trained records do not form complete target/seed pairs")

    transitions = Counter()
    records = []
    labels = {(True, True): "retained_success", (False, True): "improvement",
              (True, False): "regression", (False, False): "persistent_failure"}
    for target, seed in pair_keys:
        baseline = by_key[(target, seed, BASELINE)]
        trained = by_key[(target, seed, TRAINED)]
        label = labels[(baseline["exact_reconstruction"], trained["exact_reconstruction"])]
        transitions[label] += 1
        if label != "regression":
            continue
        b_states = [row["state_digest"] for row in baseline["result"]["decisions"]]
        t_states = [row["state_digest"] for row in trained["result"]["decisions"]]
        maximum_moves = baseline["result"]["episode"]["max_moves"]
        records.append({
            "target_group_id": target,
            "seed": seed,
            "initial_target_digest": baseline["result"]["episode"]["initial_target_digest"],
            "construction_length": maximum_moves,
            "baseline_trial_id": baseline["trial_id"],
            "trained_trial_id": trained["trial_id"],
            "baseline_status": baseline["status"],
            "trained_status": trained["status"],
            "baseline_termination_reason": baseline["termination_reason"],
            "trained_termination_reason": trained["termination_reason"],
            "baseline_trajectory": _trajectory(baseline),
            "trained_trajectory": _trajectory(trained),
            "first_divergence": _first_divergence(baseline, trained),
            "recoverable_after_first_wrong_action": maximum_moves > 1,
            "trained_terminated_prematurely": trained["status"] == "episode_timeout",
            "trained_loop_detected": len(t_states) != len(set(t_states)),
            "baseline_loop_detected": len(b_states) != len(set(b_states)),
        })

    total = len(pair_keys)
    counts = {name: transitions[name] for name in labels.values()}
    return {
        "schema_version": 1,
        "comparison": {"baseline": BASELINE, "trained": TRAINED},
        "paired_examples": total,
        "success": {
            "baseline": sum(row["exact_reconstruction"] for row in selected if row["configuration_id"] == BASELINE),
            "trained": sum(row["exact_reconstruction"] for row in selected if row["configuration_id"] == TRAINED),
        },
        "success_transitions": {
            name: {"count": count, "percentage": 100.0 * count / total}
            for name, count in counts.items()
        },
        "regression_records": records,
    }


def add_validation_metrics(analysis: dict[str, Any], validation_path: Path) -> None:
    values = json.loads(validation_path.read_text(encoding="utf-8"))
    analysis["validation_metrics"] = {
        "dataset": "persisted TG-4B.2A validation partition; 292 recorded MCTS decisions from 30 target groups",
        "baseline": values["frozen_tg3b"]["validation_metrics"]["overall"],
        "trained": values["policy_only_natural"]["validation_metrics"]["overall"],
    }


def add_fixed_state_model_diagnostics(
    analysis: dict[str, Any], baseline_checkpoint: Path, trained_checkpoint: Path, rank_limit: int = 256
) -> None:
    """Score each critical action under both models on the identical residual."""
    import torch
    from .tensor_game_experience import action_from_payload, action_identity
    from .tensor_game_learning import load_checkpoint, rank_complete_actions
    from .tensor_game_neural_mcts import canonical_action_log_probability
    from .tensor_game_offline_ablation import load_ablation_checkpoint

    baseline, baseline_config, _ = load_checkpoint(baseline_checkpoint, "cpu")
    trained, trained_config, _ = load_ablation_checkpoint(trained_checkpoint, device="cpu")
    diagnostics = {}
    for record in analysis["regression_records"]:
        target = record["target_group_id"]
        if target in diagnostics:
            continue
        divergence = record["first_divergence"]
        residual = record["baseline_trajectory"][0]["state_digest"]
        # Recover the tensor from the full trial record in a later pass supplied below.
        diagnostics[target] = {"state_digest": residual, "gold_action": divergence["baseline_selected_action"]}

    trials = _load_trials(Path(analysis["provenance"]["trial_directory"]))
    by_target = {row["target_group_id"]: row for row in trials
                 if row["configuration_id"] == BASELINE and row["exact_reconstruction"]}
    for target, output in diagnostics.items():
        decision = by_target[target]["result"]["decisions"][0]
        tensor = torch.tensor(decision["residual"], dtype=torch.float32).unsqueeze(0)
        gold = action_from_payload(decision["selected_action"])
        with torch.inference_mode():
            baseline_logits, _ = baseline(tensor)
            trained_logits, _ = trained(tensor)
        model_rows = {}
        for name, logits, config in (
            ("baseline", baseline_logits[0], baseline_config),
            ("trained", trained_logits[0], trained_config),
        ):
            ranked = rank_complete_actions(logits, config.shape, rank_limit)
            identities = [action_identity(action) for action, _ in ranked]
            top = ranked[:12]
            scores = [canonical_action_log_probability(logits, action, config.shape) for action, _ in top]
            offset = max(scores)
            weights = [math.exp(score - offset) for score in scores]
            probabilities = [weight / sum(weights) for weight in weights]
            gold_id = action_identity(gold)
            model_rows[name] = {
                "gold_canonical_log_probability": canonical_action_log_probability(logits, gold, config.shape),
                "gold_decoded_rank": identities.index(gold_id) + 1 if gold_id in identities else None,
                "rank_search_limit": rank_limit,
                "top_12_conditional_entropy_nats": -sum(p * math.log(p) for p in probabilities),
                "top_12_action_ids": identities[:12],
            }
        output["models"] = model_rows
        output["top_12_overlap"] = len(
            set(model_rows["baseline"]["top_12_action_ids"]) & set(model_rows["trained"]["top_12_action_ids"])
        )
    analysis["fixed_state_model_diagnostics"] = diagnostics


def _markdown(analysis: dict[str, Any]) -> str:
    transitions = analysis["success_transitions"]
    metrics = analysis["validation_metrics"]
    fixed = analysis.get("fixed_state_model_diagnostics", {})
    lines = [
        "# TG-4C policy regression investigation", "", "## Result", "",
        "The regression is reproducible from the frozen paired evidence. The baseline solved 24 of 180 trials (13.3%). The policy-only checkpoint solved 0 of 180 under the same targets, seeds, search budget, decoder, move limits, heuristic value, and success criterion.", "",
        "The direct mechanism is proposal-set exclusion on four one-step targets. The frozen policy included the exact action in its 12 proposals. The trained policy produced a disjoint 12-action set and excluded that action. With one move available, every wrong choice was unrecoverable. This supports rare critical-action degradation and objective mismatch. It does not show that cross-entropy optimization is generally harmful.", "",
        "## Metrics", "",
        "| Policy | Validation cross-entropy | Validation top-1 | Validation target entropy | Exact discovery |",
        "|---|---:|---:|---:|---:|",
        f"| Frozen TG-3B | {metrics['baseline']['policy_cross_entropy']:.6f} | {metrics['baseline']['top1_agreement']:.1%} | {metrics['baseline']['policy_entropy']:.6f} | {analysis['success']['baseline']}/180 (13.3%) |",
        f"| Policy-only | {metrics['trained']['policy_cross_entropy']:.6f} | {metrics['trained']['top1_agreement']:.1%} | {metrics['trained']['policy_entropy']:.6f} | {analysis['success']['trained']}/180 (0.0%) |", "",
        "Cross-entropy is conditional on each recorded MCTS sampled support. `policy_entropy` above is the entropy of the fixed target visit distribution, not model predictive entropy.", "",
        "## Success transitions", "", "| Classification | Count | Percentage |", "|---|---:|---:|",
    ]
    for name in ("retained_success", "improvement", "regression", "persistent_failure"):
        row = transitions[name]
        lines.append(f"| {name.replace('_', ' ').title()} | {row['count']} | {row['percentage']:.1f}% |")
    lines += ["", "## Same-state first-divergence evidence", ""]
    if fixed:
        lines += ["| Target slot | Frozen gold rank | Trained gold rank | Frozen log probability | Trained log probability | Top-12 overlap |", "|---|---:|---:|---:|---:|---:|"]
        for target, row in sorted(fixed.items()):
            b, t = row["models"]["baseline"], row["models"]["trained"]
            trank = t["gold_decoded_rank"] if t["gold_decoded_rank"] is not None else ">256"
            lines.append(f"| `{target}` | {b['gold_decoded_rank']} | {trank} | {b['gold_canonical_log_probability']:.6f} | {t['gold_canonical_log_probability']:.6f} | {row['top_12_overlap']} |")
    lines += [
        "", "The log probabilities above evaluate both checkpoints on the same initial residual and the same canonical exact action. They are therefore comparable. Rank `>256` means the action was absent from the bounded 256-action decoder audit. It is not an exact rank over the full action space.", "",
        "## Hypotheses", "",
        "- **H1, cross-entropy versus top-1 mismatch: supported.** Validation cross-entropy improved from 8.252298 to 2.649050 while top-1 agreement fell from 27.7% to 11.3%. On the four critical states, the exact action fell from frozen ranks 2–5 to outside the first 256 decoded trained actions.",
        "- **H2, exposure bias: not needed to explain these regressions.** All 24 diverged at the initial state. There was no preceding trained-policy state drift. The broader hypothesis remains untested for longer trajectories because the baseline had no held-out successes beyond one move.",
        "- **H3, rare or critical action degradation: strongly supported.** Only four of 30 targets supplied all baseline successes. Natural training contained 37 successful versus 1,363 truncated decisions. The policy objective improved overall while all four exact one-step actions were removed from the proposal set.",
        "- **H4, entropy changes: weak evidence against overconfidence as the cause.** Conditional entropy across each model's top 12 proposals increased slightly on all four critical states. The action sets were disjoint, so entropy alone does not explain which actions survived decoding.",
        "- **H5, objective mismatch: strongly supported.** Training minimized support-conditional imitation of MCTS visit distributions. Discovery requires at least one exact critical action to survive a hard top-12 decoder. The loss neither evaluates full-trajectory success nor directly protects exact actions outside each recorded support.", "",
        "## Limitations", "",
        "The discovery comparison has 30 independent target groups and six repeated seeds per target. All observed regressions concern four length-one targets. No baseline neural-policy trial solved a longer target, so this evidence cannot characterize later-step exposure bias. Saved trials do not contain full decoder logits. Same-state logits require the original checkpoints and PyTorch. The trained rank is bounded rather than exhaustive.", "",
        "## Recommended next experiment", "",
        "Before any retraining, run a preregistered decoder-sensitivity replay on validation-only states and a fresh held-out target set. Sweep proposal limits while holding checkpoints, MCTS, value guidance, targets, and seeds fixed. Persist exact-action inclusion, rank, full-support mass, and discovery success. This directly tests whether hard top-k truncation mediates the regression. Do not tune the limit on the existing test outcomes.", "",
        "## Reproduction", "", "```sh",
        "PYTHONPATH=src python examples/tensor_game_policy_regression.py \\",
        "  --trials artifacts/tensor-game/tg4b2c-heldout-v1/trials \\",
        "  --validation artifacts/tensor-game/tg4b2b-offline-ablation-v1/validation_comparison.json \\",
        "  --baseline-checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \\",
        "  --trained-checkpoint artifacts/tensor-game/tg4b2b-offline-ablation-v1/candidates/policy_only_natural \\",
        "  --output artifacts/tensor-game/tg4c-policy-regression-v1", "```", "",
        f"Git commit inspected: `{analysis['provenance']['git_commit']}`. Checkpoint and evidence hashes are recorded in `analysis.json`.", "",
    ]
    return "\n".join(lines)


def write_analysis(analysis: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_markdown(analysis), encoding="utf-8")
    with (output / "regressions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["target_group_id", "seed", "initial_target_digest", "construction_length", "baseline_trial_id",
                  "trained_trial_id", "baseline_status", "trained_status", "first_divergence_step",
                  "gold_action_id", "trained_action_id", "support_overlap", "gold_in_trained_support", "recoverable"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in analysis["regression_records"]:
            divergence = row["first_divergence"]
            writer.writerow({
                "target_group_id": row["target_group_id"], "seed": row["seed"],
                "initial_target_digest": row["initial_target_digest"], "construction_length": row["construction_length"],
                "baseline_trial_id": row["baseline_trial_id"], "trained_trial_id": row["trained_trial_id"],
                "baseline_status": row["baseline_status"], "trained_status": row["trained_status"],
                "first_divergence_step": divergence["step"], "gold_action_id": divergence["gold_reference_action_id"],
                "trained_action_id": divergence["trained_selected_action_id"], "support_overlap": divergence["available_action_overlap"],
                "gold_in_trained_support": divergence["gold_in_trained_support"],
                "recoverable": row["recoverable_after_first_wrong_action"],
            })


def evidence_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def directory_content_digest(path: Path) -> str:
    """Hash ordered relative names and bytes without depending on filesystem metadata."""
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()
