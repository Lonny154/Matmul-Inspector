"""Controlled TG-4B.2B offline policy/value learning ablations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

from .tensor_game_experience import REWARD_CONVENTION, load_buffer
from .tensor_game_learning import load_checkpoint, resolve_device
from .tensor_game_offline_learning import (
    VALUE_SEMANTICS,
    ExperienceDataset,
    OfflineTrainingConfig,
    _move_tensors,
    collate_experience,
    configure_trainable,
    experience_digest,
    sampled_support_log_probabilities,
)

ABLATION_VERSION = 1
PLAN_VERSION = 1
FROZEN_VALUE_SEMANTICS = "tg3b_predicted_completion_length"
CANDIDATES = {
    "policy_only_natural": {"mode": "policy_only", "balanced": False, "strict_torso_freeze": True},
    "value_only_natural": {"mode": "value_only", "balanced": False, "strict_torso_freeze": True},
    "value_only_balanced": {"mode": "value_only", "balanced": True, "strict_torso_freeze": True},
    "joint_natural": {"mode": "joint", "balanced": False, "strict_torso_freeze": False},
    "joint_balanced": {"mode": "joint", "balanced": True, "strict_torso_freeze": False},
}


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("TG-4B.2B ablations require optional PyTorch") from error
    return torch


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_evidence_digests(experience: Path, checkpoint: Path) -> dict[str, Any]:
    source_manifest = json.loads((checkpoint / "experiment.json").read_text())
    checkpoint_file = checkpoint / source_manifest["checkpoint"]
    return {
        "experience_digest": experience_digest(experience),
        "experience_files": {
            name: _file_sha256(experience / name)
            for name in ("manifest.json", "episodes.json", "decisions.json", "replay_validation.json", "dataset_audit.json")
        },
        "source_checkpoint_file": source_manifest["checkpoint"],
        "source_checkpoint_sha256": _file_sha256(checkpoint_file),
        "source_manifest_sha256": _file_sha256(checkpoint / "experiment.json"),
    }


def integrity_audit(experience: Path) -> dict[str, Any]:
    """Validate all source evidence while computing metrics only for train/validation."""
    buffer = load_buffer(experience)
    manifest = json.loads((experience / "manifest.json").read_text())
    source_audit = json.loads((experience / "dataset_audit.json").read_text())
    metadata = manifest.get("metadata", {})
    assignments = metadata.get("target_splits")
    if not assignments or source_audit["state_diversity"]["cross_split_residual_overlap"]:
        raise ValueError("source experience lacks isolated persisted target partitions")
    target_records = {row["target_digest"]: row for row in metadata["target_records"]}
    episode_by_id = {episode["episode_id"]: episode for episode in buffer.episodes}
    state_splits: dict[str, set[str]] = defaultdict(set)
    split_stats = {}
    for split in ("train", "validation"):
        episodes = [episode for episode in buffer.episodes if assignments[episode["initial_target_digest"]] == split]
        episode_ids = {episode["episode_id"] for episode in episodes}
        rows = [row for row in buffer.decisions.values() if row["episode_id"] in episode_ids]
        successful_targets = {episode["initial_target_digest"] for episode in episodes if episode["outcome"] == "success"}
        outcomes = Counter(episode["outcome"] for episode in episodes)
        outcome_by_length: dict[str, Counter] = defaultdict(Counter)
        for episode in episodes:
            length = str(target_records[episode["initial_target_digest"]]["construction_length"])
            outcome_by_length[length][episode["outcome"]] += 1
        returns = [row["observed_return_target"] for row in rows if not row["outcome_censored"]]
        split_stats[split] = {
            "target_groups": len({episode["initial_target_digest"] for episode in episodes}),
            "successful_target_groups": len(successful_targets),
            "episodes": len(episodes),
            "eligible_policy_decisions": len(rows),
            "eligible_value_decisions": sum(not row["outcome_censored"] for row in rows),
            "unique_residual_states": len({row["state_digest"] for row in rows}),
            "repeated_state_decisions": len(rows) - len({row["state_digest"] for row in rows}),
            "episode_outcomes": dict(outcomes),
            "outcomes_by_construction_length": {key: dict(value) for key, value in sorted(outcome_by_length.items())},
            "return_targets": {
                "count": len(returns), "minimum": min(returns), "maximum": max(returns),
                "mean": sum(returns) / len(returns), "frequency": dict(sorted(Counter(returns).items())),
            },
        }
        for row in rows:
            state_splits[row["state_digest"]].add(split)
            if not row["model_provenance"].get("target_id") or row["return_target_semantics"] != REWARD_CONVENTION:
                raise ValueError("experience provenance or return semantics are incompatible")
    if any(len(splits) > 1 for splits in state_splits.values()):
        raise ValueError("train/validation residual leakage detected")
    if not buffer.validate()["valid"]:
        raise ValueError("source replay validation failed")
    if any(source_audit["training_readiness"].get(key) for key in ("invalid_policy_targets", "invalid_action_records", "invalid_value_targets")):
        raise ValueError("source learning targets failed the TG-4B.2A audit")
    expected = {
        "episodes": 800, "decisions": 2000, "unique_residuals": 967,
        "train_targets": 140, "validation_targets": 30, "test_targets": 30,
        "unique_train_residuals": 687,
    }
    observed = {
        "episodes": len(buffer.episodes), "decisions": len(buffer.decisions),
        "unique_residuals": source_audit["state_diversity"]["unique_residuals"],
        "train_targets": Counter(assignments.values())["train"],
        "validation_targets": Counter(assignments.values())["validation"],
        "test_targets": Counter(assignments.values())["test"],
        "unique_train_residuals": split_stats["train"]["unique_residual_states"],
    }
    if observed != expected:
        raise ValueError(f"source experience does not match frozen TG-4B.2A evidence: {observed}")
    return {
        "valid": True,
        "source_episode_count": len(buffer.episodes),
        "source_decision_count": len(buffer.decisions),
        "persisted_target_counts": dict(Counter(assignments.values())),
        "cross_split_target_overlap": 0,
        "cross_split_residual_overlap": 0,
        "held_out_metrics_accessed": False,
        "splits_used_for_metrics": ["train", "validation"],
        "splits": split_stats,
        "frozen_expected_properties": expected,
    }


def outcome_balance_weights(dataset: ExperienceDataset, cap_ratio: float = 10.0) -> tuple[list[float], dict[str, Any]]:
    if cap_ratio < 1 or not math.isfinite(cap_ratio):
        raise ValueError("outcome weight cap must be finite and at least one")
    counts = Counter(dataset.episode_by_id[row["episode_id"]]["outcome"] for row in dataset.rows)
    if not counts:
        raise ValueError("cannot balance an empty dataset")
    maximum = max(counts.values())
    raw_by_outcome = {outcome: min(cap_ratio, maximum / count) for outcome, count in counts.items()}
    weights = [raw_by_outcome[dataset.episode_by_id[row["episode_id"]]["outcome"]] for row in dataset.rows]
    mean = sum(weights) / len(weights)
    weights = [value / mean for value in weights]
    effective = Counter()
    for row, weight in zip(dataset.rows, weights):
        effective[dataset.episode_by_id[row["episode_id"]]["outcome"]] += weight
    return weights, {
        "method": "deterministic_inverse_decision_outcome_frequency",
        "cap_ratio": cap_ratio,
        "raw_decision_frequencies": dict(counts),
        "raw_outcome_weight_ratio": raw_by_outcome,
        "normalized_mean_weight": sum(weights) / len(weights),
        "effective_weighted_frequencies": dict(effective),
        "minimum_weight": min(weights),
        "maximum_weight": max(weights),
        "repeated_states_preserved": True,
        "validation_weighting": "none",
    }


class WeightedExperienceDataset:
    def __init__(self, dataset: ExperienceDataset, weights: list[float]):
        if len(dataset) != len(weights) or any(value <= 0 or not math.isfinite(value) for value in weights):
            raise ValueError("invalid deterministic sample weights")
        self.dataset = dataset
        self.weights = weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["sample_weight"] = _torch().tensor(self.weights[index], dtype=_torch().float32)
        return item


def _aggregate_metrics(records: list[dict[str, float]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0}
    value_records = [row for row in records if row["value_valid"]]
    predictions = [row["prediction"] for row in value_records]
    targets = [row["target"] for row in value_records]
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return {
        "examples": len(records),
        "independent_target_groups": len({row["target_id"] for row in records}),
        "policy_cross_entropy": sum(row["cross_entropy"] for row in records) / len(records),
        "policy_kl": sum(row["kl"] for row in records) / len(records),
        "top1_agreement": sum(row["top1"] for row in records) / len(records),
        "policy_entropy": sum(row["entropy"] for row in records) / len(records),
        "value_examples": len(value_records),
        "value_mae": sum(abs(value) for value in errors) / len(errors) if errors else None,
        "value_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
        "value_prediction_mean": sum(predictions) / len(predictions) if predictions else None,
        "value_prediction_min": min(predictions) if predictions else None,
        "value_prediction_max": max(predictions) if predictions else None,
        "value_target_mean": sum(targets) / len(targets) if targets else None,
        "value_bias": sum(errors) / len(errors) if errors else None,
    }


def evaluate_validation(model, dataset: ExperienceDataset, model_config, batch_size: int, device) -> dict[str, Any]:
    torch = _torch()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_experience)
    records = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_tensors(batch, device)
            logits, values = model(batch["residual"])
            log_policy = sampled_support_log_probabilities(logits, batch["actions"], batch["action_mask"], model_config.shape)
            target_policy = batch["search_policy"].detach()
            cross_entropy = -(target_policy * torch.where(batch["action_mask"], log_policy, 0.0)).sum(1)
            entropy = -(target_policy * torch.where(target_policy > 0, torch.log(target_policy), 0.0)).sum(1)
            predicted_action = log_policy.masked_fill(~batch["action_mask"], -torch.inf).argmax(1)
            target_action = target_policy.argmax(1)
            for index in range(len(values)):
                records.append({
                    "cross_entropy": float(cross_entropy[index]),
                    "kl": float(cross_entropy[index] - entropy[index]),
                    "entropy": float(entropy[index]),
                    "top1": float(predicted_action[index] == target_action[index]),
                    "prediction": float(values[index]),
                    "target": float(batch["value_target"][index]),
                    "value_valid": bool(batch["value_valid"][index]),
                    "target_id": batch["target_id"][index],
                    "outcome": batch["outcome"][index],
                    "construction_length": batch["construction_length"][index],
                })
    by_outcome = {
        outcome: _aggregate_metrics([row for row in records if row["outcome"] == outcome])
        for outcome in sorted({row["outcome"] for row in records})
    }
    by_length = {
        str(length): _aggregate_metrics([row for row in records if row["construction_length"] == length])
        for length in sorted({row["construction_length"] for row in records if row["construction_length"] is not None})
    }
    return {"overall": _aggregate_metrics(records), "by_terminal_outcome": by_outcome, "by_construction_length": by_length}


def _validation_objective(mode: str, metrics: dict[str, Any], value_coefficient: float) -> tuple[str, float]:
    overall = metrics["overall"]
    if mode == "policy_only":
        return "sampled_support_policy_cross_entropy", overall["policy_cross_entropy"]
    if mode == "value_only":
        return "uncensored_return_mean_squared_error", overall["value_rmse"] ** 2
    return "policy_cross_entropy_plus_value_coefficient_times_return_mse", (
        overall["policy_cross_entropy"] + value_coefficient * overall["value_rmse"] ** 2
    )


def _candidate_value_semantics(mode: str) -> str:
    return FROZEN_VALUE_SEMANTICS if mode == "policy_only" else VALUE_SEMANTICS


def train_candidate(
    name: str,
    experience: Path,
    source_checkpoint: Path,
    output: Path,
    training: OfflineTrainingConfig,
    *,
    balanced: bool,
    weight_cap: float = 10.0,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if name not in CANDIDATES:
        raise ValueError("unknown ablation candidate")
    torch = _torch()
    device = resolve_device(training.device)
    model, model_config, source_manifest = load_checkpoint(source_checkpoint, training.device)
    train = ExperienceDataset(experience, "train")
    validation = ExperienceDataset(experience, "validation")
    if train.split_report["assignment_source"] != "persisted_precollection_target_partitions":
        raise ValueError("training requires persisted target partitions")
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)
    torch.use_deterministic_algorithms(True)
    trainable = configure_trainable(model, training.mode, training.strict_torso_freeze)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_state_digest = hashlib.sha256(b"".join(
        value.detach().cpu().contiguous().numpy().tobytes() for _, value in sorted(model.state_dict().items())
    )).hexdigest()
    if balanced:
        weights, balance_report = outcome_balance_weights(train, weight_cap)
    else:
        weights = [1.0] * len(train)
        counts = Counter(train.episode_by_id[row["episode_id"]]["outcome"] for row in train.rows)
        balance_report = {
            "method": "natural_decision_frequencies", "cap_ratio": None,
            "raw_decision_frequencies": dict(counts), "effective_weighted_frequencies": dict(counts),
            "minimum_weight": 1.0, "maximum_weight": 1.0,
            "repeated_states_preserved": True, "validation_weighting": "none",
        }
    weighted_train = WeightedExperienceDataset(train, weights)
    generator = torch.Generator().manual_seed(training.seed)
    loader = torch.utils.data.DataLoader(
        weighted_train, batch_size=training.batch_size, shuffle=True,
        generator=generator, collate_fn=collate_experience,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training.learning_rate,
    )
    output.mkdir(parents=True)
    checkpoint = output / "best_model_state.pt"
    policy_coefficient = training.policy_coefficient if training.mode != "value_only" else 0.0
    value_coefficient = training.value_coefficient if training.mode != "policy_only" else 0.0
    best = math.inf
    best_epoch = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, training.epochs + 1):
        model.train()
        totals = Counter()
        for batch in loader:
            batch = _move_tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(batch["residual"])
            log_policy = sampled_support_log_probabilities(logits, batch["actions"], batch["action_mask"], model_config.shape)
            targets = batch["search_policy"].detach()
            weights_tensor = batch["sample_weight"].detach()
            policy_examples = -(targets * torch.where(batch["action_mask"], log_policy, 0.0)).sum(1)
            policy_loss = (policy_examples * weights_tensor).sum() / weights_tensor.sum()
            valid = batch["value_valid"]
            value_errors = (values[valid] - batch["value_target"][valid].detach()) ** 2
            value_weights = weights_tensor[valid]
            value_loss = (value_errors * value_weights).sum() / value_weights.sum() if torch.any(valid) else values.sum() * 0
            loss = policy_coefficient * policy_loss + value_coefficient * value_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad), training.gradient_clip
            )
            optimizer.step()
            count = len(values)
            totals["examples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["policy_loss"] += float(policy_loss.detach()) * count
            totals["value_loss"] += float(value_loss.detach()) * count
            totals["gradient_norm"] += float(gradient_norm) * count
        validation_metrics = evaluate_validation(model, validation, model_config, training.batch_size, device)
        objective_name, objective = _validation_objective(training.mode, validation_metrics, training.value_coefficient)
        history.append({
            "epoch": epoch,
            "train": {key: totals[key] / totals["examples"] for key in ("loss", "policy_loss", "value_loss", "gradient_norm")},
            "validation_objective": objective,
            "validation": validation_metrics,
        })
        if objective < best:
            best = objective
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= training.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    final_validation = evaluate_validation(model, validation, model_config, training.batch_size, device)
    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    frozen_names = [name for name in initial_state if name not in trainable]
    changed_frozen = [name for name in frozen_names if not torch.equal(initial_state[name], final_state[name])]
    changed_trainable = [name for name in trainable if not torch.equal(initial_state[name], final_state[name])]
    source_checkpoint_file = source_checkpoint / source_manifest["checkpoint"]
    manifest = {
        "ablation_version": ABLATION_VERSION,
        "candidate": name,
        "model_config": asdict(model_config),
        "training_config": asdict(training),
        "sampling_treatment": "outcome_balanced" if balanced else "natural_frequencies",
        "balance_report": balance_report,
        "trainable_parameters": trainable,
        "strict_torso_freeze": training.strict_torso_freeze,
        "parameter_update_verification": {
            "frozen_parameter_count": len(frozen_names),
            "frozen_parameters_unchanged": not changed_frozen,
            "changed_frozen_parameters": changed_frozen,
            "changed_trainable_parameter_count": len(changed_trainable),
        },
        "value_head_semantics": _candidate_value_semantics(training.mode),
        "policy_target_semantics": "conditional_mcts_visit_distribution_over_sampled_support",
        "source_checkpoint_sha256": _file_sha256(source_checkpoint_file),
        "source_initial_state_digest": initial_state_digest,
        "experience_digest": experience_digest(experience),
        "splits_accessed": ["train", "validation"],
        "held_out_test_accessed": False,
        "optimizer": "AdamW",
        "validation_objective": objective_name,
        "best_epoch": best_epoch,
        "best_validation_objective": best,
        "history": history,
        "validation_metrics": final_validation,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "training_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_ablation_checkpoint(path: Path, expected_value_semantics: str | None = None, device: str = "cpu"):
    torch = _torch()
    from .tensor_game_learning import PolicyValueConfig, create_policy_value_model
    manifest = json.loads((path / "experiment.json").read_text())
    if manifest.get("ablation_version") != ABLATION_VERSION:
        raise ValueError("unsupported ablation checkpoint")
    if expected_value_semantics is not None and manifest["value_head_semantics"] != expected_value_semantics:
        raise ValueError("checkpoint value semantics are incompatible with the requested adapter")
    value = manifest["model_config"]
    config = PolicyValueConfig(
        shape=tuple(value["shape"]), hidden_dim=value["hidden_dim"],
        coefficients=tuple(value["coefficients"]), input_scale=value["input_scale"],
    )
    resolved = resolve_device(device)
    model = create_policy_value_model(config).to(resolved)
    model.load_state_dict(torch.load(path / manifest["checkpoint"], map_location=resolved, weights_only=True))
    model.eval()
    return model, config, manifest


def create_evaluation_plan(candidate_manifests: dict[str, dict[str, Any]], source_digests: dict[str, Any]) -> dict[str, Any]:
    checkpoints = {
        "frozen_tg3b": {
            "checkpoint_sha256": source_digests["source_checkpoint_sha256"],
            "relative_path": "frozen_source/best_model_state.pt",
            "value_semantics": FROZEN_VALUE_SEMANTICS,
            "value_adapter": "completion_length_to_leaf_value_with_terminal_overrides",
        }
    }
    for name, manifest in candidate_manifests.items():
        semantics = manifest["value_head_semantics"]
        checkpoints[name] = {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "relative_path": f"candidates/{name}/{manifest.get('checkpoint', 'best_model_state.pt')}",
            "value_semantics": semantics,
            "value_adapter": (
                "direct_return_clamped_to_terminal_bounds_with_exact_terminal_overrides"
                if semantics == VALUE_SEMANTICS else
                "completion_length_to_leaf_value_with_terminal_overrides"
            ),
        }
    return {
        "evaluation_plan_version": PLAN_VERSION,
        "status": "frozen_before_held_out_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_results_present": False,
        "checkpoint_comparisons": checkpoints,
        "target_selection": {
            "partition": "persisted_test_targets_only",
            "procedure": "all 30 held-out target groups in deterministic target-digest order",
            "training_or_selection_access": "prohibited",
        },
        "search_ablations": [
            "residual_informed_policy_plus_heuristic_value",
            "residual_informed_policy_plus_constant_max_moves_value",
            "residual_informed_policy_plus_checkpoint_value",
            "learned_policy_plus_heuristic_value",
            "learned_policy_plus_checkpoint_value",
        ],
        "exact_success_criterion": "zero_residual_and_independent_exact_integer_reconstruction",
        "trial_seeds": [3, 11, 19, 27, 35, 43],
        "mcts": {
            "simulations": 200, "proposals": 12, "c_puct": 1.5,
            "node_limit": 200, "episode_wall_time_seconds": 5.0,
            "move_limit_policy": "construction_length_upper_bound",
        },
        "outcome_metrics": [
            "exact_success_rate_by_independent_target", "verified_decomposition_length",
            "simulations_to_first_success", "episode_runtime", "truncation_rate",
        ],
        "failure_handling": {
            "truncation": "bounded_search_failure_not_mathematical_nonexistence",
            "interruption": "censored_and_excluded_from_success_denominator_sensitivity_reported",
            "invalid_reconstruction": "hard_failure_and_experiment_stop",
        },
        "statistical_analysis": {
            "unit": "originating_target_group",
            "pairing": "same_targets_seeds_and_budgets_across_configurations",
            "intervals": "target_cluster_bootstrap_with_exact_counts_also_reported",
            "multiple_comparisons": "report_all_preregistered_contrasts_without_selective_omission",
        },
        "limitations": [
            "offline validation does not establish discovery improvement",
            "return scale can dominate PUCT and requires adapter-specific controls",
            "construction length is an upper bound rather than proven tensor rank",
        ],
    }


def validate_evaluation_plan(plan: dict[str, Any]) -> None:
    if plan.get("evaluation_plan_version") != PLAN_VERSION or plan.get("status") != "frozen_before_held_out_evaluation":
        raise ValueError("unsupported or unfrozen evaluation plan")
    if plan.get("held_out_results_present") or plan["target_selection"]["partition"] != "persisted_test_targets_only":
        raise ValueError("evaluation plan improperly contains held-out execution")
    for checkpoint in plan["checkpoint_comparisons"].values():
        expected = (
            "direct_return_clamped_to_terminal_bounds_with_exact_terminal_overrides"
            if checkpoint["value_semantics"] == VALUE_SEMANTICS else
            "completion_length_to_leaf_value_with_terminal_overrides"
        )
        if checkpoint["value_adapter"] != expected:
            raise ValueError("evaluation plan uses an incompatible value adapter")


def run_ablation(
    experience: Path,
    source_checkpoint: Path,
    output: Path,
    training: OfflineTrainingConfig,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    audit = integrity_audit(experience)
    digests = source_evidence_digests(experience, source_checkpoint)
    output.mkdir(parents=True)
    smoke_config = replace(training, epochs=1, patience=1)
    smoke = train_candidate(
        "joint_natural", experience, source_checkpoint, output / "smoke_check",
        smoke_config, balanced=False,
    )
    candidates = {}
    for name, definition in CANDIDATES.items():
        config = replace(
            training,
            mode=definition["mode"],
            strict_torso_freeze=definition["strict_torso_freeze"],
        )
        candidates[name] = train_candidate(
            name, experience, source_checkpoint, output / "candidates" / name,
            config, balanced=definition["balanced"],
        )
        load_ablation_checkpoint(
            output / "candidates" / name,
            candidates[name]["value_head_semantics"],
            training.device,
        )
    frozen_model, frozen_config, _ = load_checkpoint(source_checkpoint, training.device)
    validation = ExperienceDataset(experience, "validation")
    frozen_metrics = evaluate_validation(
        frozen_model, validation, frozen_config, training.batch_size, resolve_device(training.device)
    )
    comparison = {
        "frozen_tg3b": {
            "value_semantics": FROZEN_VALUE_SEMANTICS,
            "value_metrics_interpretation": "raw_cross_semantics_diagnostic_not_calibrated_return_prediction",
            "validation_metrics": frozen_metrics,
        }
    }
    for name, manifest in candidates.items():
        comparison[name] = {
            "value_semantics": manifest["value_head_semantics"],
            "value_metrics_interpretation": (
                "direct_return_prediction" if manifest["value_head_semantics"] == VALUE_SEMANTICS
                else "raw_cross_semantics_diagnostic_not_calibrated_return_prediction"
            ),
            "best_epoch": manifest["best_epoch"],
            "validation_objective": manifest["best_validation_objective"],
            "validation_metrics": manifest["validation_metrics"],
        }
    plan = create_evaluation_plan(candidates, digests)
    validate_evaluation_plan(plan)
    summary = {
        "ablation_version": ABLATION_VERSION,
        "source_digests": digests,
        "integrity_audit": audit,
        "resolved_training_config": asdict(training),
        "models": ["frozen_tg3b", *CANDIDATES],
        "smoke_check": {
            "passed": True, "best_epoch": smoke["best_epoch"],
            "checkpoint_sha256": smoke["checkpoint_sha256"],
        },
        "candidate_summaries": {
            name: {
                "best_epoch": manifest["best_epoch"],
                "best_validation_objective": manifest["best_validation_objective"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "training_seconds": manifest["training_seconds"],
                "value_head_semantics": manifest["value_head_semantics"],
            }
            for name, manifest in candidates.items()
        },
        "held_out_test_accessed": False,
    }
    experiment = {
        "ablation_version": ABLATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experience": str(experience),
        "source_checkpoint": str(source_checkpoint),
        "output": str(output),
        "candidate_definitions": CANDIDATES,
        "artifacts": {
            "training_summary": "training_summary.json",
            "validation_comparison": "validation_comparison.json",
            "evaluation_plan": "evaluation_plan.json",
            "report": "report.md",
        },
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "validation_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (output / "evaluation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _write_report(output / "report.md", summary, comparison)
    return summary


def _write_report(path: Path, summary: dict[str, Any], comparison: dict[str, Any]) -> None:
    audit = summary["integrity_audit"]
    lines = [
        "# TG-4B.2B controlled offline learning ablation", "",
        "All candidates start from the same frozen TG-3B checkpoint. Model selection uses validation targets only. Held-out test metrics and discovery search were not run.", "",
        "## Integrity", "",
        f"Replay and leakage audit valid: **{audit['valid']}**. Train decisions: {audit['splits']['train']['eligible_policy_decisions']}. Validation decisions: {audit['splits']['validation']['eligible_policy_decisions']}.",
        f"Unique train/validation residuals: {audit['splits']['train']['unique_residual_states']} / {audit['splits']['validation']['unique_residual_states']}.", "",
        "## Validation comparison", "",
        "| Model | Best epoch | Policy CE | Policy KL | Top-1 | Value MAE | Value RMSE | Value bias |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in comparison.items():
        metrics = record["validation_metrics"]["overall"]
        lines.append(
            f"| `{name}` | {record.get('best_epoch', 'frozen')} | {metrics['policy_cross_entropy']:.6f} | "
            f"{metrics['policy_kl']:.6f} | {metrics['top1_agreement']:.4f} | {metrics['value_mae']:.6f} | "
            f"{metrics['value_rmse']:.6f} | {metrics['value_bias']:.6f} |"
        )
    lines.extend([
        "", "The frozen and policy-only value heads retain completion-length semantics. Their return-error columns are raw cross-semantics diagnostics, not calibrated return metrics.",
        "Offline validation losses do not establish improved tensor-decomposition discovery.", "",
        "## Outcome balancing", "",
        "Balanced candidates use deterministic inverse decision-outcome-frequency weights capped at a 10:1 ratio. Validation remains unweighted. Repeated states are preserved and remain correlated evidence.", "",
        "## Frozen next step", "",
        "`evaluation_plan.json` preregisters the held-out search comparison. No held-out target was evaluated during this milestone.", "",
    ])
    path.write_text("\n".join(lines))
