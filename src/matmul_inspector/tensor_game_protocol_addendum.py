"""Validation for the TG-4B.2C pre-execution protocol addendum.

This module intentionally does not load held-out target tensors or run search.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from .tensor_game_offline_ablation import FROZEN_VALUE_SEMANTICS, VALUE_SEMANTICS

ADDENDUM_SCHEMA_VERSION = 1
SUPPORTED_VALUE_SEMANTICS = {FROZEN_VALUE_SEMANTICS, VALUE_SEMANTICS}
SUPPORTED_POLICIES = {"residual_informed", "checkpoint_policy"}
SUPPORTED_VALUES = {"nonzero_heuristic", "constant_max_moves", "checkpoint_value"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "content_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_addendum(path: Path, document: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(path)
    payload = dict(document)
    payload["content_digest"] = content_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["content_digest"]


def expected_trial_count(plan: dict[str, Any], addendum: dict[str, Any]) -> int:
    target_count = addendum["trial_accounting"]["held_out_target_groups"]
    seeds = plan["trial_seeds"]
    configurations = addendum["model_search_matrix"]
    if target_count <= 0 or not seeds or not configurations:
        raise ValueError("trial accounting inputs must be nonempty")
    return target_count * len(seeds) * len(configurations)


def _validate_bootstrap(value: dict[str, Any]) -> None:
    expected = {
        "confidence_level": 0.95,
        "resamples": 10_000,
        "seed": 20_261_013,
        "resampling_unit": "originating_tensor_target_group",
        "keep_all_trial_seeds_together": True,
        "comparison_statistic": "paired_target_level_success_rate_difference",
        "confidence_interval_method": "percentile_cluster_bootstrap",
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"unsupported grouped bootstrap setting: {key}")


def _validate_matrix(plan: dict[str, Any], addendum: dict[str, Any]) -> None:
    rows = addendum["model_search_matrix"]
    identifiers = [row["configuration_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate configuration IDs")
    checkpoints = addendum["checkpoint_path_resolutions"]
    checkpoint_names = set(checkpoints)
    represented_ablations = set()
    for row in rows:
        missing = {
            "configuration_id", "frozen_plan_ablation", "policy_source", "value_source",
            "value_head_semantics", "action_proposal_method", "leaf_evaluation_method",
            "search_parameters", "role",
        } - set(row)
        if missing:
            raise ValueError(f"missing configuration mappings: {sorted(missing)}")
        if row["frozen_plan_ablation"] not in plan["search_ablations"]:
            raise ValueError("configuration uses an unknown frozen-plan ablation")
        represented_ablations.add(row["frozen_plan_ablation"])
        if row["action_proposal_method"] not in SUPPORTED_POLICIES or row["leaf_evaluation_method"] not in SUPPORTED_VALUES:
            raise ValueError("configuration uses an unsupported policy or value adapter")
        if row["value_head_semantics"] not in SUPPORTED_VALUE_SEMANTICS | {"not_applicable"}:
            raise ValueError("unsupported value semantics")
        for source_key in ("policy_source", "value_source"):
            source = row[source_key]
            if source.startswith("checkpoint:") and source.split(":", 1)[1] not in checkpoint_names:
                raise ValueError("configuration references an unresolved checkpoint")
        if row["search_parameters"] != plan["mcts"]:
            raise ValueError("configuration changes frozen MCTS parameters")
    if represented_ablations != set(plan["search_ablations"]):
        raise ValueError("missing configuration mappings for frozen search ablations")


def validate_protocol_addendum(
    plan_path: Path,
    addendum_path: Path,
    *,
    output_path: Path | None = None,
    require_approved: bool = False,
    load_models: bool = True,
    approval_path: Path | None = None,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    addendum = json.loads(addendum_path.read_text())
    if addendum.get("schema_version") != ADDENDUM_SCHEMA_VERSION:
        raise ValueError("unsupported protocol addendum schema")
    if file_sha256(plan_path) != addendum["original_plan"]["sha256"]:
        raise ValueError("frozen-plan digest mismatch")
    if content_digest(addendum) != addendum.get("content_digest"):
        raise ValueError("addendum content digest mismatch")
    if addendum["original_plan"]["path"] != str(plan_path):
        raise ValueError("addendum points to a different frozen plan")
    if plan.get("status") != "frozen_before_held_out_evaluation" or plan.get("held_out_results_present"):
        raise ValueError("frozen plan indicates held-out evaluation has begun")
    if addendum.get("held_out_evaluation_occurred") is not False:
        raise ValueError("addendum does not confirm pre-execution state")
    _validate_matrix(plan, addendum)
    _validate_bootstrap(addendum["bootstrap_analysis"])
    wall = addendum["wall_time_interpretation"]
    if wall.get("scope") != "entire_multi_decision_episode" or not wall.get("inner_search_uses_remaining_episode_time"):
        raise ValueError("wall-time semantics differ from the established episode runner")
    observed_checkpoints = {}
    model_shapes = {}
    for name, resolution in addendum["checkpoint_path_resolutions"].items():
        actual = Path(resolution["resolved_actual_path"])
        if not actual.is_file():
            raise FileNotFoundError(f"missing checkpoint resolution: {name}")
        observed = file_sha256(actual)
        if observed != resolution["expected_sha256"] or observed != resolution["observed_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {name}")
        semantics = resolution["value_head_semantics"]
        if semantics not in SUPPORTED_VALUE_SEMANTICS:
            raise ValueError(f"unsupported value semantics: {name}")
        planned = plan["checkpoint_comparisons"][name]
        if planned["checkpoint_sha256"] != observed or planned["value_semantics"] != semantics:
            raise ValueError(f"checkpoint is incompatible with frozen plan: {name}")
        observed_checkpoints[name] = observed
        if load_models:
            checkpoint_directory = actual.parent
            if name == "frozen_tg3b":
                from .tensor_game_learning import load_checkpoint

                _, config, _ = load_checkpoint(checkpoint_directory, "cpu")
            else:
                from .tensor_game_offline_ablation import load_ablation_checkpoint

                _, config, _ = load_ablation_checkpoint(checkpoint_directory, semantics, "cpu")
            model_shapes[name] = list(config.shape)
    if model_shapes and len({tuple(shape) for shape in model_shapes.values()}) != 1:
        raise ValueError("checkpoint model architectures are incompatible")
    trials = expected_trial_count(plan, addendum)
    if trials != addendum["trial_accounting"]["expected_trial_count"]:
        raise ValueError("expected trial-count mismatch")
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"evaluation output path already exists: {output_path}")
    approval = None
    if approval_path is not None:
        approval = json.loads(approval_path.read_text())
        if content_digest(approval) != approval.get("content_digest"):
            raise ValueError("approval artifact content digest mismatch")
        expected_approval = {
            "approval_status": "approved",
            "explicit_user_approval": True,
            "frozen_plan_sha256": file_sha256(plan_path),
            "addendum_content_digest": addendum["content_digest"],
            "approved_configuration_count": len(addendum["model_search_matrix"]),
            "approved_trial_count": trials,
            "episode_wall_time_seconds": plan["mcts"]["episode_wall_time_seconds"],
        }
        for key, value in expected_approval.items():
            if approval.get(key) != value:
                raise ValueError(f"approval artifact mismatch: {key}")
        if not approval.get("frozen_documents_unchanged"):
            raise ValueError("approval does not preserve frozen documents")
    wiring_valid = fixture_valid = schedule_valid = persistence_valid = False
    isolation = {}
    fixture_outcomes = {}
    if load_models:
        from .tensor_game import RankOneAction, outer_product
        from .tensor_game_heldout_execution import (
            ConfigurationFactory, TrialRecord, TrialStore, build_trial_schedule,
            validate_ablation_isolation,
        )

        factory = ConfigurationFactory(plan, addendum, "cpu")
        wiring_valid = len(factory.configurations) == len(addendum["model_search_matrix"]) == 10
        isolation = validate_ablation_isolation(factory.configurations)
        target = outer_product(RankOneAction((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)))
        for identifier in factory.configurations:
            episode, _ = factory.run_fixture_episode(identifier, target)
            fixture_outcomes[identifier] = episode["outcome"]
        fixture_valid = len(fixture_outcomes) == 10
        schedule = build_trial_schedule(factory.configurations, plan, addendum["trial_accounting"]["held_out_target_groups"])
        schedule_valid = len(schedule) == trials == len({item.trial_id for item in schedule})
        with tempfile.TemporaryDirectory() as directory:
            store = TrialStore(Path(directory), addendum["content_digest"], observed_checkpoints)
            store.initialize()
            sample = schedule[0]
            record = TrialRecord(
                trial_id=sample.trial_id, status="bounded_search_failure",
                termination_reason="synthetic_fixture", configuration_id=sample.configuration_id,
                target_group_id=sample.target_group_id, seed=sample.seed, attempted=True,
                exact_reconstruction=False, checkpoint_provenance=sample.checkpoint_provenance,
            )
            persistence_valid = store.write(record) == "recorded" and store.write(record) == "already_recorded"
            store.initialize()
            persistence_valid = persistence_valid and sample.trial_id in store.completed_trial_ids()
    review_approved = approval is not None
    if require_approved and not review_approved:
        raise PermissionError("execution rejected without a verified approval artifact")
    return {
        "valid": True,
        "protocol_valid": True,
        "configuration_wiring_valid": wiring_valid,
        "fixture_execution_valid": fixture_valid,
        "schedule_valid": schedule_valid,
        "persistence_valid": persistence_valid,
        "review_approved": review_approved,
        "approval_status": addendum["approval_status"],
        "approval_artifact": str(approval_path) if approval_path else None,
        "execution_ready": all((wiring_valid, fixture_valid, schedule_valid, persistence_valid, review_approved)),
        "held_out_targets_loaded": False,
        "held_out_evaluation_occurred": False,
        "configuration_count": len(addendum["model_search_matrix"]),
        "expected_trial_count": trials,
        "checkpoint_digests": observed_checkpoints,
        "checkpoint_model_shapes": model_shapes,
        "ablation_isolation": isolation,
        "fixture_outcomes": fixture_outcomes,
        "unresolved_ambiguities": [],
    }
