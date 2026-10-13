"""Approval-gated, resumable TG-4B.2C held-out execution runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tensor_game_heldout_execution import (
    ConfigurationFactory, TrialStore, build_trial_schedule, infrastructure_failure_record,
    record_from_episode,
)
from .tensor_game_protocol_addendum import validate_protocol_addendum


def _target_payload(value):
    return tuple(tuple(tuple(line) for line in plane) for plane in value)


def run_heldout_evaluation(
    *, plan_path: Path, addendum_path: Path, approval_path: Path,
    experience_path: Path, output_path: Path, resume: bool = False,
    progress_every: int = 10, device: str = "cpu",
) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress interval must be positive")
    if output_path.exists() and not resume:
        raise FileExistsError(output_path)
    if resume and not output_path.exists():
        raise FileNotFoundError("resume output does not exist")
    readiness = validate_protocol_addendum(
        plan_path, addendum_path, approval_path=approval_path,
        require_approved=True, load_models=True,
    )
    if not readiness["execution_ready"]:
        raise PermissionError("protocol is not execution ready")
    plan = json.loads(plan_path.read_text())
    addendum = json.loads(addendum_path.read_text())
    factory = ConfigurationFactory(plan, addendum, device)
    schedule = build_trial_schedule(factory.configurations, plan)

    # Held-out tensors are opened only inside this approval-gated execution path.
    manifest = json.loads((experience_path / "manifest.json").read_text())
    records = manifest["metadata"]["target_records"]
    targets = sorted((row for row in records if row["split"] == "test"), key=lambda row: row["target_digest"])
    if len(targets) != 30:
        raise ValueError("persisted held-out partition does not contain 30 targets")
    target_by_slot = {f"test-target-digest-order-{index:03d}": row for index, row in enumerate(targets)}

    checkpoint_digests = readiness["checkpoint_digests"]
    store = TrialStore(output_path, addendum["content_digest"], checkpoint_digests)
    store.initialize()
    store.recover_progress(schedule)
    completed = store.completed_trial_ids()
    executed = 0
    for trial in schedule:
        if trial.trial_id in completed:
            continue
        target_record = target_by_slot[trial.target_group_id]
        try:
            episode, decisions = factory.run_episode(
                trial.configuration_id, _target_payload(target_record["target"]),
                max_moves=target_record["construction_length"], seed=trial.seed,
                experiment_id="tg4b2c-heldout-v1",
            )
            record = record_from_episode(trial, episode, decisions)
            store.write(record)
        except KeyboardInterrupt as error:
            store.write(infrastructure_failure_record(trial, error, interrupted=True))
            store.update_progress_manifest(schedule)
            raise
        except Exception as error:
            store.write(infrastructure_failure_record(trial, error))
        executed += 1
        if executed % progress_every == 0:
            progress = store.update_progress_manifest(schedule)
            print(json.dumps(progress, sort_keys=True), flush=True)
    progress = store.update_progress_manifest(schedule)
    return {**progress, "executed_this_invocation": executed, "output": str(output_path)}
