"""Execution foundation for the frozen TG-4B.2C protocol.

No function in this module discovers or loads held-out target tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .tensor_game_experience import CollectionConfig, play_episode
from .tensor_game_learning import COEFFICIENTS, load_checkpoint
from .tensor_game_neural_mcts import (
    ConstantValueEvaluator,
    DirectReturnValueEvaluator,
    NeuralActionPolicy,
    NeuralValueEvaluator,
)
from .tensor_game_offline_ablation import FROZEN_VALUE_SEMANTICS, VALUE_SEMANTICS, load_ablation_checkpoint
from .tensor_game_protocol_addendum import file_sha256
from .tensor_game_search import HeuristicValueEvaluator, ResidualInformedPolicy

COMPLETION_SEMANTICS = {"known_completion_length", FROZEN_VALUE_SEMANTICS}
DIRECT_RETURN_SEMANTICS = {VALUE_SEMANTICS}
TRIAL_RECORD_VERSION = 1


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutableSearchConfiguration:
    configuration_id: str
    policy_kind: str
    policy_checkpoint: str | None
    value_kind: str
    value_checkpoint: str | None
    value_semantics: str
    coefficients: tuple[int, ...]
    simulations: int
    proposals: int
    c_puct: float
    node_limit: int
    episode_wall_time_seconds: float
    move_limit_policy: str
    model_provenance: dict[str, Any]

    @property
    def collection_config(self) -> CollectionConfig:
        return CollectionConfig(
            simulations=self.simulations,
            max_actions=self.proposals,
            c_puct=self.c_puct,
            node_limit=self.node_limit,
            wall_time_seconds=self.episode_wall_time_seconds,
            coefficients=self.coefficients,
        )


@dataclass(frozen=True)
class ScheduledTrial:
    trial_id: str
    configuration_id: str
    target_group_id: str
    seed: int
    episode_configuration: dict[str, Any]
    checkpoint_provenance: dict[str, Any]


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    status: str
    termination_reason: str
    configuration_id: str
    target_group_id: str
    seed: int
    attempted: bool
    exact_reconstruction: bool
    checkpoint_provenance: dict[str, Any]
    result: dict[str, Any] | None = None
    exception_type: str | None = None
    diagnostic: str | None = None
    schema_version: int = TRIAL_RECORD_VERSION

    def __post_init__(self) -> None:
        allowed = {"completed_success", "bounded_search_failure", "episode_timeout", "infrastructure_failure", "interrupted"}
        if self.status not in allowed:
            raise ValueError("unknown trial status")
        if self.exact_reconstruction and self.status != "completed_success":
            raise ValueError("only completed-success records may claim exact reconstruction")
        if self.status == "infrastructure_failure" and not self.exception_type:
            raise ValueError("infrastructure failure requires an exception type")


class ConfigurationFactory:
    def __init__(self, plan: dict[str, Any], addendum: dict[str, Any], device: str = "cpu"):
        self.plan = plan
        self.addendum = addendum
        self.device = device
        self._models: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        self.configurations = self._parse_all()

    def _checkpoint(self, reference: str) -> str:
        if not reference.startswith("checkpoint:"):
            raise ValueError("checkpoint source must use an explicit checkpoint reference")
        name = reference.split(":", 1)[1]
        if name not in self.addendum["checkpoint_path_resolutions"]:
            raise ValueError(f"missing checkpoint reference: {name}")
        return name

    def _load_model(self, name: str):
        if name in self._models:
            return self._models[name]
        resolution = self.addendum["checkpoint_path_resolutions"][name]
        path = Path(resolution["resolved_actual_path"])
        observed = file_sha256(path)
        if observed != resolution["expected_sha256"] or observed != self.plan["checkpoint_comparisons"][name]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {name}")
        semantics = resolution["value_head_semantics"]
        if semantics not in COMPLETION_SEMANTICS | DIRECT_RETURN_SEMANTICS:
            raise ValueError(f"unsupported checkpoint semantics: {name}")
        loaded = (
            load_checkpoint(path.parent, self.device)
            if name == "frozen_tg3b"
            else load_ablation_checkpoint(path.parent, semantics, self.device)
        )
        if tuple(loaded[1].coefficients) != tuple(COEFFICIENTS):
            raise ValueError("checkpoint coefficient encoding is incompatible")
        self._models[name] = loaded
        return loaded

    def _parse_row(self, row: dict[str, Any]) -> ExecutableSearchConfiguration:
        required = {
            "configuration_id", "policy_source", "value_source", "value_head_semantics",
            "action_proposal_method", "leaf_evaluation_method", "search_parameters",
        }
        if required - set(row):
            raise ValueError(f"missing executable configuration fields: {sorted(required - set(row))}")
        search = row["search_parameters"]
        if search != self.plan["mcts"]:
            raise ValueError("configuration search parameters differ from frozen plan")
        if row["policy_source"] == "residual_informed":
            policy_kind, policy_checkpoint = "residual_informed", None
        else:
            policy_kind, policy_checkpoint = "neural", self._checkpoint(row["policy_source"])
        if row["action_proposal_method"] != ("residual_informed" if policy_kind == "residual_informed" else "checkpoint_policy"):
            raise ValueError("policy source and proposal method disagree")
        value_source = row["value_source"]
        if value_source in {"nonzero_heuristic", "constant_max_moves"}:
            value_kind, value_checkpoint = value_source, None
            if row["value_head_semantics"] != "not_applicable":
                raise ValueError("non-neural value source must not declare checkpoint semantics")
        else:
            value_checkpoint = self._checkpoint(value_source)
            semantics = self.addendum["checkpoint_path_resolutions"][value_checkpoint]["value_head_semantics"]
            if semantics != row["value_head_semantics"]:
                raise ValueError("value source and declared semantics disagree")
            value_kind = "completion_length" if semantics in COMPLETION_SEMANTICS else "direct_return"
        expected_leaf = "checkpoint_value" if value_checkpoint else value_kind
        if row["leaf_evaluation_method"] != expected_leaf:
            raise ValueError("value source and leaf evaluation method disagree")
        for checkpoint in filter(None, (policy_checkpoint, value_checkpoint)):
            _, model_config, _ = self._load_model(checkpoint)
            if tuple(model_config.shape) != (4, 4, 4):
                raise ValueError("checkpoint tensor shape is incompatible")
        provenance = {
            "configuration_id": row["configuration_id"],
            "policy_kind": policy_kind, "policy_checkpoint": policy_checkpoint,
            "value_kind": value_kind, "value_checkpoint": value_checkpoint,
            "value_semantics": row["value_head_semantics"],
            "checkpoint_digests": {
                name: self.addendum["checkpoint_path_resolutions"][name]["expected_sha256"]
                for name in sorted(set(filter(None, (policy_checkpoint, value_checkpoint))))
            },
        }
        return ExecutableSearchConfiguration(
            configuration_id=row["configuration_id"], policy_kind=policy_kind,
            policy_checkpoint=policy_checkpoint, value_kind=value_kind,
            value_checkpoint=value_checkpoint, value_semantics=row["value_head_semantics"],
            coefficients=tuple(COEFFICIENTS), simulations=search["simulations"],
            proposals=search["proposals"], c_puct=search["c_puct"], node_limit=search["node_limit"],
            episode_wall_time_seconds=search["episode_wall_time_seconds"],
            move_limit_policy=search["move_limit_policy"], model_provenance=provenance,
        )

    def _parse_all(self) -> dict[str, ExecutableSearchConfiguration]:
        result = {}
        for row in self.addendum["model_search_matrix"]:
            parsed = self._parse_row(row)
            if parsed.configuration_id in result:
                raise ValueError("duplicate configuration identifier")
            result[parsed.configuration_id] = parsed
        return result

    def policy_factory(self, config: ExecutableSearchConfiguration) -> Callable[[], Any]:
        if config.policy_kind == "residual_informed":
            return lambda: ResidualInformedPolicy((4, 4, 4), config.coefficients)
        model, model_config, _ = self._load_model(config.policy_checkpoint)
        return lambda: NeuralActionPolicy(model, model_config, self.device)

    def evaluator_factory(self, config: ExecutableSearchConfiguration, max_moves: int) -> Callable[[], Any]:
        if config.value_kind == "nonzero_heuristic":
            return HeuristicValueEvaluator
        if config.value_kind == "constant_max_moves":
            return lambda: ConstantValueEvaluator(float(max_moves))
        model, model_config, _ = self._load_model(config.value_checkpoint)
        if config.value_kind == "completion_length":
            return lambda: NeuralValueEvaluator(model, model_config, self.device)
        if config.value_kind == "direct_return":
            return lambda: DirectReturnValueEvaluator(model, model_config, self.device)
        raise ValueError("unsupported value evaluator")

    def run_episode(
        self, configuration_id: str, target, max_moves: int, seed: int,
        experiment_id: str,
    ):
        config = self.configurations[configuration_id]
        return play_episode(
            target, max_moves,
            policy_factory=self.policy_factory(config),
            evaluator_factory=self.evaluator_factory(config, max_moves),
            config=config.collection_config, seed=seed,
            experiment_id=experiment_id,
            configuration_id=config.configuration_id,
            model_provenance=config.model_provenance,
            episode_wall_time_seconds=config.episode_wall_time_seconds,
        )

    def run_fixture_episode(self, configuration_id: str, target, max_moves: int = 1, seed: int = 20261013):
        return self.run_episode(
            configuration_id, target, max_moves, seed,
            "tg4b2c-synthetic-wiring-fixture",
        )


def validate_ablation_isolation(configurations: dict[str, ExecutableSearchConfiguration]) -> dict[str, bool]:
    c = configurations
    checks = {
        "policy_only_value_fixed": c["frozen_policy_heuristic_value"].value_kind == c["policy_only_natural_policy_heuristic_value"].value_kind == "nonzero_heuristic",
        "value_only_policy_fixed": all(c[name].policy_kind == "residual_informed" for name in (
            "residual_frozen_checkpoint_value", "residual_value_only_natural_checkpoint_value",
            "residual_value_only_balanced_checkpoint_value")),
        "value_balance_differs_only_checkpoint": c["residual_value_only_natural_checkpoint_value"].value_kind == c["residual_value_only_balanced_checkpoint_value"].value_kind == "direct_return",
        "joint_checkpoints_matched": all(c[name].policy_checkpoint == c[name].value_checkpoint for name in (
            "frozen_policy_frozen_value", "joint_natural_policy_value", "joint_balanced_policy_value")),
        "common_search_budgets": len({(x.simulations, x.proposals, x.c_puct, x.node_limit, x.episode_wall_time_seconds, x.move_limit_policy) for x in c.values()}) == 1,
    }
    if not all(checks.values()):
        raise ValueError("ablation isolation check failed")
    return checks


def build_trial_schedule(configurations: dict[str, ExecutableSearchConfiguration], plan: dict[str, Any], target_count: int = 30) -> list[ScheduledTrial]:
    schedule = []
    for target_index in range(target_count):
        target_group_id = f"test-target-digest-order-{target_index:03d}"
        for seed in plan["trial_seeds"]:
            for identifier in sorted(configurations):
                config = configurations[identifier]
                core = {"configuration_id": identifier, "target_group_id": target_group_id, "seed": seed}
                schedule.append(ScheduledTrial(
                    trial_id=_digest(core), **core,
                    episode_configuration={
                        "simulations": config.simulations, "proposals": config.proposals,
                        "c_puct": config.c_puct, "node_limit": config.node_limit,
                        "episode_wall_time_seconds": config.episode_wall_time_seconds,
                        "move_limit_policy": config.move_limit_policy,
                    },
                    checkpoint_provenance=config.model_provenance["checkpoint_digests"],
                ))
    keys = {(x.configuration_id, x.target_group_id, x.seed) for x in schedule}
    if len(schedule) != target_count * len(plan["trial_seeds"]) * len(configurations) or len(keys) != len(schedule):
        raise ValueError("trial schedule is incomplete or contains duplicates")
    return schedule


class TrialStore:
    def __init__(self, root: Path, protocol_digest: str, checkpoint_digests: dict[str, str]):
        self.root = root
        self.protocol = {"protocol_digest": protocol_digest, "checkpoint_digests": checkpoint_digests}

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "resume_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.protocol:
                raise ValueError("resume protocol or checkpoint digests conflict")
        else:
            self._atomic_write(manifest, self.protocol)
        (self.root / "trials").mkdir(exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write(self, record: TrialRecord) -> str:
        path = self.root / "trials" / f"{record.trial_id}.json"
        payload = asdict(record)
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("conflicting duplicate trial result")
            return "already_recorded"
        self._atomic_write(path, payload)
        return "recorded"

    def completed_trial_ids(self) -> set[str]:
        trials = self.root / "trials"
        return {path.stem for path in trials.glob("*.json")} if trials.exists() else set()

    def progress(self, schedule: list[ScheduledTrial]) -> dict[str, int]:
        recorded = self.completed_trial_ids()
        scheduled = {trial.trial_id for trial in schedule}
        unknown = recorded - scheduled
        if unknown:
            raise ValueError("result store contains trials outside the frozen schedule")
        return {
            "scheduled": len(scheduled),
            "recorded": len(recorded),
            "never_started": len(scheduled - recorded),
        }

    def update_progress_manifest(self, schedule: list[ScheduledTrial]) -> dict[str, int]:
        progress = self.progress(schedule)
        self._atomic_write(self.root / "progress.json", progress)
        return progress

    def recover_progress(self, schedule: list[ScheduledTrial]) -> dict[str, int]:
        """Rebuild progress from durable trial records after any crash window."""
        return self.update_progress_manifest(schedule)


def record_from_episode(trial: ScheduledTrial, episode: dict[str, Any], decisions: list[dict[str, Any]]) -> TrialRecord:
    outcome = episode["outcome"]
    if outcome == "success":
        status = "completed_success"
    elif episode["stopping_reason"] == "episode_wall_time_limit":
        status = "episode_timeout"
    elif outcome == "interrupted":
        status = "interrupted"
    else:
        status = "bounded_search_failure"
    return TrialRecord(
        trial_id=trial.trial_id, status=status, termination_reason=episode["stopping_reason"],
        configuration_id=trial.configuration_id, target_group_id=trial.target_group_id,
        seed=trial.seed, attempted=True, exact_reconstruction=episode["exact_reconstruction"],
        checkpoint_provenance=trial.checkpoint_provenance,
        result={"episode": episode, "decisions": decisions},
    )


def infrastructure_failure_record(trial: ScheduledTrial, error: BaseException, *, interrupted: bool = False) -> TrialRecord:
    diagnostic = " ".join(str(error).split())[:500] or "no diagnostic text"
    return TrialRecord(
        trial_id=trial.trial_id, status="interrupted" if interrupted else "infrastructure_failure",
        termination_reason="external_interruption" if interrupted else "infrastructure_exception",
        configuration_id=trial.configuration_id, target_group_id=trial.target_group_id,
        seed=trial.seed, attempted=True, exact_reconstruction=False,
        checkpoint_provenance=trial.checkpoint_provenance,
        exception_type=type(error).__name__ if not interrupted else None, diagnostic=diagnostic,
    )
