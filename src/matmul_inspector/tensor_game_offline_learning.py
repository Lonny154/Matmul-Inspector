"""Offline policy/value learning from validated TensorGame replay buffers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any, Sequence

from .tensor_game_experience import REWARD_CONVENTION, ReplayBuffer, action_from_payload, load_buffer
from .tensor_game_learning import COEFFICIENTS, PolicyValueConfig, create_policy_value_model, resolve_device

EXPERIMENT_VERSION = 1
VALUE_SEMANTICS = "direct_observed_mcts_return_not_completion_length"
MODES = ("policy_only", "value_only", "joint")


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("offline TensorGame learning requires optional PyTorch") from error
    return torch


def experience_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("manifest.json", "episodes.json", "decisions.json"):
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OfflineTrainingConfig:
    mode: str = "joint"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 0.0005
    policy_coefficient: float = 1.0
    value_coefficient: float = 1.0
    gradient_clip: float = 1.0
    patience: int = 3
    seed: int = 42
    device: str = "cpu"
    strict_torso_freeze: bool = False

    def __post_init__(self):
        if self.mode not in MODES or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("invalid training mode, epoch count, or batch size")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.patience <= 0:
            raise ValueError("optimizer settings must be positive")
        if self.policy_coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("loss coefficients must be nonnegative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")


def grouped_splits(
    buffer: ReplayBuffer,
    target_assignments: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = {}
    for episode in buffer.episodes:
        groups.setdefault(episode["initial_target_digest"], []).append(episode)
    ordered = sorted(groups)
    if target_assignments is None:
        if len(ordered) < 3:
            raise ValueError("at least three independent target groups are required")
        assignments = {}
        for index, target in enumerate(ordered):
            assignments[target] = "test" if index == len(ordered) - 1 else "validation" if index == len(ordered) - 2 else "train"
        assignment_source = "deterministic_digest_fallback"
    else:
        if set(target_assignments) != set(groups):
            raise ValueError("persisted target partitions do not match collected target groups")
        if any(split not in {"train", "validation", "test"} for split in target_assignments.values()):
            raise ValueError("persisted target partition contains an invalid split")
        assignments = dict(target_assignments)
        assignment_source = "persisted_precollection_target_partitions"
    episode_split = {
        episode["episode_id"]: assignments[episode["initial_target_digest"]]
        for episode in buffer.episodes
    }
    residual_splits = {}
    collisions = []
    for decision in buffer.decisions.values():
        split = episode_split[decision["episode_id"]]
        previous = residual_splits.setdefault(decision["state_digest"], split)
        if previous != split:
            collisions.append(decision["state_digest"])
    if collisions:
        raise ValueError("identical residual tensors cross grouped splits")
    return episode_split, {
        "target_groups": len(groups), "assignments": assignments, "assignment_source": assignment_source,
        "episode_counts": {name: sum(value == name for value in episode_split.values()) for name in ("train", "validation", "test")},
        "residual_collisions": collisions,
        "target_collisions": [],
        "excluded_decisions": 0,
        "warning": "small replay buffer; split metrics are pipeline diagnostics only" if len(groups) < 10 else None,
    }


class ExperienceDataset:
    """Validated variable-support replay decisions, optionally exposed to PyTorch."""

    def __init__(self, path: Path, split: str):
        if split not in {"train", "validation", "test"}:
            raise ValueError("invalid split")
        self.buffer = load_buffer(path)
        manifest = json.loads((path / "manifest.json").read_text())
        persisted = manifest.get("metadata", {}).get("target_splits")
        if persisted is not None:
            collected = {episode["initial_target_digest"] for episode in self.buffer.episodes}
            persisted = {target: value for target, value in persisted.items() if target in collected}
        episode_split, self.split_report = grouped_splits(self.buffer, persisted)
        self.target_by_episode = {
            episode["episode_id"]: episode["initial_target_digest"]
            for episode in self.buffer.episodes
        }
        self.episode_by_id = {episode["episode_id"]: episode for episode in self.buffer.episodes}
        self.rows = [row for row in self.buffer.decisions.values() if episode_split[row["episode_id"]] == split]
        self.split = split
        self.unique_residuals = len({row["state_digest"] for row in self.rows})
        for row in self.rows:
            support = row["sampled_actions"]
            if not support or len({item["action_id"] for item in support}) != len(support):
                raise ValueError("empty or duplicate sampled action support")
            probabilities = [item["search_probability"] for item in support]
            if any(value < 0 or not math.isfinite(value) for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
                raise ValueError("invalid search-policy distribution")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        torch = _torch()
        row = self.rows[index]
        actions = [action_from_payload(item["action"]) for item in row["sampled_actions"]]
        flat = [action.u + action.v + action.w for action in actions]
        value_valid = row["observed_return_target"] is not None and not row["outcome_censored"]
        return {
            "residual": torch.tensor(row["residual"], dtype=torch.float32),
            "actions": torch.tensor(flat, dtype=torch.long),
            "search_policy": torch.tensor([item["search_probability"] for item in row["sampled_actions"]], dtype=torch.float32),
            "visits": torch.tensor([item["visits"] for item in row["sampled_actions"]], dtype=torch.long),
            "value_target": torch.tensor(float(row["observed_return_target"] or 0.0)),
            "value_valid": torch.tensor(value_valid),
            "episode_id": row["episode_id"], "target_id": self.target_by_episode[row["episode_id"]],
            "outcome": self.episode_by_id[row["episode_id"]]["outcome"],
            "outcome_censored": row["outcome_censored"],
            "provenance": row["model_provenance"],
        }


def collate_experience(rows):
    torch = _torch()
    maximum = max(row["actions"].shape[0] for row in rows)
    width = rows[0]["actions"].shape[1]
    actions = torch.zeros((len(rows), maximum, width), dtype=torch.long)
    policy = torch.zeros((len(rows), maximum), dtype=torch.float32)
    visits = torch.zeros((len(rows), maximum), dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = row["actions"].shape[0]
        actions[index, :count] = row["actions"]
        policy[index, :count] = row["search_policy"]
        visits[index, :count] = row["visits"]
        mask[index, :count] = True
    return {
        "residual": torch.stack([row["residual"] for row in rows]),
        "actions": actions, "search_policy": policy, "visits": visits, "action_mask": mask,
        "value_target": torch.stack([row["value_target"] for row in rows]),
        "value_valid": torch.stack([row["value_valid"] for row in rows]),
        "episode_id": [row["episode_id"] for row in rows],
        "target_id": [row["target_id"] for row in rows],
        "outcome": [row["outcome"] for row in rows],
        "outcome_censored": [row["outcome_censored"] for row in rows],
        "provenance": [row["provenance"] for row in rows],
    }


def _move_tensors(batch, device):
    torch = _torch()
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def sampled_support_log_probabilities(logits, actions, mask, shape):
    """Differentiable sign-aggregated joint probabilities normalized on support."""
    torch = _torch()
    if (
        logits.ndim != 3 or actions.ndim != 3 or mask.shape != actions.shape[:2]
        or len(shape) != 3 or any(value <= 0 for value in shape)
        or sum(shape) != logits.shape[1] or actions.shape[2] != sum(shape)
    ):
        raise ValueError("invalid sampled-support tensor shapes")
    coefficient_class = {-1: 0, 0: 1, 1: 2}
    log_coefficients = torch.log_softmax(logits, dim=-1)
    output = torch.full(mask.shape, -torch.inf, dtype=logits.dtype, device=logits.device)
    for batch in range(actions.shape[0]):
        if not torch.any(mask[batch]):
            raise ValueError("empty sampled action support")
        identities = set()
        scores = []
        positions = []
        for action_index in torch.nonzero(mask[batch], as_tuple=False).flatten().tolist():
            values = tuple(int(x) for x in actions[batch, action_index].tolist())
            first, second = shape[0], shape[0] + shape[1]
            u, v, w = values[:first], values[first:second], values[second:]
            forms = ((u, v, w), (tuple(-x for x in u), tuple(-x for x in v), w), (tuple(-x for x in u), v, tuple(-x for x in w)), (u, tuple(-x for x in v), tuple(-x for x in w)))
            identity = min(form[0] + form[1] + form[2] for form in forms)
            if identity in identities:
                raise ValueError("duplicate sampled action")
            identities.add(identity)
            form_scores = []
            for form in forms:
                flat = form[0] + form[1] + form[2]
                indices = torch.tensor([coefficient_class[x] for x in flat], device=logits.device)
                form_scores.append(log_coefficients[batch].gather(1, indices[:, None]).sum())
            scores.append(torch.logsumexp(torch.stack(form_scores), dim=0))
            positions.append(action_index)
        normalized = torch.stack(scores) - torch.logsumexp(torch.stack(scores), dim=0)
        output[batch, positions] = normalized
    return output


def offline_loss(logits, values, batch, shape, policy_coefficient=1.0, value_coefficient=1.0):
    torch = _torch()
    log_policy = sampled_support_log_probabilities(logits, batch["actions"], batch["action_mask"], shape)
    policy_loss = -(batch["search_policy"] * torch.where(batch["action_mask"], log_policy, 0.0)).sum(1).mean()
    valid = batch["value_valid"]
    value_loss = torch.nn.functional.mse_loss(values[valid], batch["value_target"][valid]) if torch.any(valid) else values.sum() * 0
    return policy_coefficient * policy_loss + value_coefficient * value_loss, policy_loss, value_loss


def configure_trainable(model, mode: str, strict_torso_freeze: bool):
    if mode not in MODES: raise ValueError("invalid mode")
    for parameter in model.parameters(): parameter.requires_grad = True
    if mode == "policy_only":
        for parameter in model.value_head.parameters(): parameter.requires_grad = False
    elif mode == "value_only":
        for parameter in model.policy_head.parameters(): parameter.requires_grad = False
    if strict_torso_freeze:
        for parameter in model.features.parameters(): parameter.requires_grad = False
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def evaluate_offline(model, dataset, config, device, shape):
    torch = _torch(); loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_experience)
    totals = {"examples": 0, "policy_cross_entropy": 0.0, "policy_kl": 0.0, "top1": 0, "value_abs": 0.0, "value_sq": 0.0, "value_count": 0, "entropy": 0.0}; predictions=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_tensors(batch, device)
            logits, values = model(batch["residual"]); logp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],shape)
            ce=-(batch["search_policy"]*torch.where(batch["action_mask"],logp,0.0)).sum(1)
            entropy=-(batch["search_policy"]*torch.where(batch["search_policy"]>0,torch.log(batch["search_policy"]),0.0)).sum(1)
            count=len(values); totals["examples"]+=count; totals["policy_cross_entropy"]+=float(ce.sum()); totals["policy_kl"]+=float((ce-entropy).sum()); totals["entropy"]+=float(entropy.sum())
            masked=logp.masked_fill(~batch["action_mask"],-torch.inf); target=batch["search_policy"].argmax(1); totals["top1"]+=int((masked.argmax(1)==target).sum())
            valid=batch["value_valid"]; errors=values[valid]-batch["value_target"][valid]; totals["value_abs"]+=float(errors.abs().sum()); totals["value_sq"]+=float((errors*errors).sum()); totals["value_count"]+=int(valid.sum()); predictions.extend(values.tolist())
    n=max(1,totals["examples"]); v=max(1,totals["value_count"])
    return {"examples":totals["examples"],"policy_cross_entropy":totals["policy_cross_entropy"]/n,"policy_kl":totals["policy_kl"]/n,"top1_agreement":totals["top1"]/n,"policy_entropy":totals["entropy"]/n,"value_mae":totals["value_abs"]/v,"value_rmse":math.sqrt(totals["value_sq"]/v),"value_examples":totals["value_count"],"value_prediction_min":min(predictions) if predictions else None,"value_prediction_max":max(predictions) if predictions else None}


def train_offline(experience: Path, source_checkpoint: Path, output: Path, training: OfflineTrainingConfig):
    torch=_torch()
    if output.exists(): raise FileExistsError(output)
    from .tensor_game_learning import load_checkpoint
    source_model, model_config, source_manifest=load_checkpoint(source_checkpoint,training.device); device=resolve_device(training.device)
    source_sha=hashlib.sha256((source_checkpoint/source_manifest["checkpoint"]).read_bytes()).hexdigest()
    train=ExperienceDataset(experience,"train"); validation=ExperienceDataset(experience,"validation"); test=ExperienceDataset(experience,"test")
    random.seed(training.seed); torch.manual_seed(training.seed); torch.use_deterministic_algorithms(True); trainable=configure_trainable(source_model,training.mode,training.strict_torso_freeze)
    optimizer=torch.optim.AdamW((p for p in source_model.parameters() if p.requires_grad),lr=training.learning_rate)
    generator=torch.Generator().manual_seed(training.seed)
    loader=torch.utils.data.DataLoader(train,batch_size=training.batch_size,shuffle=True,generator=generator,collate_fn=collate_experience)
    output.mkdir(parents=True); best=math.inf; best_epoch=None; stale=0; history=[]; checkpoint=output/"best_model_state.pt"; started=time.perf_counter()
    baseline={split:evaluate_offline(source_model,data,training,device,model_config.shape) for split,data in (("validation",validation),("test",test))}
    policy_coefficient=training.policy_coefficient if training.mode!="value_only" else 0.0
    value_coefficient=training.value_coefficient if training.mode!="policy_only" else 0.0
    for epoch in range(1,training.epochs+1):
        source_model.train(); total=0.0; count=0
        for batch in loader:
            batch=_move_tensors(batch,device); optimizer.zero_grad(set_to_none=True); logits,values=source_model(batch["residual"]); loss,_,_=offline_loss(logits,values,batch,model_config.shape,policy_coefficient,value_coefficient); loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in source_model.parameters() if p.requires_grad),training.gradient_clip); optimizer.step(); total+=float(loss.detach())*len(values); count+=len(values)
        metrics=evaluate_offline(source_model,validation,training,device,model_config.shape); history.append({"epoch":epoch,"train_loss":total/max(1,count),"validation":metrics})
        score=policy_coefficient*metrics["policy_cross_entropy"]+value_coefficient*metrics["value_rmse"]
        if score<best: best=score; best_epoch=epoch; stale=0; torch.save(source_model.state_dict(),checkpoint)
        else: stale+=1
        if stale>=training.patience: break
    source_model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); final={split:evaluate_offline(source_model,data,training,device,model_config.shape) for split,data in (("validation",validation),("test",test))}
    elapsed=time.perf_counter()-started
    observed_returns=[row["observed_return_target"] for row in train.buffer.decisions.values() if row["observed_return_target"] is not None and not row["outcome_censored"]]
    outcomes={name:sum(episode["outcome"]==name for episode in train.buffer.episodes) for name in ("success","truncated","interrupted")}
    manifest={"experiment_version":EXPERIMENT_VERSION,"value_head_semantics":VALUE_SEMANTICS,"reward_convention":REWARD_CONVENTION,"model_config":asdict(model_config),"training_config":asdict(training),"optimizer":"AdamW","effective_loss_coefficients":{"policy":policy_coefficient,"value":value_coefficient},"trainable_parameters":trainable,"shared_torso_caveat":not training.strict_torso_freeze and training.mode!="joint","experience_digest":experience_digest(experience),"dataset_digest":experience_digest(experience),"source_checkpoint_sha256":source_sha,"split_report":train.split_report,"effective_samples":{"train":len(train),"validation":len(validation),"test":len(test),"unique_train_residuals":train.unique_residuals,"censored_value_targets":sum(row["outcome_censored"] for row in train.buffer.decisions.values())},"return_target_distribution":{"count":len(observed_returns),"minimum":min(observed_returns),"maximum":max(observed_returns),"mean":sum(observed_returns)/len(observed_returns)},"episode_outcomes":outcomes,"baseline":baseline,"history":history,"best_epoch":best_epoch,"best_validation_score":best,"final":final,"checkpoint":checkpoint.name,"training_seconds":elapsed,"training_examples_per_second":len(train)*len(history)/elapsed,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":str(device),"platform":platform.platform()}}
    (output/"experiment.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); return manifest


def load_offline_checkpoint(output: Path, device="cpu"):
    torch=_torch(); manifest=json.loads((output/"experiment.json").read_text())
    if manifest.get("experiment_version")!=EXPERIMENT_VERSION or manifest.get("value_head_semantics")!=VALUE_SEMANTICS: raise ValueError("unsupported offline checkpoint")
    value=manifest["model_config"]; config=PolicyValueConfig(shape=tuple(value["shape"]),hidden_dim=value["hidden_dim"],coefficients=tuple(value["coefficients"]),input_scale=value["input_scale"]); model=create_policy_value_model(config).to(resolve_device(device)); model.load_state_dict(torch.load(output/manifest["checkpoint"],map_location=device,weights_only=True)); model.eval(); return model,config,manifest
