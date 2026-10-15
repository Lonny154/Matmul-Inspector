"""Optional PyTorch policy/value learning for synthetic TensorGame data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, subtract_action, tensor_shape, validate_action
from .tensor_game_dataset import (
    SyntheticDataset,
    TorchSyntheticDataset,
    action_from_dict,
    load_dataset,
)
from .tensor_game_search import (
    HybridSampledPolicy,
    ResidualInformedPolicy,
    UniformSampledPolicy,
    canonicalize_action,
)

EXPERIMENT_VERSION = 1
COEFFICIENTS = (-1, 0, 1)
COEFFICIENT_TO_CLASS = {-1: 0, 0: 1, 1: 2}


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("TensorGame learning requires optional PyTorch") from error
    return torch


@dataclass(frozen=True)
class PolicyValueConfig:
    shape: tuple[int, int, int] = (4, 4, 4)
    hidden_dim: int = 256
    coefficients: tuple[int, ...] = COEFFICIENTS
    input_scale: float = 4.0

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.coefficients != COEFFICIENTS:
            raise ValueError("the initial neural policy requires coefficient order (-1, 0, 1)")
        if self.input_scale <= 0 or not math.isfinite(self.input_scale):
            raise ValueError("input_scale must be finite and positive")

    @property
    def input_width(self) -> int:
        return math.prod(self.shape)

    @property
    def action_width(self) -> int:
        return sum(self.shape)


def create_policy_value_model(config: PolicyValueConfig):
    """Create the optional two-layer MLP without importing Torch at module import."""
    torch = _torch()

    class PolicyValueMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Linear(config.input_width, config.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(config.hidden_dim, config.hidden_dim),
                torch.nn.GELU(),
            )
            self.policy_head = torch.nn.Linear(
                config.hidden_dim, config.action_width * len(config.coefficients)
            )
            self.value_head = torch.nn.Linear(config.hidden_dim, 1)

        def forward(self, residual):
            flattened = residual.reshape(residual.shape[0], -1).to(torch.float32)
            shared = self.features(flattened / config.input_scale)
            logits = self.policy_head(shared).reshape(
                residual.shape[0], config.action_width, len(config.coefficients)
            )
            value = self.value_head(shared).squeeze(-1)
            return logits, value

    return PolicyValueMLP()


def encode_action_coefficients(actions):
    """Map exact coefficient values to policy class indices."""
    torch = _torch()
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions)
    result = torch.full_like(actions, -1, dtype=torch.long)
    for coefficient, class_index in COEFFICIENT_TO_CLASS.items():
        result = torch.where(actions == coefficient, class_index, result)
    if torch.any(result < 0):
        raise ValueError("action contains a coefficient outside {-1, 0, 1}")
    return result


def valid_action_log_probabilities(logits, targets, action_mask):
    """Return log summed probability of each row's deduplicated valid actions."""
    torch = _torch()
    if logits.ndim != 3 or targets.ndim != 3 or action_mask.ndim != 2:
        raise ValueError("expected logits [B,W,3], targets [B,A,W], and mask [B,A]")
    if logits.shape[0] != targets.shape[0] or logits.shape[1] != targets.shape[2]:
        raise ValueError("policy logits and complete-action targets have incompatible shapes")
    if targets.shape[:2] != action_mask.shape or logits.shape[2] != 3:
        raise ValueError("action mask or coefficient-class dimension is invalid")
    log_coefficients = torch.log_softmax(logits, dim=-1)
    row_values = []
    for batch_index in range(logits.shape[0]):
        valid = targets[batch_index][action_mask[batch_index]]
        if valid.numel() == 0:
            raise ValueError("every example must contain at least one valid complete action")
        valid = torch.unique(valid, dim=0)
        classes = encode_action_coefficients(valid)
        expanded = log_coefficients[batch_index].unsqueeze(0).expand(valid.shape[0], -1, -1)
        action_log_probabilities = expanded.gather(2, classes.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        row_values.append(torch.logsumexp(action_log_probabilities, dim=0))
    return torch.stack(row_values)


def policy_value_loss(
    logits,
    value_prediction,
    targets,
    action_mask,
    completion_length,
    value_coefficient: float,
):
    torch = _torch()
    if value_coefficient < 0:
        raise ValueError("value_coefficient must be nonnegative")
    policy = -valid_action_log_probabilities(logits, targets, action_mask).mean()
    value = torch.nn.functional.mse_loss(value_prediction, completion_length.to(value_prediction.dtype))
    return policy + value_coefficient * value, policy, value


def _action_from_flat(values: Sequence[int], shape: tuple[int, int, int]) -> RankOneAction:
    first, second, third = shape
    return RankOneAction(
        values[:first],
        values[first : first + second],
        values[first + second : first + second + third],
    )


def rank_complete_actions(logits, shape: tuple[int, int, int], max_actions: int) -> tuple[tuple[RankOneAction, float], ...]:
    """Beam-rank valid complete actions under the independent coefficient model."""
    return rank_complete_actions_audited(logits, shape, max_actions)[0]


def rank_complete_actions_audited(logits, shape: tuple[int, int, int], max_actions: int):
    """Return ranked actions plus explicit internal decoding-effort counters."""
    torch = _torch()
    if logits.shape != (sum(shape), 3) or max_actions <= 0:
        raise ValueError("single-example logits or max_actions are invalid")
    started = time.perf_counter()
    log_probabilities = torch.log_softmax(logits.detach().cpu(), dim=-1)
    # Retain enough alternatives to escape a high-probability all-zero factor.
    beam_width = max(256, max_actions * 32)
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    internal_candidates = 0
    for position in range(log_probabilities.shape[0]):
        candidates = []
        for prefix, score in beam:
            for class_index, coefficient in enumerate(COEFFICIENTS):
                candidates.append((prefix + (coefficient,), score + float(log_probabilities[position, class_index])))
        internal_candidates += len(candidates)
        candidates.sort(key=lambda item: (-item[1], item[0]))
        beam = candidates[:beam_width]
    accepted: dict[RankOneAction, float] = {}
    invalid = 0
    duplicates = 0
    decoded = 0
    for values, score in beam:
        decoded += 1
        action = _action_from_flat(values, shape)
        try:
            validate_action(action, shape, frozenset(COEFFICIENTS))
        except ValueError:
            invalid += 1
            continue
        canonical = canonicalize_action(action)
        if canonical in accepted:
            duplicates += 1
        accepted[canonical] = max(score, accepted.get(canonical, -math.inf))
        if len(accepted) >= max_actions:
            break
    ranked = tuple(sorted(accepted.items(), key=lambda item: (-item[1], item[0].u, item[0].v, item[0].w)))
    return ranked, {
        "requested": max_actions,
        "internal_candidates_considered": internal_candidates,
        "final_beam_size": len(beam),
        "decoded_candidates": decoded,
        "invalid_rejected": invalid,
        "canonical_duplicates_rejected": duplicates,
        "returned": len(ranked),
        "valid_unique": len(ranked),
        "generation_seconds": time.perf_counter() - started,
    }


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    value_coefficient: float = 1.0
    gradient_clip: float = 1.0
    patience: int = 5
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.value_coefficient < 0 or self.gradient_clip <= 0 or self.patience <= 0:
            raise ValueError("loss coefficient, clipping, and patience must be valid")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def resolve_device(requested: str):
    torch = _torch()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")


def _set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _epoch(model, loader, device, value_coefficient, optimizer=None, gradient_clip=1.0):
    torch = _torch()
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "gradient_norm": 0.0, "samples": 0}
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            residual = batch["residual"].to(device)
            targets = batch["next_actions"].to(device)
            mask = batch["action_mask"].to(device)
            completion = batch["known_completion_length"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, value = model(residual)
            loss, policy, value_loss = policy_value_loss(
                logits, value, targets, mask, completion, value_coefficient
            )
            gradient_norm = 0.0
            if training:
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
                optimizer.step()
            count = residual.shape[0]
            totals["samples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["policy_loss"] += float(policy.detach()) * count
            totals["value_loss"] += float(value_loss.detach()) * count
            totals["gradient_norm"] += gradient_norm * count
    elapsed = time.perf_counter() - started
    count = max(1, totals.pop("samples"))
    return {
        **{key: value / count for key, value in totals.items()},
        "seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed else 0.0,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def train_model(
    dataset: SyntheticDataset,
    output: Path,
    model_config: PolicyValueConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """Train once, select by validation loss, and persist an explicit state dict."""
    torch = _torch()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    model = create_policy_value_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    generator = torch.Generator().manual_seed(training_config.seed)
    train_data = TorchSyntheticDataset(dataset, "train")
    validation_data = TorchSyntheticDataset(dataset, "validation")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=training_config.batch_size, shuffle=True, generator=generator
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=training_config.batch_size, shuffle=False
    )
    history = []
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    checkpoint = output / "best_model_state.pt"
    for epoch in range(1, training_config.epochs + 1):
        train_metrics = _epoch(
            model, train_loader, device, training_config.value_coefficient, optimizer, training_config.gradient_clip
        )
        validation_metrics = _epoch(
            model, validation_loader, device, training_config.value_coefficient
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= training_config.patience:
            break
    wall_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, dataset, "test", model_config, training_config.batch_size, device)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
    }
    dataset_content_digest = hashlib.sha256(
        "".join(
            record["content_digest"]
            for record in (*dataset.decompositions, *dataset.samples)
        ).encode("ascii")
    ).hexdigest()
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_schema_version": dataset.manifest["schema_version"],
        "dataset_content_digest": dataset_content_digest,
        "split_decomposition_counts": dataset.manifest["split_decomposition_counts"],
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "optimizer": "AdamW",
        "scheduler": None,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_wall_seconds": wall_seconds,
        "history": history,
        "test_metrics": test_metrics,
        "environment": environment,
        "maximum_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "checkpoint": checkpoint.name,
    }
    _atomic_json(output / "experiment.json", manifest)
    return manifest


def load_checkpoint(experiment: Path, device: str = "cpu"):
    torch = _torch()
    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("unsupported TensorGame learning experiment version")
    config_value = manifest["model_config"]
    config = PolicyValueConfig(
        shape=tuple(config_value["shape"]),
        hidden_dim=config_value["hidden_dim"],
        coefficients=tuple(config_value["coefficients"]),
        input_scale=config_value["input_scale"],
    )
    resolved = resolve_device(device)
    model = create_policy_value_model(config).to(resolved)
    state = torch.load(experiment / manifest["checkpoint"], map_location=resolved, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config, manifest


def _flat_action(action: RankOneAction) -> tuple[int, ...]:
    return action.u + action.v + action.w


def evaluate_model(model, dataset: SyntheticDataset, split: str, config: PolicyValueConfig, batch_size: int, device) -> dict[str, Any]:
    torch = _torch()
    adapter = TorchSyntheticDataset(dataset, split)
    split_samples = [sample for sample in dataset.samples if sample["split"] == split]
    loader = torch.utils.data.DataLoader(adapter, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0
    matches = 0
    valid_predictions = 0
    known_mass = 0.0
    absolute_error = 0.0
    squared_error = 0.0
    value_by_length: dict[str, list[float]] = {}
    value_predictions = []
    nnz_changes = []
    exact_transitions = 0
    with torch.no_grad():
        offset = 0
        for batch in loader:
            residuals = batch["residual"].to(device)
            logits, values = model(residuals)
            log_mass = valid_action_log_probabilities(
                logits, batch["next_actions"].to(device), batch["action_mask"].to(device)
            )
            for row in range(residuals.shape[0]):
                source = split_samples[offset + row]
                known = {canonicalize_action(action_from_dict(item)) for item in source["next_actions"]}
                ranked = rank_complete_actions(logits[row], config.shape, 1)
                if ranked:
                    valid_predictions += 1
                    predicted = ranked[0][0]
                    matches += predicted in known
                    residual = tuple(tuple(tuple(line) for line in plane) for plane in source["residual"])
                    updated = subtract_action(residual, predicted)
                    exact_transitions += 1
                    before = sum(value != 0 for plane in residual for line in plane for value in line)
                    after = sum(value != 0 for plane in updated for line in plane for value in line)
                    nnz_changes.append(before - after)
                target_value = float(source["known_completion_length"])
                prediction = float(values[row])
                error = prediction - target_value
                absolute_error += abs(error)
                squared_error += error * error
                value_by_length.setdefault(str(source["known_completion_length"]), []).append(abs(error))
                value_predictions.append(prediction)
            known_mass += float(torch.exp(log_mass).sum())
            total += residuals.shape[0]
            offset += residuals.shape[0]
    return {
        "examples": total,
        "top1_known_action_match": matches / total if total else 0.0,
        "valid_nonzero_action_fraction": valid_predictions / total if total else 0.0,
        "exact_transition_fraction": exact_transitions / total if total else 0.0,
        "average_known_action_probability_mass": known_mass / total if total else 0.0,
        "value_mae": absolute_error / total if total else 0.0,
        "value_rmse": math.sqrt(squared_error / total) if total else 0.0,
        "value_mae_by_known_completion_length": {
            key: sum(values) / len(values) for key, values in sorted(value_by_length.items())
        },
        "value_prediction_min": min(value_predictions) if value_predictions else None,
        "value_prediction_max": max(value_predictions) if value_predictions else None,
        "value_prediction_mean": sum(value_predictions) / len(value_predictions) if value_predictions else None,
        "mean_residual_nonzero_reduction": sum(nnz_changes) / len(nnz_changes) if nnz_changes else 0.0,
    }


def compare_proposals(
    model,
    dataset: SyntheticDataset,
    split: str,
    model_config: PolicyValueConfig,
    proposal_count: int,
    seeds: Sequence[int],
    device,
) -> dict[str, Any]:
    """Compare proposal quality only; known factors are used solely for scoring."""
    torch = _torch()
    samples = [sample for sample in dataset.samples if sample["split"] == split]
    results: dict[str, Any] = {}
    learned_hits = valid = proposed = unique = 0
    model.eval()
    with torch.no_grad():
        for sample in samples:
            residual = torch.tensor(sample["residual"], dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(residual)
            ranked = rank_complete_actions(logits[0], model_config.shape, proposal_count)
            known = {canonicalize_action(action_from_dict(item)) for item in sample["next_actions"]}
            actions = [item[0] for item in ranked]
            proposed += proposal_count
            valid += len(actions)
            unique += len(set(actions))
            learned_hits += any(action in known for action in actions)
    results["learned"] = {
        "examples_with_known_action": learned_hits,
        "hit_rate": learned_hits / len(samples) if samples else 0.0,
        "valid_fraction": valid / proposed if proposed else 0.0,
        "unique_fraction": unique / valid if valid else 0.0,
    }
    factories = {
        "uniform_sampled": lambda: UniformSampledPolicy(model_config.shape),
        "residual_informed": lambda: ResidualInformedPolicy(model_config.shape),
        "hybrid": lambda: HybridSampledPolicy(model_config.shape),
    }
    for name, factory in factories.items():
        hits = accepted = requested = unique_count = 0
        for seed in seeds:
            rng = random.Random(seed)
            policy = factory()
            for sample in samples:
                residual = tuple(tuple(tuple(line) for line in plane) for plane in sample["residual"])
                proposals = policy.propose_actions(
                    residual, sample["known_completion_length"], rng, proposal_count
                )
                known = {canonicalize_action(action_from_dict(item)) for item in sample["next_actions"]}
                actions = [item.action for item in proposals]
                requested += proposal_count
                accepted += len(actions)
                unique_count += len(set(actions))
                hits += any(action in known for action in actions)
        denominator = len(samples) * len(seeds)
        results[name] = {
            "examples_with_known_action": hits,
            "hit_rate": hits / denominator if denominator else 0.0,
            "valid_fraction": accepted / requested if requested else 0.0,
            "unique_fraction": unique_count / accepted if accepted else 0.0,
            "seeds": list(seeds),
        }
    return results
