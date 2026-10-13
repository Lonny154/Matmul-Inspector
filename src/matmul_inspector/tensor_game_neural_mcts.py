"""Optional neural policy and value adapters for sampled TensorGame MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
from typing import Any

from .tensor_game import RankOneAction, Tensor, tensor_shape, validate_action, zero_tensor
from .tensor_game_learning import COEFFICIENTS, PolicyValueConfig, rank_complete_actions_audited
from .tensor_game_search import (
    InstrumentedPolicy,
    ProposalMetrics,
    ProposedAction,
    ResidualInformedPolicy,
    ValueMetrics,
    _normalize,
    canonicalize_action,
)

NEURAL_POLICY_VERSION = "tg3c.neural_policy.v1"
NEURAL_VALUE_VERSION = "tg3c.neural_value.v1"
HYBRID_POLICY_VERSION = "tg3c.neural_residual_hybrid.v1"


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("neural TensorGame search requires optional PyTorch") from error
    return torch


def _sign_forms(action: RankOneAction) -> tuple[RankOneAction, ...]:
    u, v, w = action.u, action.v, action.w
    return tuple(dict.fromkeys((
        RankOneAction(u, v, w),
        RankOneAction(tuple(-x for x in u), tuple(-x for x in v), w),
        RankOneAction(tuple(-x for x in u), v, tuple(-x for x in w)),
        RankOneAction(u, tuple(-x for x in v), tuple(-x for x in w)),
    )))


def canonical_action_log_probability(logits: Any, action: RankOneAction, shape: tuple[int, int, int]) -> float:
    """Exactly aggregate the independent-model mass of all sign-equivalent forms."""
    torch = _torch()
    if tuple(logits.shape) != (sum(shape), len(COEFFICIENTS)):
        raise ValueError("policy logits have the wrong shape")
    log_coefficients = torch.log_softmax(logits.detach(), dim=-1)
    class_by_coefficient = {value: index for index, value in enumerate(COEFFICIENTS)}
    scores = []
    for form in _sign_forms(action):
        values = form.u + form.v + form.w
        try:
            indices = [class_by_coefficient[value] for value in values]
        except KeyError as error:
            raise ValueError("action contains a coefficient outside the model vocabulary") from error
        scores.append(sum(log_coefficients[position, index] for position, index in enumerate(indices)))
    return float(torch.logsumexp(torch.stack(scores), dim=0).detach().cpu())


@dataclass
class NeuralActionPolicy(InstrumentedPolicy):
    """Decode bounded complete actions and normalize their sign-aggregated mass."""

    model: Any
    config: PolicyValueConfig
    device: Any = "cpu"
    name: str = field(default="neural", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)
    inference_calls: int = field(default=0, init=False)
    inference_seconds: float = field(default=0.0, init=False)
    prior_entropies: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if tuple(self.config.coefficients) != COEFFICIENTS:
            raise ValueError("checkpoint coefficient order is incompatible")

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.inference_calls = 0
        self.inference_seconds = 0.0
        self.prior_entropies = []

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        del rng
        if tensor_shape(residual) != self.config.shape:
            raise ValueError("residual shape does not match the neural checkpoint")
        if remaining_moves <= 0 or max_actions <= 0:
            return ()
        torch = _torch()
        started = time.perf_counter()
        tensor = torch.tensor(residual, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = self.model(tensor)
        self.inference_calls += 1
        self.inference_seconds += time.perf_counter() - started
        ranked, audit = rank_complete_actions_audited(logits[0], self.config.shape, max_actions)
        self.metrics.attempted += audit["decoded_candidates"]
        self.metrics.valid += audit["decoded_candidates"] - audit["invalid_rejected"]
        self.metrics.duplicates += audit["canonical_duplicates_rejected"]
        scored = [(action, canonical_action_log_probability(logits[0], action, self.config.shape)) for action, _ in ranked]
        if not scored:
            self.metrics.generation_seconds += time.perf_counter() - started
            return ()
        offset = max(score for _, score in scored)
        normalized = _normalize([(action, math.exp(score - offset)) for action, score in scored])
        result = tuple(ProposedAction(item.action, item.prior, ("neural",)) for item in normalized)
        self.metrics.accepted += len(result)
        self.prior_entropies.append(-sum(item.prior * math.log(item.prior) for item in result))
        self.metrics.generation_seconds += time.perf_counter() - started
        return result


@dataclass
class NeuralValueEvaluator:
    """Convert predicted future completion length into the MCTS leaf-return scale."""

    model: Any
    config: PolicyValueConfig
    device: Any = "cpu"
    name: str = field(default="neural_completion_length", init=False)
    metrics: ValueMetrics = field(default_factory=ValueMetrics, init=False, repr=False)
    trace_limit: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    clipping_by_depth: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    unique_state_digests: set[str] = field(default_factory=set, init=False, repr=False)
    raw_predictions: list[float] = field(default_factory=list, init=False, repr=False)
    clamped_lengths: list[float] = field(default_factory=list, init=False, repr=False)
    clip_reasons: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def reset_metrics(self) -> None:
        self.metrics = ValueMetrics()
        self.trace = []
        self.clipping_by_depth = {}
        self.unique_state_digests = set()
        self.raw_predictions = []
        self.clamped_lengths = []
        self.clip_reasons = {}

    def evaluate(self, residual, remaining_moves, max_moves, target_nonzeros):
        del target_nonzeros
        if tensor_shape(residual) != self.config.shape:
            raise ValueError("residual shape does not match the neural checkpoint")
        if residual == zero_tensor(self.config.shape):
            return float(max_moves)
        if remaining_moves <= 0:
            return float(-(max_moves + 1))
        torch = _torch()
        started = time.perf_counter()
        tensor = torch.tensor(residual, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            _, prediction = self.model(tensor)
        self.metrics.inference_seconds += time.perf_counter() - started
        raw = float(prediction[0].detach().cpu())
        if not math.isfinite(raw):
            clip_reason = "nonfinite"
        elif raw < 0.0:
            clip_reason = "negative"
        elif raw > remaining_moves:
            clip_reason = "above_remaining_budget"
        else:
            clip_reason = None
        clipped = clip_reason is not None
        length = float(remaining_moves) if not math.isfinite(raw) else min(float(remaining_moves), max(0.0, raw))
        value = float(max_moves) - length
        depth = max_moves - remaining_moves
        digest = hashlib.sha256(json.dumps(residual, separators=(",", ":")).encode()).hexdigest()
        self.unique_state_digests.add(digest)
        self.raw_predictions.append(raw)
        self.clamped_lengths.append(length)
        self.metrics.calls += 1
        self.metrics.clipped += int(clipped)
        self.metrics.predictions.append(value)
        if clipped:
            self.clipping_by_depth[depth] = self.clipping_by_depth.get(depth, 0) + 1
            self.clip_reasons[clip_reason] = self.clip_reasons.get(clip_reason, 0) + 1
        if len(self.trace) < self.trace_limit:
            self.trace.append({
                "state_digest": digest,
                "depth": depth,
                "remaining_moves": remaining_moves,
                "raw_predicted_length": raw,
                "clamped_predicted_length": length,
                "converted_leaf_value": value,
                "clipped": clipped,
                "clip_reason": clip_reason,
                "residual_nonzeros": sum(x != 0 for plane in residual for row in plane for x in row),
            })
        return value


@dataclass
class ConstantValueEvaluator:
    """Residual-independent nonterminal control on the MCTS leaf-value scale."""

    value: float = 0.0
    name: str = field(default="constant_value", init=False)
    metrics: ValueMetrics = field(default_factory=ValueMetrics, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("constant value must be finite")

    def reset_metrics(self) -> None:
        self.metrics = ValueMetrics()

    def evaluate(self, residual, remaining_moves, max_moves, target_nonzeros):
        del residual, remaining_moves, max_moves, target_nonzeros
        self.metrics.calls += 1
        self.metrics.predictions.append(self.value)
        return self.value


@dataclass
class NeuralResidualHybridPolicy(InstrumentedPolicy):
    """Use a fixed neural/residual allocation and deduplicate across sources."""

    neural: NeuralActionPolicy
    residual: ResidualInformedPolicy
    neural_fraction: float = 0.5
    exploration_fraction: float = 0.0
    name: str = field(default="neural_residual_hybrid", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)
    last_provenance: dict[RankOneAction, tuple[str, ...]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.neural_fraction <= 1 or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("mixture fractions must be in [0, 1]")
        if self.neural.config.shape != self.residual.shape:
            raise ValueError("hybrid policy component shapes differ")

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.neural.reset_metrics()
        self.residual.reset_metrics()
        self.last_provenance = {}

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        neural_count = min(max_actions, round(max_actions * self.neural_fraction))
        residual_count = max_actions - neural_count
        if self.exploration_fraction and max_actions > 1:
            residual_count = max(residual_count, round(max_actions * self.exploration_fraction))
            neural_count = max_actions - residual_count
        before = (
            self.neural.metrics.attempted, self.neural.metrics.valid, self.neural.metrics.duplicates,
            self.residual.metrics.attempted, self.residual.metrics.valid, self.residual.metrics.duplicates,
        )
        neural = self.neural.propose_actions(residual, remaining_moves, rng, neural_count) if neural_count else ()
        sampled = self.residual.propose_actions(residual, remaining_moves, rng, residual_count) if residual_count else ()
        provenance: dict[RankOneAction, set[str]] = {}
        weighted = []
        for source, proposals, weight in (
            ("neural", neural, self.neural_fraction),
            ("residual", sampled, 1.0 - self.neural_fraction),
        ):
            for item in proposals:
                action = canonicalize_action(item.action)
                provenance.setdefault(action, set()).add(source)
                weighted.append((action, max(weight, 1e-12) * item.prior))
        normalized = _normalize(weighted)
        self.last_provenance = {action: tuple(sorted(values)) for action, values in provenance.items()}
        result = tuple(
            ProposedAction(item.action, item.prior, self.last_provenance[item.action])
            for item in normalized
        )
        self.metrics.attempted += self.neural.metrics.attempted - before[0] + self.residual.metrics.attempted - before[3]
        self.metrics.valid += self.neural.metrics.valid - before[1] + self.residual.metrics.valid - before[4]
        self.metrics.duplicates += self.neural.metrics.duplicates - before[2] + self.residual.metrics.duplicates - before[5]
        self.metrics.duplicates += len(neural) + len(sampled) - len(result)
        self.metrics.accepted += len(result)
        self.metrics.generation_seconds = self.neural.metrics.generation_seconds + self.residual.metrics.generation_seconds
        return result
