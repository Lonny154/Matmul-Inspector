"""Experimental final-ranking supervision for the TensorGame decoder."""

from __future__ import annotations

from typing import Sequence

from .tensor_game import RankOneAction, validate_action
from .tensor_game_learning import rank_complete_actions
from .tensor_game_prefix_coverage import COEFFICIENTS, sign_forms
from .tensor_game_search import canonicalize_action


def complete_action_score(logits, action: RankOneAction):
    """Best sign-equivalent complete-sequence score under decoder semantics."""
    import torch

    validate_action(action, (4, 4, 4), frozenset(COEFFICIENTS))
    logp = torch.log_softmax(logits, dim=-1)
    scores = []
    for form in sign_forms(canonicalize_action(action)):
        indices = torch.tensor(
            [COEFFICIENTS.index(token) for token in form], device=logits.device
        )
        scores.append(logp.gather(1, indices[:, None]).sum())
    return torch.stack(scores).max()


def select_hard_negative(logits, *, requested_actions: int = 12) -> RankOneAction:
    """Select the lowest-ranked accepted proposal from an exact detached decode."""
    ranked = rank_complete_actions(logits.detach(), (4, 4, 4), requested_actions)
    if not ranked:
        raise ValueError("decoder returned no valid negative candidates")
    return ranked[-1][0]


def final_ranking_loss(
    logits,
    positive_actions,
    positive_mask,
    *,
    requested_actions: int = 12,
    margin: float = 0.1,
):
    """State-level hinge against the current final accepted-action cutoff.

    The best verified positive represents the state-level goal of protecting at
    least one successful continuation. Decoder selection is detached. The
    selected positive and hard-negative scores remain differentiable.
    """
    import torch

    if margin < 0 or positive_actions.shape[:2] != positive_mask.shape:
        raise ValueError("invalid final-ranking loss inputs")
    if torch.any(positive_mask.sum(1) == 0):
        raise ValueError("final-ranking loss requires nonempty positives")
    losses = []
    for batch_index in range(logits.shape[0]):
        positives: list[RankOneAction] = []
        for action_index in torch.nonzero(
            positive_mask[batch_index], as_tuple=False
        ).flatten().tolist():
            values = tuple(int(x) for x in positive_actions[batch_index, action_index])
            positives.append(RankOneAction(values[:4], values[4:8], values[8:]))
        negative = select_hard_negative(
            logits[batch_index], requested_actions=requested_actions
        )
        positive_score = torch.stack(
            [complete_action_score(logits[batch_index], action) for action in positives]
        ).max()
        negative_score = complete_action_score(logits[batch_index], negative)
        losses.append(torch.relu(margin + negative_score - positive_score))
    return torch.stack(losses).mean()


def score_actions(logits, actions: Sequence[RankOneAction]):
    """Convenience helper for diagnostics without changing score semantics."""
    return [complete_action_score(logits, action) for action in actions]
