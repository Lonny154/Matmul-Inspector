"""Experiment-only prefix tracing and loss aligned with the TensorGame decoder."""

from __future__ import annotations

from typing import Any, Sequence

from .tensor_game import RankOneAction

COEFFICIENTS = (-1, 0, 1)


def sign_forms(action: RankOneAction) -> tuple[tuple[int, ...], ...]:
    u, v, w = action.u, action.v, action.w
    return tuple(
        dict.fromkeys(
            (
                u + v + w,
                tuple(-x for x in u) + tuple(-x for x in v) + w,
                tuple(-x for x in u) + v + tuple(-x for x in w),
                u + tuple(-x for x in v) + tuple(-x for x in w),
            )
        )
    )


def _advance_beam(log_probabilities, beam, position: int, width: int):
    candidates = [
        (prefix + (coefficient,), score + log_probabilities[position, class_index])
        for prefix, score in beam
        for class_index, coefficient in enumerate(COEFFICIENTS)
    ]
    candidates.sort(key=lambda item: (-float(item[1].detach()), item[0]))
    return candidates[:width], candidates


def trace_positive_prefixes(
    logits,
    actions: Sequence[RankOneAction],
    *,
    requested_actions: int = 12,
) -> list[dict[str, Any]]:
    """Trace sign-equivalent positive prefixes through exact production beam stages."""
    import torch

    if logits.ndim != 2 or logits.shape[1] != 3 or requested_actions <= 0:
        raise ValueError("invalid decoder trace inputs")
    width = max(256, requested_actions * 32)
    logp = torch.log_softmax(logits.detach().cpu(), dim=-1)
    beam = [((), torch.zeros((), dtype=logp.dtype))]
    forms = [sign_forms(action) for action in actions]
    first_pruned = [None] * len(actions)
    rows = []
    for position in range(logp.shape[0]):
        beam, candidates = _advance_beam(logp, beam, position, width)
        exact_ranks = {prefix: rank for rank, (prefix, _) in enumerate(candidates, 1)}
        surviving = {prefix: score for prefix, score in beam}
        cutoff = beam[-1][1]
        for action_index, action_forms in enumerate(forms):
            prefixes = tuple(dict.fromkeys(form[: position + 1] for form in action_forms))
            ranked = [(exact_ranks[prefix], prefix) for prefix in prefixes if prefix in exact_ranks]
            best_rank, best_prefix = min(ranked) if ranked else (None, prefixes[0])
            survivors = [(score, prefix) for prefix, score in surviving.items() if prefix in prefixes]
            survives = bool(survivors)
            if not survives and first_pruned[action_index] is None:
                first_pruned[action_index] = position + 1
            if survivors:
                positive_score, best_prefix = max(
                    survivors, key=lambda item: (float(item[0]), tuple(-x for x in item[1]))
                )
            else:
                positive_score = sum(
                    logp[token_position, COEFFICIENTS.index(token)]
                    for token_position, token in enumerate(best_prefix)
                )
            rows.append(
                {
                    "action_index": action_index,
                    "prefix_length": position + 1,
                    "prefix_tokens": best_prefix,
                    "cumulative_score": float(positive_score),
                    "cutoff_score": float(cutoff),
                    "score_margin": float(positive_score - cutoff),
                    "prefix_rank_in_expanded_candidates": best_rank,
                    "expanded_candidates": len(candidates),
                    "beam_width": width,
                    "survives": survives,
                    "first_pruning_position": first_pruned[action_index],
                }
            )
    return rows


def prefix_survival_loss(
    logits,
    positive_actions,
    positive_mask,
    *,
    requested_actions: int = 12,
    margin: float = 0.1,
    compiled_forms=None,
):
    """Hinge loss through each positive's first failed width-384 beam stage.

    Beam membership is selected from detached scores with the production
    lexicographic tie-break. Selected cumulative scores and cutoff scores remain
    attached, making each fixed membership region differentiable.
    """
    import torch

    if margin < 0 or positive_actions.shape[:2] != positive_mask.shape:
        raise ValueError("invalid prefix loss inputs")
    if torch.any(positive_mask.sum(1) == 0):
        raise ValueError("prefix loss requires nonempty positives")
    width = max(256, requested_actions * 32)
    logp = torch.log_softmax(logits, dim=-1)
    example_losses = []
    if compiled_forms is None:
        compiled_forms = compile_positive_sign_forms(positive_actions, positive_mask)
    if len(compiled_forms) != logits.shape[0]:
        raise ValueError("compiled prefix targets do not match batch size")
    for batch_index in range(logits.shape[0]):
        forms_by_action = compiled_forms[batch_index]
        beam = [((), logp[batch_index].sum() * 0)]
        active = [True] * len(forms_by_action)
        penalties = []
        seen_prefixes = set()
        for position in range(logits.shape[1]):
            beam, _ = _advance_beam(logp[batch_index], beam, position, width)
            if len(beam) < width:
                continue
            cutoff = beam[-1][1]
            surviving = {prefix for prefix, _ in beam}
            for action_index, action_forms in enumerate(forms_by_action):
                if not active[action_index]:
                    continue
                prefixes = tuple(dict.fromkeys(form[: position + 1] for form in action_forms))
                key = (position + 1, prefixes)
                if key not in seen_prefixes:
                    scores = [
                        sum(
                            logp[batch_index, token_position, COEFFICIENTS.index(token)]
                            for token_position, token in enumerate(prefix)
                        )
                        for prefix in prefixes
                    ]
                    positive_score = torch.stack(scores).max()
                    penalties.append(torch.relu(margin + cutoff - positive_score))
                    seen_prefixes.add(key)
                if not any(prefix in surviving for prefix in prefixes):
                    active[action_index] = False
        example_losses.append(torch.stack(penalties).mean() if penalties else logits[batch_index].sum() * 0)
    return torch.stack(example_losses).mean()


def compile_positive_sign_forms(positive_actions, positive_mask):
    """Cache immutable sign-equivalent teacher encodings for repeated updates."""
    if positive_actions.shape[:2] != positive_mask.shape:
        raise ValueError("invalid prefix target inputs")
    compiled = []
    for batch_index in range(positive_actions.shape[0]):
        row = []
        for action_index in positive_mask[batch_index].nonzero(as_tuple=False).flatten().tolist():
            values = tuple(int(x) for x in positive_actions[batch_index, action_index].tolist())
            row.append(sign_forms(RankOneAction(values[:4], values[4:8], values[8:])))
        compiled.append(tuple(row))
    return tuple(compiled)
