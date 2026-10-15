"""Diagnostic utilities for TensorGame policy-update and factorization audits."""

from __future__ import annotations

import hashlib
from typing import Iterable

from .tensor_game import RankOneAction
from .tensor_game_prefix_coverage import COEFFICIENTS


def state_dict_digest(model) -> str:
    """Hash tensor names, dtypes, shapes, and bytes for restoration checks."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def flattened_gradients(parameters: Iterable, loss):
    """Return a detached flattened gradient without mutating ``.grad`` fields."""
    import torch

    params = tuple(parameters)
    gradients = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    return torch.cat(
        [torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p, g in zip(params, gradients)]
    )


def gradient_cosine(first, second) -> float:
    import torch

    denominator = first.norm() * second.norm()
    return float(torch.dot(first, second) / denominator) if denominator else 0.0


def component_log_scores(logits, action: RankOneAction) -> tuple[float, float, float]:
    """Teacher-forced canonical u, v, and w log scores."""
    import torch

    logp = torch.log_softmax(logits.detach(), dim=-1)
    values = action.u + action.v + action.w
    indices = torch.tensor([COEFFICIENTS.index(x) for x in values], device=logits.device)
    token_scores = logp.gather(1, indices[:, None]).squeeze(1)
    return tuple(float(token_scores[start : start + 4].sum()) for start in (0, 4, 8))


def component_ranks(logits, action: RankOneAction) -> tuple[int, int, int]:
    """Rank each canonical nonzero four-token factor among 80 possibilities."""
    import itertools
    import torch

    logp = torch.log_softmax(logits.detach().cpu(), dim=-1)
    ranks = []
    for offset, target in zip((0, 4, 8), (action.u, action.v, action.w)):
        values = [x for x in itertools.product(COEFFICIENTS, repeat=4) if any(x)]
        scored = []
        for vector in values:
            score = sum(float(logp[offset + i, COEFFICIENTS.index(token)]) for i, token in enumerate(vector))
            scored.append((vector, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranks.append(next(i for i, (vector, _) in enumerate(scored, 1) if vector == tuple(target)))
    return tuple(ranks)


def counterfactual_segment_logits(trained_logits, frozen_logits, segment: str):
    """Replace one synthetic action segment with frozen-policy logits."""
    if segment not in {"u", "v", "w", "all"}:
        raise ValueError("segment must be u, v, w, or all")
    result = trained_logits.clone()
    bounds = {"u": (0, 4), "v": (4, 8), "w": (8, 12), "all": (0, 12)}[segment]
    result[bounds[0] : bounds[1]] = frozen_logits[bounds[0] : bounds[1]]
    return result


def architecture_semantics() -> dict:
    return {
        "encoder": "two-layer residual-conditioned MLP with GELU",
        "policy_output_shape": [12, 3],
        "coefficient_classes": [-1, 0, 1],
        "factorization": "conditionally independent coefficients given shared residual features",
        "autoregressive": False,
        "token_conditioning": False,
        "complete_sequence_score": "sum of 12 token log-softmax scores",
        "sign_equivalence": "four factor-sign forms scored; decoder canonicalizes complete actions",
        "decoder": "width max(256, requested_actions*32) cumulative-score beam",
    }
