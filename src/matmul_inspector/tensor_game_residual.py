"""Stable production API for Tensor Game residual candidate providers."""

from .tensor_game_bounded_residual import (
    BoundedResidualPolicy,
    ExhaustiveResidualPolicy,
    create_residual_candidate_provider,
)

__all__ = (
    "BoundedResidualPolicy",
    "ExhaustiveResidualPolicy",
    "create_residual_candidate_provider",
)
