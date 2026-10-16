"""Validated 2x2-block Tensor Game lowering contract with a lazy Triton backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    sum_actions,
    validate_action,
)
from .tensor_game_hardware_cost import decomposition_cost


@dataclass(frozen=True)
class LoweringMetadata:
    name: str
    block_grid: tuple[int, int, int]
    flattening: str
    rank: int
    block_gemms: int
    preparation_additions: int
    reconstruction_additions: int
    coefficient_scalings: int
    execution_preparation_updates: int
    execution_reconstruction_updates: int
    persistent_materializations: int
    peak_live_block_temporaries: int
    input_dtype: str
    output_dtype: str
    accumulation_dtype: str
    triton_dot_input_precision: str


@dataclass(frozen=True)
class BilinearLowering:
    """A validated bilinear algorithm ready for 2x2-block execution."""

    name: str
    actions: tuple[RankOneAction, ...]
    metadata: LoweringMetadata

    def execute(self, left, right):
        """Execute on contiguous square CUDA FP32 tensors via Triton."""
        from .tensor_game_triton_runtime import execute_lowering

        return execute_lowering(self, left, right)


def create_lowering(name: str, actions: Sequence[RankOneAction]) -> BilinearLowering:
    """Validate an exact 2x2 decomposition and return immutable lowering data."""
    frozen = tuple(actions)
    if not frozen:
        raise ValueError("decomposition must contain at least one action")
    shape = (4, 4, 4)
    for action in frozen:
        validate_action(action, shape, allowed_coefficients=None)
    target = matrix_multiplication_tensor(2, 2, 2)
    if sum_actions(frozen, shape) != target:
        raise ValueError("decomposition does not exactly reconstruct 2x2 matrix multiplication")
    cost = decomposition_cost(frozen, target)
    # Sequential execution materializes two combinations and one product per
    # rank term over time, four output accumulators, and one assembled output.
    metadata = LoweringMetadata(
        name=name,
        block_grid=(2, 2, 2),
        flattening="row-major blocks: index = block_row * 2 + block_column",
        rank=len(frozen),
        block_gemms=len(frozen),
        preparation_additions=cost.pre_additions,
        reconstruction_additions=cost.post_additions,
        coefficient_scalings=cost.coefficient_scalings,
        execution_preparation_updates=sum(
            value != 0 for action in frozen for value in action.u + action.v
        ),
        execution_reconstruction_updates=sum(
            value != 0 for action in frozen for value in action.w
        ),
        persistent_materializations=3 * len(frozen) + 5,
        peak_live_block_temporaries=len(frozen) + 5,
        input_dtype="float32",
        output_dtype="float32",
        accumulation_dtype="float32",
        triton_dot_input_precision="ieee",
    )
    return BilinearLowering(name=name, actions=frozen, metadata=metadata)


def reference_lowering(name: str) -> BilinearLowering:
    """Create one of the two checked reference algorithms."""
    if name == "conventional":
        return create_lowering(name, standard_2x2_decomposition())
    if name == "strassen":
        return create_lowering(name, strassen_2x2_decomposition())
    raise ValueError("unknown reference lowering; expected 'conventional' or 'strassen'")


def lowering_manifest(lowering: BilinearLowering) -> dict[str, object]:
    """Return deterministic JSON-ready coefficients and lowering metadata."""
    return {
        "schema": "tensor_game_triton_lowering.v1",
        "metadata": asdict(lowering.metadata),
        "actions": [
            {"u": list(action.u), "v": list(action.v), "w": list(action.w)}
            for action in lowering.actions
        ],
    }
