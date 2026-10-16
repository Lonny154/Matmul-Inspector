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

DOT_PRECISIONS = ("ieee", "tf32x3")
EXECUTION_STRATEGIES = ("unfused", "fused_input")
RECONSTRUCTION_STRATEGIES = ("baseline", "fused")


def validate_dot_precision(precision: str) -> str:
    if precision not in DOT_PRECISIONS:
        raise ValueError(f"unsupported dot precision {precision!r}; expected one of {DOT_PRECISIONS}")
    return precision


def validate_execution_strategy(strategy: str) -> str:
    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError(
            f"unsupported execution strategy {strategy!r}; expected one of {EXECUTION_STRATEGIES}"
        )
    return strategy


def validate_reconstruction_strategy(reconstruction: str) -> str:
    if reconstruction not in RECONSTRUCTION_STRATEGIES:
        raise ValueError(
            f"unsupported reconstruction strategy {reconstruction!r}; "
            f"expected one of {RECONSTRUCTION_STRATEGIES}"
        )
    return reconstruction


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

    def execute(
        self, left, right, precision: str = "ieee", strategy: str = "unfused",
        reconstruction: str = "baseline",
    ):
        """Execute on contiguous square CUDA FP32 tensors via Triton."""
        from .tensor_game_triton_runtime import execute_lowering

        return execute_lowering(self, left, right, precision, strategy, reconstruction)


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


def lowering_manifest(
    lowering: BilinearLowering,
    precision: str = "ieee",
    strategy: str = "unfused",
    reconstruction: str = "baseline",
) -> dict[str, object]:
    """Return deterministic JSON-ready coefficients and lowering metadata."""
    precision = validate_dot_precision(precision)
    strategy = validate_execution_strategy(strategy)
    reconstruction = validate_reconstruction_strategy(reconstruction)
    metadata = asdict(lowering.metadata)
    metadata["triton_dot_input_precision"] = precision
    return {
        "schema": "tensor_game_triton_lowering.v2",
        "metadata": metadata,
        "selected_dot_input_precision": precision,
        "supported_dot_input_precisions": list(DOT_PRECISIONS),
        "execution_strategy": strategy,
        "supported_execution_strategies": list(EXECUTION_STRATEGIES),
        "reconstruction_strategy": reconstruction,
        "supported_reconstruction_strategies": list(RECONSTRUCTION_STRATEGIES),
        "strategy_metadata": {
            "preparation_kernel_launches": (
                2 * lowering.metadata.rank + lowering.metadata.execution_preparation_updates
                if strategy == "unfused" else 0
            ),
            "gemm_kernel_launches": lowering.metadata.block_gemms,
            "reconstruction_kernel_launches": lowering.metadata.execution_reconstruction_updates + 4 if reconstruction == "baseline" else 1,
            "output_assembly_copy_launches": 4 if reconstruction == "baseline" else 0,
            "intermediate_output_block_buffers": 4 if reconstruction == "baseline" else 0,
            "total_device_kernel_launches_estimate": (
                (2 * lowering.metadata.rank + lowering.metadata.execution_preparation_updates)
                if strategy == "unfused" else 0
            ) + lowering.metadata.block_gemms + (
                lowering.metadata.execution_reconstruction_updates + 8
                if reconstruction == "baseline" else 1
            ),
            "full_size_input_combination_buffers": 2 * lowering.metadata.rank if strategy == "unfused" else 0,
            "rank_product_buffers": lowering.metadata.rank,
            "input_combinations_formed_in_gemm_load_path": strategy == "fused_input",
        },
        "actions": [
            {"u": list(action.u), "v": list(action.v), "w": list(action.w)}
            for action in lowering.actions
        ],
    }
