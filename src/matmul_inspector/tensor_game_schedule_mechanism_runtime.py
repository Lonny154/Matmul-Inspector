"""Stage probes that launch the unchanged HA-Schedule-3 Triton kernels."""

from __future__ import annotations

import triton
import triton.language as tl

from .tensor_game_schedule_gpu_runtime import _materialize_u_form, _shared_form_gemm
from .tensor_game_triton_runtime import fused_input_gemm, fused_reconstruction
from .tensor_game_schedule_mechanism_study import RECOMPUTED, STORED


@triton.jit
def _diagnostic_tiny_launch(flag):
    # Separate scratch word. Never touches a matrix input, product, or output.
    tl.store(flag, tl.load(flag))


def _validate(lowering, workspace, mode: str) -> None:
    if mode not in (STORED, RECOMPUTED):
        raise ValueError(f"unknown rank-8 mechanism mode: {mode}")
    if len(lowering.actions) != 8 or lowering.actions[0].u != lowering.actions[1].u:
        raise ValueError("mechanism probe requires the selected shared-form rank-8 factors")
    if mode == STORED and workspace["prepared"] is None:
        raise ValueError("stored consumer needs its prepared form")


def run_materialization(lowering, left, workspace, capture: dict | None = None):
    _validate(lowering, workspace, STORED)
    half = left.shape[0] // 2
    compiled = _materialize_u_form[(triton.cdiv(half * half, 256),)](
        left, workspace["prepared"], half, left.stride(0), left.stride(1),
        *lowering.actions[0].u, block=256, num_warps=4)
    if capture is not None:
        capture["materialization"] = compiled
    return workspace["prepared"]


def run_consumer_product(lowering, left, right, workspace, mode: str, index: int,
                         capture: dict | None = None):
    _validate(lowering, workspace, mode)
    if index not in (0, 1):
        raise ValueError("only the first two products consume the shared form")
    half = left.shape[0] // 2
    action = lowering.actions[index]
    form = workspace["prepared"] if mode == STORED else left
    compiled = _shared_form_gemm[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
        left, right, form, workspace["products"][index], half,
        left.stride(0), left.stride(1), right.stride(0), right.stride(1),
        *action.u, *action.v, use_prepared=mode == STORED,
        block_m=32, block_n=32, block_k=16, num_warps=4, num_stages=3)
    if capture is not None:
        capture[f"consumer_{index}"] = compiled
    return workspace["products"][index]


def run_consumers(lowering, left, right, workspace, mode: str,
                  capture: dict | None = None):
    for index in (0, 1):
        run_consumer_product(lowering, left, right, workspace, mode, index, capture)
    return workspace["products"][:2]


def run_common_gemms(lowering, left, right, workspace, capture: dict | None = None):
    for index in range(2, 8):
        action = lowering.actions[index]
        handle = {} if capture is not None else None
        fused_input_gemm(left, right, action.u, action.v,
                         output=workspace["products"][index], capture=handle)
        if capture is not None:
            capture[f"common_gemm_{index}"] = handle["compiled"]
    return workspace["products"][2:]


def run_reconstruction(lowering, left, workspace, capture: dict | None = None):
    handle = {} if capture is not None else None
    output = fused_reconstruction(workspace["products"], lowering.actions,
                                  left.shape[0], output=workspace["output"], capture=handle)
    if capture is not None:
        capture["reconstruction"] = handle["compiled"]
    return output


def execute_staged(lowering, left, right, workspace, mode: str,
                   capture: dict | None = None):
    """Validation control: exact same kernel order as the production path."""
    _validate(lowering, workspace, mode)
    if mode == STORED:
        run_materialization(lowering, left, workspace, capture)
    run_consumers(lowering, left, right, workspace, mode, capture)
    run_common_gemms(lowering, left, right, workspace, capture)
    return run_reconstruction(lowering, left, workspace, capture)


def run_dummy_launch(scratch, capture: dict | None = None):
    compiled = _diagnostic_tiny_launch[(1,)](scratch, num_warps=4)
    if capture is not None:
        capture["diagnostic_dummy"] = compiled
    return scratch
