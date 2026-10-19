"""Optional Triton runtime for bounded intentional HBM value lifetimes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_schedule_hbm_lowering import FAMILY_HBM, HBMBackedPlan, validate_hbm_plan
from .tensor_game_schedule_lowering_runtime import (
    _accumulate_kernel, _materialize_form_kernel, _output_kernel,
)
from .tensor_game_schedule_mixed_runtime import _operand, _source_coefficients
from .tensor_game_schedule_model import ExecutionDag, Schedule
from .tensor_game_triton_runtime import _fp32_gemm_kernel


@triton.jit
def _write_hbm_target(product, previous, partial, output, row, col, mask,
                      half: tl.constexpr, coefficient: tl.constexpr,
                      has_previous: tl.constexpr, write_partial: tl.constexpr,
                      write_final: tl.constexpr, output_block: tl.constexpr):
    value = coefficient * product
    if has_previous:
        value += tl.load(previous + row * half + col, mask=mask, other=0)
    if write_partial:
        tl.store(partial + row * half + col, value, mask=mask)
    if write_final:
        dst_row = row + (output_block // 2) * half
        dst_col = col + (output_block % 2) * half
        tl.store(output + dst_row * (2 * half) + dst_col, value, mask=mask)


@triton.jit
def _hbm_gemm_group_kernel(
        left, right, prepared_a, prepared_b, form_a, form_b, product_buffer,
        previous0, previous1, partial0, partial1, output,
        half: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
        stride_b0: tl.constexpr, stride_b1: tl.constexpr,
        u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
        v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
        prepared_u: tl.constexpr, prepared_v: tl.constexpr,
        write_form_a: tl.constexpr, write_form_b: tl.constexpr,
        write_product: tl.constexpr,
        has_target0: tl.constexpr, coefficient0: tl.constexpr,
        has_previous0: tl.constexpr, write_partial0: tl.constexpr,
        write_final0: tl.constexpr, output_block0: tl.constexpr,
        has_target1: tl.constexpr, coefficient1: tl.constexpr,
        has_previous1: tl.constexpr, write_partial1: tl.constexpr,
        write_final1: tl.constexpr, output_block1: tl.constexpr,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    row = pid_m * block_m + tl.arange(0, block_m)
    col = pid_n * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start in range(0, half, block_k):
        a_mask = (row[:, None] < half) & (start + ks[None, :] < half)
        b_mask = (start + ks[:, None] < half) & (col[None, :] < half)
        a = _operand(left, prepared_a, row[:, None], start + ks[None, :], a_mask,
                     half, stride_a0, stride_a1, u0, u1, u2, u3, prepared_u)
        b = _operand(right, prepared_b, start + ks[:, None], col[None, :], b_mask,
                     half, stride_b0, stride_b1, v0, v1, v2, v3, prepared_v)
        # One CTA owns each form tile. Other GEMM CTAs compute the same input
        # locally, avoiding any unsynchronized read-after-write within a grid.
        if write_form_a and pid_n == 0:
            tl.store(form_a + row[:, None] * half + start + ks[None, :], a, mask=a_mask)
        if write_form_b and pid_m == 0:
            tl.store(form_b + (start + ks[:, None]) * half + col[None, :], b, mask=b_mask)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee", out_dtype=tl.float32)
    mask = (row[:, None] < half) & (col[None, :] < half)
    if write_product:
        tl.store(product_buffer + row[:, None] * half + col[None, :], accumulator, mask=mask)
    if has_target0:
        _write_hbm_target(accumulator, previous0, partial0, output, row[:, None], col[None, :], mask,
                          half, coefficient0, has_previous0, write_partial0, write_final0, output_block0)
    if has_target1:
        _write_hbm_target(accumulator, previous1, partial1, output, row[:, None], col[None, :], mask,
                          half, coefficient1, has_previous1, write_partial1, write_final1, output_block1)


@triton.jit
def _hbm_accumulate_output_kernel(product, previous, partial, output,
                                  half: tl.constexpr, coefficient: tl.constexpr,
                                  has_previous: tl.constexpr, write_partial: tl.constexpr,
                                  output_block: tl.constexpr, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    row, col = offset // half, offset % half
    mask = offset < half * half
    value = coefficient * tl.load(product + offset, mask=mask, other=0)
    if has_previous:
        value += tl.load(previous + offset, mask=mask, other=0)
    if write_partial:
        tl.store(partial + offset, value, mask=mask)
    dst_row = row + (output_block // 2) * half
    dst_col = col + (output_block % 2) * half
    tl.store(output + dst_row * (2 * half) + dst_col, value, mask=mask)


def allocate_hbm(dag: ExecutionDag, schedule: Schedule, plan: HBMBackedPlan):
    """Allocate one stable buffer per intended HBM intermediate; no reuse."""
    validate_hbm_plan(dag, schedule, plan)
    half = dag.config.matrix_size // 2
    values = {buffer.value_id: torch.empty((half, half), device="cuda",
                                           dtype=torch.float16 if buffer.dtype == "fp16" else torch.float32)
              for buffer in plan.buffer_plan if buffer.output_block is None}
    output = torch.empty((2 * half, 2 * half), device="cuda", dtype=torch.float16)
    return {"values": values, "output": output}


def audit_hbm_workspace(dag: ExecutionDag, schedule: Schedule, plan: HBMBackedPlan, workspace) -> dict:
    """Match every planned logical HBM buffer to its allocated runtime buffer."""
    planning = validate_hbm_plan(dag, schedule, plan)
    expected = {buffer.value_id for buffer in plan.buffer_plan if buffer.output_block is None}
    actual = set(workspace["values"])
    if expected != actual:
        raise ValueError(f"logical HBM buffers differ: missing={expected - actual}, extra={actual - expected}")
    half = dag.config.matrix_size // 2
    for buffer in plan.buffer_plan:
        if buffer.output_block is None:
            tensor = workspace["values"][buffer.value_id]
            expected_dtype = torch.float16 if buffer.dtype == "fp16" else torch.float32
            if tensor.shape != (half, half) or tensor.dtype != expected_dtype or not tensor.is_cuda:
                raise ValueError(f"invalid buffer for {buffer.value_id}")
    if workspace["output"].shape != (2 * half, 2 * half) or workspace["output"].dtype != torch.float16:
        raise ValueError("invalid final-output buffer")
    return {**planning, "actual_hbm_values": len(actual),
            "planned_hbm_value_ids": sorted(expected), "actual_hbm_value_ids": sorted(actual),
            "status": "PASS"}


def execute_hbm(dag: ExecutionDag, schedule: Schedule, plan: HBMBackedPlan,
                left, right, workspace, capture: list | None = None):
    """Execute the bounded plan in group order on the current CUDA stream."""
    base = plan.schedule_plan
    if base.family_id != FAMILY_HBM or base.mapping_status != "EXACT_MAPPING":
        raise ValueError("only exact HBM-backed plans can use this runtime")
    size = dag.config.matrix_size
    if (left.shape != (size, size) or right.shape != (size, size) or left.dtype != torch.float16 or
            right.dtype != torch.float16 or not left.is_cuda or not right.is_cuda):
        raise ValueError("HBM template requires square CUDA FP16 inputs")
    audit_hbm_workspace(dag, schedule, plan, workspace)
    half = size // 2
    blocks = {f"{side}{index}": matrix[(index // 2) * half:(index // 2 + 1) * half,
                                          (index % 2) * half:(index % 2 + 1) * half]
              for side, matrix in (("A", left), ("B", right)) for index in range(4)}
    values = {**blocks, **workspace["values"]}
    nodes = {node.node_id: node for node in dag.nodes}
    if capture is not None:
        capture.clear()
    for launch in base.launch_sequence:
        role = launch["role"]
        if role == FAMILY_HBM:
            source_a, source_b = launch["input_sources"]
            mode_a, mode_b = launch["input_source_modes"]
            targets = launch["reconstruction_targets"]
            target0 = targets[0] if targets else None
            target1 = targets[1] if len(targets) == 2 else None
            def prior(target):
                return values[target["previous_id"]] if target and target["previous_id"] else left
            def partial(target):
                return values[target["accumulation_id"]] if target and target["write_partial"] else left
            compiled = _hbm_gemm_group_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                left, right,
                values[source_a] if mode_a == "HBM_FORM" else left,
                values[source_b] if mode_b == "HBM_FORM" else right,
                values[source_a] if source_a in launch["form_writes"] else left,
                values[source_b] if source_b in launch["form_writes"] else right,
                values[launch["fused_gemm_id"]] if launch["product_write"] else left,
                prior(target0), prior(target1), partial(target0), partial(target1), workspace["output"],
                half, *left.stride(), *right.stride(),
                *_source_coefficients(nodes, source_a), *_source_coefficients(nodes, source_b),
                mode_a == "HBM_FORM", mode_b == "HBM_FORM",
                source_a in launch["form_writes"], source_b in launch["form_writes"],
                launch["product_write"],
                bool(target0), target0["coefficient"] if target0 else 0,
                bool(target0 and target0["previous_id"]), target0["write_partial"] if target0 else False,
                target0["write_final"] if target0 else False, target0["output_block"] if target0 else 0,
                bool(target1), target1["coefficient"] if target1 else 0,
                bool(target1 and target1["previous_id"]), target1["write_partial"] if target1 else False,
                target1["write_final"] if target1 else False, target1["output_block"] if target1 else 0,
                block_m=32, block_n=32, block_k=16, num_warps=4, num_stages=3)
        elif role == "HBM_FINALIZE":
            target = launch["target"]
            compiled = _hbm_accumulate_output_kernel[(triton.cdiv(half * half, 256),)](
                values[target["product_id"]],
                values[target["previous_id"]] if target["previous_id"] else left,
                values[target["accumulation_id"]] if target["write_partial"] else left,
                workspace["output"], half, target["coefficient"], bool(target["previous_id"]),
                target["write_partial"], target["output_block"], block=256, num_warps=4)
        else:
            node = nodes[launch["node_id"]]
            if role == "LINEAR_FORM":
                sources = [blocks[f"{node.node_id[0]}{index}"] for index in range(4)]
                compiled = _materialize_form_kernel[(triton.cdiv(half * half, 256),)](
                    *sources, values[node.node_id], *(stride for tensor in sources for stride in tensor.stride()),
                    *node.coefficients, half, block=256, num_warps=4)
            elif role == "MATMUL":
                a, b = (values[key] for key in node.dependencies)
                c = values[node.node_id]
                compiled = _fp32_gemm_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                    a, b, c, half, half, half, *a.stride(), *b.stride(), *c.stride(),
                    block_m=32, block_n=32, block_k=16,
                    input_precision="ieee", num_warps=4, num_stages=3)
            elif role == "ACCUMULATE":
                product = values[node.dependencies[-1]]
                previous = values[node.dependencies[0]] if len(node.dependencies) == 2 else product
                compiled = _accumulate_kernel[(triton.cdiv(half * half, 256),)](
                    product, previous, values[node.node_id], node.coefficients[0],
                    len(node.dependencies) == 2, half * half, block=256, num_warps=4)
            elif role == "OUTPUT":
                compiled = _output_kernel[(triton.cdiv(half * half, 256),)](
                    values[node.dependencies[0]], workspace["output"], half, node.output_block,
                    block=256, num_warps=4)
            else:
                raise ValueError(f"unsupported launch role {role}")
        if capture is not None:
            capture.append({"group_id": launch["group_id"], "role": role,
                            "node_id": launch["node_id"], "kernel": compiled.name, "compiled": compiled})
    return workspace["output"]
