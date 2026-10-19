"""Optional Triton execution template for singleton-HBM schedule groups."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_schedule_lowering import FAMILY_MATERIALIZED, ScheduleLoweringPlan
from .tensor_game_schedule_model import ExecutionDag, Schedule
from .tensor_game_triton_runtime import _fp32_gemm_kernel


@triton.jit
def _materialize_form_kernel(x0, x1, x2, x3, dst,
                             s00: tl.constexpr, s01: tl.constexpr,
                             s10: tl.constexpr, s11: tl.constexpr,
                             s20: tl.constexpr, s21: tl.constexpr,
                             s30: tl.constexpr, s31: tl.constexpr,
                             c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr,
                             half: tl.constexpr, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    row, col = offset // half, offset % half
    mask = offset < half * half
    value = tl.zeros((block,), dtype=tl.float32)
    if c0 != 0:
        value += c0 * tl.load(x0 + row * s00 + col * s01, mask=mask, other=0).to(tl.float32)
    if c1 != 0:
        value += c1 * tl.load(x1 + row * s10 + col * s11, mask=mask, other=0).to(tl.float32)
    if c2 != 0:
        value += c2 * tl.load(x2 + row * s20 + col * s21, mask=mask, other=0).to(tl.float32)
    if c3 != 0:
        value += c3 * tl.load(x3 + row * s30 + col * s31, mask=mask, other=0).to(tl.float32)
    tl.store(dst + offset, value.to(tl.float16), mask=mask)


@triton.jit
def _accumulate_kernel(product, previous, dst, coefficient: tl.constexpr,
                       has_previous: tl.constexpr, count: tl.constexpr, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    mask = offset < count
    value = coefficient * tl.load(product + offset, mask=mask, other=0)
    if has_previous:
        value += tl.load(previous + offset, mask=mask, other=0)
    tl.store(dst + offset, value, mask=mask)


@triton.jit
def _output_kernel(source, output, half: tl.constexpr, output_block: tl.constexpr,
                   block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    row, col = offset // half, offset % half
    mask = offset < half * half
    value = tl.load(source + offset, mask=mask, other=0)
    dst_row = row + (output_block // 2) * half
    dst_col = col + (output_block % 2) * half
    tl.store(output + dst_row * (2 * half) + dst_col, value, mask=mask)


def allocate_materialized(dag: ExecutionDag):
    half = dag.config.matrix_size // 2
    values = {node.node_id: torch.empty((half, half), device="cuda",
                                        dtype=torch.float16 if node.operation == "LINEAR_FORM" else torch.float32)
              for node in dag.nodes if node.operation in {"LINEAR_FORM", "MATMUL", "ACCUMULATE"}}
    return {"values": values, "output": torch.empty((2 * half, 2 * half), device="cuda", dtype=torch.float16)}


def execute_materialized(dag: ExecutionDag, schedule: Schedule, plan: ScheduleLoweringPlan,
                         left, right, workspace, capture: list | None = None):
    """Execute the schedule's singleton-HBM groups in their specified order."""
    if plan.family_id != FAMILY_MATERIALIZED or plan.mapping_status != "EXACT_MAPPING":
        raise ValueError("only exact MATERIALIZED_DAG plans can use this template")
    if plan.schedule_hash != schedule.schedule_hash or plan.decomposition_id != dag.decomposition_id:
        raise ValueError("schedule or decomposition provenance mismatch")
    size = dag.config.matrix_size
    if (left.shape != (size, size) or right.shape != (size, size) or left.dtype != torch.float16
            or right.dtype != torch.float16 or not left.is_cuda or not right.is_cuda):
        raise ValueError("template requires matching square CUDA FP16 inputs")
    half = size // 2
    blocks = {f"{side}{index}": matrix[(index // 2) * half:(index // 2 + 1) * half,
                                          (index % 2) * half:(index % 2 + 1) * half]
              for side, matrix in (("A", left), ("B", right)) for index in range(4)}
    values = {**blocks, **workspace["values"]}
    nodes = {node.node_id: node for node in dag.nodes}
    if capture is not None:
        capture.clear()
    for group in schedule.groups:
        if len(group.operations) != 1 or group.operations[0].location != "HBM":
            raise ValueError("schedule is outside singleton-HBM template")
        node = nodes[group.operations[0].node_id]
        if node.operation == "LINEAR_FORM":
            side = node.node_id[0]
            source = [blocks[f"{side}{index}"] for index in range(4)]
            compiled = _materialize_form_kernel[(triton.cdiv(half * half, 256),)](
                *source, values[node.node_id], *(stride for tensor in source for stride in tensor.stride()),
                *node.coefficients, half, block=256, num_warps=4)
        elif node.operation == "MATMUL":
            a, b = (values[key] for key in node.dependencies)
            c = values[node.node_id]
            compiled = _fp32_gemm_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                a, b, c, half, half, half, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), block_m=32, block_n=32, block_k=16,
                input_precision="ieee", num_warps=4, num_stages=3)
        elif node.operation == "ACCUMULATE":
            product = values[node.dependencies[-1]]
            previous = values[node.dependencies[0]] if len(node.dependencies) == 2 else product
            compiled = _accumulate_kernel[(triton.cdiv(half * half, 256),)](
                product, previous, values[node.node_id], node.coefficients[0], len(node.dependencies) == 2,
                half * half, block=256, num_warps=4)
        elif node.operation == "OUTPUT":
            compiled = _output_kernel[(triton.cdiv(half * half, 256),)](
                values[node.dependencies[0]], workspace["output"], half, node.output_block,
                block=256, num_warps=4)
        else:
            raise ValueError(f"unsupported node operation: {node.operation}")
        if capture is not None:
            capture.append({"group_id": group.group_id, "role": node.operation,
                            "node_id": node.node_id, "kernel": compiled.name,
                            "compiled": compiled})
    return workspace["output"]
