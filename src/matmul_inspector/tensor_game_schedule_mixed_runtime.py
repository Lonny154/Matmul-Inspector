"""Optional CUDA/Triton execution for bounded mixed GEMM/reconstruction groups."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_schedule_lowering import ScheduleLoweringPlan
from .tensor_game_schedule_mixed_lowering import FAMILY_MIXED
from .tensor_game_schedule_lowering_runtime import (
    _accumulate_kernel, _materialize_form_kernel, _output_kernel,
)
from .tensor_game_schedule_model import ExecutionDag, Schedule
from .tensor_game_triton_runtime import _fp32_gemm_kernel


@triton.jit
def _operand(full, prepared, row, col, mask, half: tl.constexpr,
             stride0: tl.constexpr, stride1: tl.constexpr,
             c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr,
             use_prepared: tl.constexpr):
    if use_prepared:
        result = tl.load(prepared + row * half + col, mask=mask, other=0)
    else:
        result = tl.full(row.shape, 0, tl.float32)
        if c0 != 0:
            result += c0 * tl.load(full + row * stride0 + col * stride1, mask=mask, other=0).to(tl.float32)
        if c1 != 0:
            result += c1 * tl.load(full + row * stride0 + (half + col) * stride1,
                                   mask=mask, other=0).to(tl.float32)
        if c2 != 0:
            result += c2 * tl.load(full + (half + row) * stride0 + col * stride1,
                                   mask=mask, other=0).to(tl.float32)
        if c3 != 0:
            result += c3 * tl.load(full + (half + row) * stride0 + (half + col) * stride1,
                                   mask=mask, other=0).to(tl.float32)
        result = result.to(tl.float16)
    return result


@triton.jit
def _write_target(accum, previous, partial, output, row, col, mask,
                  half: tl.constexpr, coefficient: tl.constexpr,
                  has_previous: tl.constexpr, final: tl.constexpr,
                  output_block: tl.constexpr):
    value = coefficient * accum
    if has_previous:
        value += tl.load(previous + row * half + col, mask=mask, other=0)
    if final:
        dst_row = row + (output_block // 2) * half
        dst_col = col + (output_block % 2) * half
        tl.store(output + dst_row * (2 * half) + dst_col, value, mask=mask)
    else:
        tl.store(partial + row * half + col, value, mask=mask)


@triton.jit
def _fused_gemm_reconstruction_kernel(
        left, right, prepared_a, prepared_b,
        previous0, previous1, partial0, partial1, output,
        half: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
        stride_b0: tl.constexpr, stride_b1: tl.constexpr,
        u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
        v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
        prepared_u: tl.constexpr, prepared_v: tl.constexpr,
        coefficient0: tl.constexpr, has_previous0: tl.constexpr,
        final0: tl.constexpr, output_block0: tl.constexpr,
        has_target1: tl.constexpr, coefficient1: tl.constexpr,
        has_previous1: tl.constexpr, final1: tl.constexpr,
        output_block1: tl.constexpr,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    row = tl.program_id(0) * block_m + tl.arange(0, block_m)
    col = tl.program_id(1) * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    for start in range(0, half, block_k):
        a_mask = (row[:, None] < half) & (start + ks[None, :] < half)
        b_mask = (start + ks[:, None] < half) & (col[None, :] < half)
        a = _operand(left, prepared_a, row[:, None], start + ks[None, :], a_mask,
                     half, stride_a0, stride_a1, u0, u1, u2, u3, prepared_u)
        b = _operand(right, prepared_b, start + ks[:, None], col[None, :], b_mask,
                     half, stride_b0, stride_b1, v0, v1, v2, v3, prepared_v)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee", out_dtype=tl.float32)
    mask = (row[:, None] < half) & (col[None, :] < half)
    _write_target(accumulator, previous0, partial0, output, row[:, None], col[None, :], mask,
                  half, coefficient0, has_previous0, final0, output_block0)
    if has_target1:
        _write_target(accumulator, previous1, partial1, output, row[:, None], col[None, :], mask,
                      half, coefficient1, has_previous1, final1, output_block1)


def allocate_mixed(dag: ExecutionDag, schedule: Schedule, plan: ScheduleLoweringPlan):
    """Allocate only values explicitly located in HBM; never a fused product."""
    if plan.family_id != FAMILY_MIXED or plan.mapping_status != "EXACT_MAPPING":
        raise ValueError("mixed workspace requires an exact mixed plan")
    nodes = {node.node_id: node for node in dag.nodes}
    half = dag.config.matrix_size // 2
    values = {op.node_id: torch.empty((half, half), device="cuda",
                                      dtype=torch.float16 if nodes[op.node_id].operation == "LINEAR_FORM"
                                      else torch.float32)
              for group in schedule.groups for op in group.operations
              if op.location == "HBM" and nodes[op.node_id].operation != "OUTPUT"}
    return {"values": values,
            "output": torch.empty((2 * half, 2 * half), device="cuda", dtype=torch.float16)}


def _source_coefficients(nodes, source):
    node = nodes[source]
    if node.operation == "INPUT":
        return tuple(int(index == node.output_block) for index in range(4))
    if node.operation != "LINEAR_FORM":
        raise ValueError(f"unsupported GEMM source {source}")
    return node.coefficients


def execute_mixed(dag: ExecutionDag, schedule: Schedule, plan: ScheduleLoweringPlan,
                  left, right, workspace, capture: list | None = None):
    """Run the plan exactly in launch order, with one kernel per mixed group."""
    if plan.family_id != FAMILY_MIXED or plan.mapping_status != "EXACT_MAPPING":
        raise ValueError("only exact mixed plans can use this template")
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
    for launch in plan.launch_sequence:
        if launch["role"] == FAMILY_MIXED:
            source_a, source_b = launch["input_sources"]
            target0 = launch["reconstruction_targets"][0]
            target1 = launch["reconstruction_targets"][1] if len(launch["reconstruction_targets"]) == 2 else None
            prepared_a = source_a in workspace["values"]
            prepared_b = source_b in workspace["values"]
            def prior(target):
                return values[target["previous_id"]] if target and target["previous_id"] else left
            def partial(target):
                return values[target["accumulation_id"]] if target and target["write"] == "PARTIAL_FP32" else left
            compiled = _fused_gemm_reconstruction_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                left, right, values[source_a] if prepared_a else left,
                values[source_b] if prepared_b else right,
                prior(target0), prior(target1), partial(target0), partial(target1), workspace["output"],
                half, *left.stride(), *right.stride(),
                *_source_coefficients(nodes, source_a), *_source_coefficients(nodes, source_b),
                prepared_a, prepared_b, target0["coefficient"], bool(target0["previous_id"]),
                target0["write"] == "FINAL_FP16", target0["output_block"],
                bool(target1), target1["coefficient"] if target1 else 0,
                bool(target1 and target1["previous_id"]),
                target1["write"] == "FINAL_FP16" if target1 else False,
                target1["output_block"] if target1 else 0,
                block_m=32, block_n=32, block_k=16, num_warps=4, num_stages=3)
        else:
            node = nodes[launch["node_id"]]
            if node.operation == "LINEAR_FORM":
                source = [blocks[f"{node.node_id[0]}{index}"] for index in range(4)]
                compiled = _materialize_form_kernel[(triton.cdiv(half * half, 256),)](
                    *source, values[node.node_id], *(stride for tensor in source for stride in tensor.stride()),
                    *node.coefficients, half, block=256, num_warps=4)
            elif node.operation == "MATMUL":
                a, b = (values[key] for key in node.dependencies)
                c = values[node.node_id]
                compiled = _fp32_gemm_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                    a, b, c, half, half, half, *a.stride(), *b.stride(), *c.stride(),
                    block_m=32, block_n=32, block_k=16,
                    input_precision="ieee", num_warps=4, num_stages=3)
            elif node.operation == "ACCUMULATE":
                product = values[node.dependencies[-1]]
                previous = values[node.dependencies[0]] if len(node.dependencies) == 2 else product
                compiled = _accumulate_kernel[(triton.cdiv(half * half, 256),)](
                    product, previous, values[node.node_id], node.coefficients[0],
                    len(node.dependencies) == 2, half * half, block=256, num_warps=4)
            elif node.operation == "OUTPUT":
                compiled = _output_kernel[(triton.cdiv(half * half, 256),)](
                    values[node.dependencies[0]], workspace["output"], half, node.output_block,
                    block=256, num_warps=4)
            else:
                raise ValueError(f"unsupported singleton node {node.node_id}")
        if capture is not None:
            capture.append({"group_id": launch["group_id"], "role": launch["role"],
                            "node_id": launch["node_id"], "kernel": compiled.name,
                            "compiled": compiled})
    return workspace["output"]
