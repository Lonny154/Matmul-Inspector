"""Optional bounded Triton execution of validated HA-Schedule-9 IR plans."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_schedule_hbm_runtime import allocate_hbm, execute_hbm
from .tensor_game_schedule_hbm_lowering import FAMILY_HBM, HBMBackedPlan, classify_hbm_schedule
from .tensor_game_schedule_ir import ScheduleLoweringIR, validate_ir
from .tensor_game_schedule_ir_planner import IRLoweringPlan
from .tensor_game_schedule_lowering import FAMILY_LEGACY, FAMILY_MATERIALIZED
from .tensor_game_schedule_lowering_runtime import (
    _accumulate_kernel, _materialize_form_kernel, _output_kernel, allocate_materialized,
    execute_materialized,
)
from .tensor_game_schedule_mixed_lowering import FAMILY_MIXED
from .tensor_game_schedule_mixed_runtime import _source_coefficients, allocate_mixed, execute_mixed
from .tensor_game_schedule_model import ExecutionDag, Schedule
from .tensor_game_schedule_gpu_runtime import allocate_workspace, execute_selected
from .tensor_game_triton_runtime import _fp32_gemm_kernel
from .tensor_game_triton_lowering import create_lowering


@triton.jit
def _ir_operand(full, prepared, row, col, mask, half: tl.constexpr,
                stride0: tl.constexpr, stride1: tl.constexpr,
                c0, c1, c2, c3, use_prepared):
    """The HA-7 operand arithmetic with a broadcast-stable tile shape."""
    if use_prepared:
        result = tl.load(prepared + row * half + col, mask=mask, other=0)
    else:
        result = tl.full(mask.shape, 0, tl.float32)
        if c0 != 0:
            result += c0 * tl.load(full + row * stride0 + col * stride1, mask=mask, other=0).to(tl.float32)
        if c1 != 0:
            result += c1 * tl.load(full + row * stride0 + (half + col) * stride1, mask=mask, other=0).to(tl.float32)
        if c2 != 0:
            result += c2 * tl.load(full + (half + row) * stride0 + col * stride1, mask=mask, other=0).to(tl.float32)
        if c3 != 0:
            result += c3 * tl.load(full + (half + row) * stride0 + (half + col) * stride1,
                                   mask=mask, other=0).to(tl.float32)
        result = result.to(tl.float16)
    return result


@triton.jit
def _ordered_group_kernel(left, right, op_ptrs, output, program: tl.constexpr,
                          half: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
                          stride_b0: tl.constexpr, stride_b1: tl.constexpr,
                          block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    """One CTA owns each output tile and executes bounded semantic ops in order."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    row = pid_m * block_m + tl.arange(0, block_m)
    col = pid_n * block_n + tl.arange(0, block_n)
    mask = (row[:, None] < half) & (col[None, :] < half)
    tile_row, tile_col = tl.broadcast(row[:, None], col[None, :])
    ks = tl.arange(0, block_k)
    product_current = tl.zeros((block_m, block_n), dtype=tl.float32)
    value0 = tl.zeros((block_m, block_n), dtype=tl.float32)
    value1 = tl.zeros((block_m, block_n), dtype=tl.float32)
    value2 = tl.zeros((block_m, block_n), dtype=tl.float32)
    value3 = tl.zeros((block_m, block_n), dtype=tl.float32)
    for i in tl.static_range(len(program)):
        op = program[i]
        ptr = op_ptrs[i]
        if op[0] == 1:  # PREPARE_FORM or RECOMPUTE_FORM
            if op[6] >= 0 and op[7] == 0:
                if op[1] == 0:
                    form = _ir_operand(left, left, tile_row, tile_col, mask, half,
                                    stride_a0, stride_a1, op[2], op[3], op[4], op[5], False)
                else:
                    form = _ir_operand(right, right, tile_row, tile_col, mask, half,
                                    stride_b0, stride_b1, op[2], op[3], op[4], op[5], False)
                tl.store(ptr[2] + row[:, None] * half + col[None, :], form, mask=mask)
        elif op[0] == 2:  # GEMM, possibly storing operand forms and product
            product_current = tl.zeros((block_m, block_n), dtype=tl.float32)
            for start in range(0, half, block_k):
                a_mask = (row[:, None] < half) & (start + ks[None, :] < half)
                b_mask = (start + ks[:, None] < half) & (col[None, :] < half)
                if op[9] >= 0:
                    a = _ir_operand(left, ptr[0], row[:, None], start + ks[None, :], a_mask,
                                 half, stride_a0, stride_a1, op[1], op[2], op[3], op[4], True)
                else:
                    a = _ir_operand(left, left, row[:, None], start + ks[None, :], a_mask,
                                 half, stride_a0, stride_a1, op[1], op[2], op[3], op[4], False)
                if op[10] >= 0:
                    b = _ir_operand(right, ptr[1], start + ks[:, None], col[None, :], b_mask,
                                 half, stride_b0, stride_b1, op[5], op[6], op[7], op[8], True)
                else:
                    b = _ir_operand(right, right, start + ks[:, None], col[None, :], b_mask,
                                 half, stride_b0, stride_b1, op[5], op[6], op[7], op[8], False)
                if op[11] >= 0 and pid_n == 0:
                    tl.store(ptr[2] + row[:, None] * half + start + ks[None, :], a, mask=a_mask)
                if op[12] >= 0 and pid_m == 0:
                    tl.store(ptr[3] + (start + ks[:, None]) * half + col[None, :], b, mask=b_mask)
                product_current = tl.dot(a, b, product_current, input_precision="ieee", out_dtype=tl.float32)
            if op[13] >= 0:
                tl.store(ptr[7] + row[:, None] * half + col[None, :], product_current, mask=mask)
        elif op[0] == 3:  # ordered reconstruction contribution
            if op[1] < 0:
                product = product_current
            else:
                product = tl.load(ptr[4] + row[:, None] * half + col[None, :], mask=mask, other=0).to(tl.float32)
            updated = op[4] * product
            if op[2] == -2:
                if op[5] == 0:
                    updated += value0
                elif op[5] == 1:
                    updated += value1
                elif op[5] == 2:
                    updated += value2
                else:
                    updated += value3
            elif op[2] >= 0:
                updated += tl.load(ptr[5] + row[:, None] * half + col[None, :], mask=mask, other=0)
            if op[3] >= 0:
                tl.store(ptr[6] + row[:, None] * half + col[None, :], updated, mask=mask)
            if op[5] == 0:
                value0 = updated
            elif op[5] == 1:
                value1 = updated
            elif op[5] == 2:
                value2 = updated
            else:
                value3 = updated
        elif op[0] == 4:  # FINALIZE_OUTPUT
            if op[1] == -2:
                if op[2] == 0:
                    final = value0
                elif op[2] == 1:
                    final = value1
                elif op[2] == 2:
                    final = value2
                else:
                    final = value3
            else:
                final = tl.load(ptr[5] + row[:, None] * half + col[None, :], mask=mask, other=0).to(tl.float32)
            dst_row = row[:, None] + (op[2] // 2) * half
            dst_col = col[None, :] + (op[2] % 2) * half
            tl.store(output + dst_row * (2 * half) + dst_col, final, mask=mask)


def _descriptor(dag, ir, group, buffer_index):
    nodes = {node.node_id: node for node in dag.nodes}
    values = {value.value_id: value for value in ir.values}
    operations = {operation.op_id: operation for operation in ir.operations}
    selected = [operations[op_id] for op_id in group.op_ids]
    gemm = next((operation for operation in selected if operation.kind == "GEMM"), None)
    description = []

    def prior_index(value_id, *, current_gemm=False):
        value = values[value_id]
        if value.producer_op and operations[value.producer_op].group_id == group.group_id:
            return -1 if current_gemm else -2
        if value.buffer_id is None:
            raise ValueError("group needs an unavailable cross-launch value")
        return buffer_index[value.buffer_id]

    for op in selected:
        node = nodes[op.source_node_id]
        if op.kind in ("PREPARE_FORM", "RECOMPUTE_FORM"):
            operand_form = gemm is not None and op.output in gemm.inputs
            description.append((1, int(node.node_id.startswith("B")), *node.coefficients,
                                buffer_index[values[op.output].buffer_id] if values[op.output].buffer_id else -1,
                                operand_form))
        elif op.kind == "GEMM":
            source_a, source_b = op.inputs
            u = _source_coefficients(nodes, values[source_a].source_node_id)
            v = _source_coefficients(nodes, values[source_b].source_node_id)
            prepared = []
            form_writes = []
            for source in (source_a, source_b):
                value = values[source]
                producer = operations[value.producer_op] if value.producer_op else None
                same_group = producer is not None and producer.group_id == group.group_id
                prepared.append(buffer_index[value.buffer_id] if value.buffer_id and not same_group else -1)
                form_writes.append(buffer_index[value.buffer_id] if value.buffer_id and same_group else -1)
            product_write = buffer_index[values[op.output].buffer_id] if values[op.output].buffer_id else -1
            description.append((2, *u, *v, *prepared, *form_writes, product_write))
        elif op.kind == "RECONSTRUCT":
            product = prior_index(op.inputs[-1], current_gemm=True)
            previous = prior_index(op.inputs[0]) if len(op.inputs) == 2 else -1
            destination = buffer_index[values[op.output].buffer_id] if values[op.output].buffer_id else -1
            description.append((3, product, previous, destination, op.coefficient[0], op.output_block))
        elif op.kind == "FINALIZE_OUTPUT":
            description.append((4, prior_index(op.inputs[0]), op.output_block))
        else:
            raise ValueError(f"unknown IR op {op.kind}")
    # Triton validates tuple indexing before eliminating constexpr branches.
    # Keep one fixed descriptor width for every semantic operation.
    return tuple(tuple(int(value) for value in item) + (-9,) * (14 - len(item))
                 for item in description)


def allocate_ir(dag: ExecutionDag, ir: ScheduleLoweringIR, plan: IRLoweringPlan):
    if plan.backend != "IR_PRIMITIVES" or plan.mapping_status != "EXACT_MAPPING":
        raise ValueError("IR workspace requires an exact primitive plan")
    half = dag.config.matrix_size // 2
    values = {buffer.value_id: torch.empty((half, half), device="cuda",
                                           dtype=torch.float16 if buffer.dtype == "fp16" else torch.float32)
              for buffer in ir.buffer_plan if buffer.output_block is None}
    output = torch.empty((2 * half, 2 * half), device="cuda", dtype=torch.float16)
    return {"values": values, "output": output}


def audit_ir_workspace(dag, schedule, ir, plan, workspace):
    validate_ir(dag, schedule, ir)
    if plan.ir_hash != ir.ir_hash or plan.source_schedule_hash != schedule.schedule_hash:
        raise ValueError("IR plan identity mismatch")
    expected = {buffer.value_id for buffer in ir.buffer_plan if buffer.output_block is None}
    if set(workspace["values"]) != expected:
        raise ValueError("planned and actual logical HBM buffers differ")
    half = dag.config.matrix_size // 2
    for value_id, tensor in workspace["values"].items():
        value = next(value for value in ir.values if value.value_id == value_id)
        if (tensor.shape != (half, half) or tensor.dtype !=
                (torch.float16 if value.dtype == "fp16" else torch.float32) or not tensor.is_cuda):
            raise ValueError("HBM workspace shape/dtype mismatch")
    if (workspace["output"].shape != (2 * half, 2 * half) or
            workspace["output"].dtype != torch.float16 or not workspace["output"].is_cuda):
        raise ValueError("output workspace shape/dtype mismatch")
    return {"status": "PASS", "planned_hbm_values": len(expected), "actual_hbm_values": len(workspace["values"])}


def execute_ir(dag: ExecutionDag, schedule: Schedule, ir: ScheduleLoweringIR,
               plan: IRLoweringPlan, left, right, workspace, capture: list | None = None):
    """Run one primitive per IR group, in schedule order, on the current stream."""
    if plan.backend != "IR_PRIMITIVES" or plan.mapping_status != "EXACT_MAPPING":
        raise ValueError("plan does not select exact IR primitives")
    audit_ir_workspace(dag, schedule, ir, plan, workspace)
    size, half = dag.config.matrix_size, dag.config.matrix_size // 2
    if left.shape != (size, size) or right.shape != (size, size) or left.dtype != torch.float16 or (
            right.dtype != torch.float16 or not left.is_cuda or not right.is_cuda):
        raise ValueError("IR primitives require matching CUDA FP16 matrices")
    value_by_id = {value.value_id: value for value in ir.values}
    op_by_id = {op.op_id: op for op in ir.operations}
    buffers = tuple(workspace["values"][buffer.value_id] for buffer in ir.buffer_plan if buffer.output_block is None)
    # Output entries share the output tensor but do not occupy pointer slots.
    buffer_index = {buffer.buffer_id: index for index, buffer in enumerate(
        buffer for buffer in ir.buffer_plan if buffer.output_block is None)}
    blocks = {f"{side}{index}": matrix[(index // 2) * half:(index // 2 + 1) * half,
                                         (index % 2) * half:(index % 2 + 1) * half]
              for side, matrix in (("A", left), ("B", right)) for index in range(4)}
    values = {**blocks, **workspace["values"]}
    if capture is not None:
        capture.clear()
    for group, launch in zip(ir.groups, plan.launch_sequence):
        primitive = launch["role"]
        if primitive in ("ORDERED_ONE_GEMM", "ORDERED_RECON_FINAL"):
            descriptor = _descriptor(dag, ir, group, buffer_index)
            def pointer(index):
                return buffers[index] if index >= 0 else left
            op_ptrs = []
            for item in descriptor:
                if item[0] == 1:
                    op_ptrs.append((left, right, pointer(item[6]), right, left, left, left, left))
                elif item[0] == 2:
                    op_ptrs.append((pointer(item[9]), pointer(item[10]), pointer(item[11]),
                                    pointer(item[12]), left, left, left, pointer(item[13])))
                elif item[0] == 3:
                    op_ptrs.append((left, right, left, right, pointer(item[1]), pointer(item[2]),
                                    pointer(item[3]), left))
                else:
                    op_ptrs.append((left, right, left, right, left, pointer(item[1]), left, left))
            compiled = _ordered_group_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                left, right, tuple(op_ptrs), workspace["output"], descriptor,
                half, *left.stride(), *right.stride(), block_m=32, block_n=32, block_k=16,
                num_warps=4, num_stages=3)
        else:
            op = op_by_id[group.op_ids[0]]
            node = next(node for node in dag.nodes if node.node_id == op.source_node_id)
            if primitive == "SINGLETON_FORM":
                sources = [blocks[f"{node.node_id[0]}{index}"] for index in range(4)]
                compiled = _materialize_form_kernel[(triton.cdiv(half * half, 256),)](
                    *sources, values[op.output], *(stride for tensor in sources for stride in tensor.stride()),
                    *node.coefficients, half, block=256, num_warps=4)
            elif primitive == "SINGLETON_GEMM":
                a, b = (values[value_by_id[value_id].source_node_id] for value_id in op.inputs)
                c = values[op.output]
                compiled = _fp32_gemm_kernel[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                    a, b, c, half, half, half, *a.stride(), *b.stride(), *c.stride(),
                    block_m=32, block_n=32, block_k=16, input_precision="ieee", num_warps=4, num_stages=3)
            elif primitive == "SINGLETON_RECON":
                product = values[value_by_id[op.inputs[-1]].source_node_id]
                previous = values[value_by_id[op.inputs[0]].source_node_id] if len(op.inputs) == 2 else product
                compiled = _accumulate_kernel[(triton.cdiv(half * half, 256),)](
                    product, previous, values[op.output], op.coefficient[0], len(op.inputs) == 2,
                    half * half, block=256, num_warps=4)
            elif primitive == "SINGLETON_FINAL":
                previous = values[value_by_id[op.inputs[0]].source_node_id]
                compiled = _output_kernel[(triton.cdiv(half * half, 256),)](
                    previous, workspace["output"], half, op.output_block, block=256, num_warps=4)
            else:
                raise ValueError(f"unsupported primitive {primitive}")
        if capture is not None:
            capture.append({"group_id": group.group_id, "role": primitive,
                            "kernel": compiled.name, "compiled": compiled})
    return workspace["output"]


def allocate_backend(dag: ExecutionDag, schedule: Schedule, ir: ScheduleLoweringIR,
                     plan: IRLoweringPlan, *, lowering_name: str | None = None,
                     legacy_plan=None):
    """Select the IR backend or an unchanged legacy runtime explicitly."""
    if plan.backend == "IR_PRIMITIVES":
        return allocate_ir(dag, ir, plan)
    if plan.backend != "LEGACY_KERNEL_REUSE":
        raise ValueError("unknown backend")
    old = legacy_plan if legacy_plan is not None else classify_hbm_schedule(
        dag, schedule, lowering_name=lowering_name)
    base = old.schedule_plan if isinstance(old, HBMBackedPlan) else old
    if old.lowering_plan_hash != plan.legacy_plan_hash:
        raise ValueError("legacy plan hash mismatch")
    if base.family_id == FAMILY_HBM:
        return allocate_hbm(dag, schedule, old)
    if base.family_id == FAMILY_MIXED:
        return allocate_mixed(dag, schedule, base)
    if base.family_id == FAMILY_MATERIALIZED:
        return allocate_materialized(dag)
    if base.family_id == FAMILY_LEGACY:
        mode = {"rank8_stored": "stored_form", "rank8_recomputed": "recomputed_form"}.get(
            lowering_name, "reference")
        return allocate_workspace(create_lowering(dag.algorithm, dag.actions), dag.config.matrix_size, mode)
    raise ValueError("legacy family is not executable")


def execute_backend(dag: ExecutionDag, schedule: Schedule, ir: ScheduleLoweringIR,
                    plan: IRLoweringPlan, left, right, workspace, *,
                    lowering_name: str | None = None, legacy_plan=None,
                    capture: list | None = None):
    """Run a selected backend; legacy paths keep their original implementations."""
    if plan.backend == "IR_PRIMITIVES":
        return execute_ir(dag, schedule, ir, plan, left, right, workspace, capture=capture)
    old = legacy_plan if legacy_plan is not None else classify_hbm_schedule(
        dag, schedule, lowering_name=lowering_name)
    base = old.schedule_plan if isinstance(old, HBMBackedPlan) else old
    if old.lowering_plan_hash != plan.legacy_plan_hash:
        raise ValueError("legacy plan hash mismatch")
    if base.family_id == FAMILY_HBM:
        return execute_hbm(dag, schedule, old, left, right, workspace, capture=capture)
    if base.family_id == FAMILY_MIXED:
        return execute_mixed(dag, schedule, base, left, right, workspace, capture=capture)
    if base.family_id == FAMILY_MATERIALIZED:
        return execute_materialized(dag, schedule, base, left, right, workspace, capture=capture)
    if base.family_id == FAMILY_LEGACY:
        mode = {"rank8_stored": "stored_form", "rank8_recomputed": "recomputed_form"}.get(
            lowering_name, "reference")
        return execute_selected(create_lowering(dag.algorithm, dag.actions), left, right, workspace, mode)
    raise ValueError("legacy family is not executable")
