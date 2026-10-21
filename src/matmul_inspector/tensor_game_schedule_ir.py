"""CPU-only, bounded schedule-semantics IR for the retained Tensor Game DAGs.

Values are execution instances, not anonymous compiler temporaries. HBM
placement is an explicit value property; group edges encode typed loads and
stores without adding redundant executable load/store operations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace

from .tensor_game_schedule_calibration import stable_hash
from .tensor_game_schedule_hbm_lowering import HBMBufferPlan
from .tensor_game_schedule_model import ExecutionDag, Schedule, _resolve_schedule

IR_VERSION = "ha-schedule-9-ir-v1"
OP_KIND = {"LINEAR_FORM": "PREPARE_FORM", "MATMUL": "GEMM",
           "ACCUMULATE": "RECONSTRUCT", "OUTPUT": "FINALIZE_OUTPUT"}
VALUE_KIND = {"INPUT": "INPUT", "LINEAR_FORM": "PREPARED_FORM",
              "MATMUL": "GEMM_PRODUCT", "ACCUMULATE": "PARTIAL_OUTPUT",
              "OUTPUT": "FINAL_OUTPUT"}


@dataclass(frozen=True)
class LoweringValue:
    value_id: str
    value_class: str
    source_node_id: str
    dtype: str
    logical_shape: tuple[int, int]
    byte_size: int
    producer_op: str | None
    consumer_ops: tuple[str, ...]
    location: str
    buffer_id: str | None


@dataclass(frozen=True)
class LoweringOp:
    op_id: str
    kind: str
    source_node_id: str
    group_id: str
    inputs: tuple[str, ...]
    output: str
    location: str
    coefficient: tuple[int, ...]
    output_block: int | None
    term_index: int | None


@dataclass(frozen=True)
class LoweringGroup:
    group_id: str
    op_ids: tuple[str, ...]
    intended_strategy: str


@dataclass(frozen=True)
class ScheduleLoweringIR:
    version: str
    source_schedule_hash: str
    dag_hash: str
    algorithm: str
    decomposition_id: str
    values: tuple[LoweringValue, ...]
    operations: tuple[LoweringOp, ...]
    groups: tuple[LoweringGroup, ...]
    buffer_plan: tuple[HBMBufferPlan, ...]
    output_transitions: tuple[dict, ...]
    recomputation_records: tuple[dict, ...]
    provenance: dict
    ir_hash: str


def _payload(ir: ScheduleLoweringIR) -> dict:
    record = asdict(ir)
    record.pop("ir_hash")
    return record


def _value_id(node_id: str, ordinal: int) -> str:
    return node_id if ordinal == 1 else f"{node_id}@{ordinal}"


def translate_schedule(dag: ExecutionDag, schedule: Schedule) -> ScheduleLoweringIR:
    """Translate a legal schedule. Invalid source schedules are errors, not IR_UNSUPPORTED."""
    check, executions = _resolve_schedule(dag, schedule)
    if not check.valid:
        raise ValueError(f"invalid source schedule: {check.issues}")
    nodes = {node.node_id: node for node in dag.nodes}
    half = dag.config.matrix_size // 2
    counts: Counter[str] = Counter()
    names: dict[str, str] = {}
    operations = []
    for index, execution in enumerate(executions):
        node = nodes[execution.node_id]
        counts[node.node_id] += 1
        value_id = _value_id(node.node_id, counts[node.node_id])
        names[execution.key] = value_id
        operations.append(LoweringOp(f"op:{index}",
            "RECOMPUTE_FORM" if node.node_id in schedule.recompute_allowed else OP_KIND[node.operation],
            node.node_id, schedule.groups[execution.group_index].group_id,
            tuple(dep.removeprefix("input:") if dep.startswith("input:") else names[dep]
                  for dep in execution.dependencies),
            value_id, execution.location, node.coefficients, node.output_block, node.term_index))
    consumers: dict[str, list[str]] = defaultdict(list)
    for operation in operations:
        for value_id in operation.inputs:
            consumers[value_id].append(operation.op_id)
    values = []
    for node in dag.nodes:
        if node.operation == "INPUT":
            values.append(LoweringValue(node.node_id, "INPUT", node.node_id, node.dtype,
                                        (half, half), node.byte_count, None,
                                        tuple(consumers[node.node_id]), "INPUT", None))
    for operation in operations:
        node = nodes[operation.source_node_id]
        buffer_id = ("output" if node.operation == "OUTPUT" else f"buf:{node.node_id}") if operation.location == "HBM" else None
        values.append(LoweringValue(operation.output, VALUE_KIND[node.operation], node.node_id,
                                    node.dtype, (half, half), node.byte_count, operation.op_id,
                                    tuple(consumers[operation.output]), operation.location, buffer_id))
    ops_by_group: dict[str, list[str]] = defaultdict(list)
    for operation in operations:
        ops_by_group[operation.group_id].append(operation.op_id)
    groups = tuple(LoweringGroup(group.group_id, tuple(ops_by_group[group.group_id]),
                                 "SINGLE_KERNEL" if len(group.operations) <= 8 and
                                 sum(nodes[op.node_id].operation == "MATMUL" for op in group.operations) <= 1
                                 else "UNSUPPORTED") for group in schedule.groups)
    op_by_id = {op.op_id: op for op in operations}
    buffer_plan = []
    for value in values:
        if value.location != "HBM":
            continue
        producer = op_by_id[value.producer_op]
        last = op_by_id[value.consumer_ops[-1]].group_id if value.consumer_ops else producer.group_id
        buffer_plan.append(HBMBufferPlan(value.value_id, value.buffer_id, value.byte_size, value.dtype,
                                         producer.group_id, last,
                                         nodes[value.source_node_id].output_block if value.value_class == "FINAL_OUTPUT" else None))
    transitions = []
    for block in range(4):
        contributions = [op for op in operations if op.kind == "RECONSTRUCT" and op.output_block == block]
        transitions.append({"output_block": block,
                            "contribution_ops": tuple(op.op_id for op in contributions),
                            "contribution_nodes": tuple(op.source_node_id for op in contributions),
                            "coefficients": tuple(op.coefficient[0] for op in contributions),
                            "states": tuple(["UNINITIALIZED", *(["PARTIAL"] * len(contributions)), "FINAL"])})
    recomputations = tuple({"op_id": op.op_id, "source_node_id": op.source_node_id,
                            "input_values": op.inputs, "group_id": op.group_id}
                           for op in operations if op.kind == "RECOMPUTE_FORM")
    provisional = ScheduleLoweringIR(IR_VERSION, schedule.schedule_hash, dag.dag_hash,
        dag.algorithm, dag.decomposition_id, tuple(values), tuple(operations), groups,
        tuple(buffer_plan), tuple(transitions), recomputations,
        {"source": "schedule_execution_instances", "source_dag_hash": dag.dag_hash}, "")
    ir = replace(provisional, ir_hash=stable_hash(_payload(provisional)))
    validate_ir(dag, schedule, ir)
    return ir


def validate_ir(dag: ExecutionDag, schedule: Schedule, ir: ScheduleLoweringIR) -> dict:
    """Validate IR independently of the GPU planner. Raises on any mismatch."""
    if ir.version != IR_VERSION or ir.source_schedule_hash != schedule.schedule_hash or ir.dag_hash != dag.dag_hash:
        raise ValueError("IR source identity mismatch")
    if ir.algorithm != dag.algorithm or ir.decomposition_id != dag.decomposition_id:
        raise ValueError("IR decomposition mismatch")
    if ir.ir_hash != stable_hash(_payload(ir)):
        raise ValueError("IR hash mismatch")
    source_check, executions = _resolve_schedule(dag, schedule)
    if not source_check.valid:
        raise ValueError("source schedule invalid")
    nodes = {node.node_id: node for node in dag.nodes}
    if len(ir.operations) != len(executions) or len(ir.groups) != len(schedule.groups):
        raise ValueError("missing or extra IR operations or groups")
    if len({value.value_id for value in ir.values}) != len(ir.values):
        raise ValueError("duplicate IR value identity")
    if len({operation.op_id for operation in ir.operations}) != len(ir.operations):
        raise ValueError("duplicate IR operation identity")
    by_value = {value.value_id: value for value in ir.values}
    by_op = {op.op_id: op for op in ir.operations}
    counts: Counter[str] = Counter()
    names = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    group_index = {group.group_id: index for index, group in enumerate(schedule.groups)}
    seen_ops = set()
    for group, actual in zip(schedule.groups, ir.groups):
        expected_op_ids = tuple(op.op_id for op in ir.operations if op.group_id == group.group_id)
        if (actual.group_id != group.group_id or len(actual.op_ids) != len(group.operations)
                or actual.op_ids != expected_op_ids):
            raise ValueError("IR group boundary differs from schedule")
        if actual.intended_strategy not in ("SINGLE_KERNEL", "MULTI_KERNEL_BOUNDED", "UNSUPPORTED"):
            raise ValueError("invalid group strategy")
        for op_id in actual.op_ids:
            if op_id not in by_op or op_id in seen_ops or by_op[op_id].group_id != group.group_id:
                raise ValueError("orphan, duplicate, or misplaced IR operation")
            seen_ops.add(op_id)
    if len(seen_ops) != len(ir.operations):
        raise ValueError("orphan IR operation")
    for index, (execution, operation) in enumerate(zip(executions, ir.operations)):
        node = nodes[execution.node_id]
        counts[node.node_id] += 1
        expected_value = _value_id(node.node_id, counts[node.node_id])
        expected_inputs = tuple(dep.removeprefix("input:") if dep.startswith("input:") else names[dep]
                                for dep in execution.dependencies)
        expected_kind = "RECOMPUTE_FORM" if node.node_id in schedule.recompute_allowed else OP_KIND[node.operation]
        if (operation.op_id != f"op:{index}" or operation.source_node_id != node.node_id or
                operation.kind != expected_kind or operation.output != expected_value or
                operation.inputs != expected_inputs or operation.location != execution.location or
                operation.group_id != schedule.groups[execution.group_index].group_id or
                operation.coefficient != node.coefficients or operation.output_block != node.output_block):
            raise ValueError("IR operation or dependency differs from source schedule")
        names[execution.key] = expected_value
        for dependency in operation.inputs:
            value = by_value.get(dependency)
            if value is None:
                raise ValueError("missing producer value")
            if value.producer_op is not None:
                producer = by_op.get(value.producer_op)
                if producer is None or int(producer.op_id.split(":")[1]) >= index:
                    raise ValueError("dependency is cyclic or producer is missing")
                if group_index[producer.group_id] < execution.group_index and value.location != "HBM":
                    raise ValueError("ON_CHIP or EPHEMERAL value crosses a group boundary")
                if group_index[producer.group_id] < execution.group_index and not value.buffer_id:
                    raise ValueError("HBM load has no valid store")
            consumers[dependency].append(operation.op_id)
        if operation.kind == "RECOMPUTE_FORM" and (operation.location != "EPHEMERAL" or
                any(by_value[dependency].value_class != "INPUT" for dependency in operation.inputs)):
            raise ValueError("recomputation inputs unavailable")
    if len(ir.values) != len(executions) + 8:
        raise ValueError("IR value table missing input or execution value")
    half = dag.config.matrix_size // 2
    expected_buffers = []
    for value in ir.values:
        node = nodes.get(value.source_node_id)
        if node is None or value.value_class != VALUE_KIND[node.operation] or value.dtype != node.dtype or (
                value.logical_shape != (half, half) or value.byte_size != node.byte_count or
                value.consumer_ops != tuple(consumers[value.value_id])):
            raise ValueError("IR value metadata or consumers differ from DAG")
        if value.value_class == "INPUT":
            if value.producer_op is not None or value.location != "INPUT" or value.buffer_id is not None:
                raise ValueError("invalid input value")
            continue
        producer = by_op.get(value.producer_op)
        if producer is None or producer.output != value.value_id or producer.location != value.location:
            raise ValueError("IR value producer mismatch")
        expected_buffer = ("output" if node.operation == "OUTPUT" else f"buf:{node.node_id}") if value.location == "HBM" else None
        if value.buffer_id != expected_buffer:
            raise ValueError("missing or hidden HBM buffer")
        if value.location == "HBM":
            last = by_op[value.consumer_ops[-1]].group_id if value.consumer_ops else producer.group_id
            expected_buffers.append(HBMBufferPlan(value.value_id, expected_buffer, value.byte_size, value.dtype,
                producer.group_id, last, node.output_block if node.operation == "OUTPUT" else None))
    if ir.buffer_plan != tuple(expected_buffers):
        raise ValueError("HBM buffer plan differs from IR values or lifetime")
    seen_buffers = {}
    for buffer in ir.buffer_plan:
        if buffer.buffer_id in seen_buffers and not (buffer.buffer_id == "output" and
                buffer.output_block not in seen_buffers[buffer.buffer_id]):
            raise ValueError("illegal overlapping HBM buffer reuse")
        seen_buffers.setdefault(buffer.buffer_id, set()).add(buffer.output_block)
    expected_transitions = []
    for block in range(4):
        required = [node for node in dag.nodes if node.operation == "ACCUMULATE" and node.output_block == block]
        actual = [op for op in ir.operations if op.kind == "RECONSTRUCT" and op.output_block == block]
        finals = [op for op in ir.operations if op.kind == "FINALIZE_OUTPUT" and op.output_block == block]
        if [op.source_node_id for op in actual] != [node.node_id for node in required]:
            raise ValueError("duplicate, omitted, or reordered reconstruction contribution")
        if len(finals) != 1 or finals[0].source_node_id != dag.output_ids[block] or (
                int(finals[0].op_id.split(":")[1]) <= int(actual[-1].op_id.split(":")[1])):
            raise ValueError("output not finalized exactly once after its contributions")
        expected_transitions.append({"output_block": block,
            "contribution_ops": tuple(op.op_id for op in actual),
            "contribution_nodes": tuple(op.source_node_id for op in actual),
            "coefficients": tuple(op.coefficient[0] for op in actual),
            "states": tuple(["UNINITIALIZED", *(["PARTIAL"] * len(actual)), "FINAL"])})
    if ir.output_transitions != tuple(expected_transitions):
        raise ValueError("invalid partial-output or FINAL state transition")
    expected_recomputations = tuple({"op_id": op.op_id, "source_node_id": op.source_node_id,
        "input_values": op.inputs, "group_id": op.group_id} for op in ir.operations if op.kind == "RECOMPUTE_FORM")
    if ir.recomputation_records != expected_recomputations:
        raise ValueError("recomputation record mismatch")
    return {"status": "IR_VALID", "value_count": len(ir.values), "operation_count": len(ir.operations),
            "group_count": len(ir.groups), "hbm_buffer_count": len(ir.buffer_plan)}


def ir_schema() -> dict:
    return {"version": IR_VERSION, "operations": sorted({*OP_KIND.values(), "RECOMPUTE_FORM"}),
            "value_classes": sorted(set(VALUE_KIND.values())),
            "placements": ["INPUT", "HBM", "ON_CHIP", "EPHEMERAL"],
            "group_strategies": ["SINGLE_KERNEL", "MULTI_KERNEL_BOUNDED", "UNSUPPORTED"],
            "hbm_contract": "HA8 one stable logical buffer per HBM intermediate; disjoint output quadrants share output"}
