"""Exact block-level Tensor Game DAGs and CPU-only schedule/traffic accounting.

Locations and launch groups are abstract. Counts are not GPU traffic or latency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

from .tensor_game import RankOneAction
from .tensor_game_candidate_dataset import actions_payload, validate_decomposition
from .tensor_game_hardware_cost import linear_combination_cost
from .tensor_game_triton_lowering import reference_lowering

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
POLICIES = ("MATERIALIZE_ALL", "FUSE_INPUT_PREPARATION", "FUSE_OUTPUT_RECONSTRUCTION",
            "FUSE_INPUT_AND_OUTPUT", "RECOMPUTE_CHEAP_FORMS")
LOCATIONS = ("HBM", "ON_CHIP", "EPHEMERAL")


def _digest(value: object) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


@dataclass(frozen=True)
class BlockConfig:
    matrix_size: int = 128
    input_dtype: str = "fp32"
    accumulation_dtype: str = "fp32"
    output_dtype: str = "fp32"

    def __post_init__(self) -> None:
        if not isinstance(self.matrix_size, int) or isinstance(self.matrix_size, bool) or self.matrix_size <= 0 or self.matrix_size % 2:
            raise ValueError("matrix size must be a positive even integer")
        for name in ("input_dtype", "accumulation_dtype", "output_dtype"):
            if getattr(self, name) not in DTYPE_BYTES:
                raise ValueError(f"unsupported {name}: {getattr(self, name)}")

    @property
    def block_elements(self) -> int:
        return (self.matrix_size // 2) ** 2

    def block_bytes(self, dtype: str) -> int:
        return self.block_elements * DTYPE_BYTES[dtype]


@dataclass(frozen=True)
class DagNode:
    node_id: str
    operation: str
    dependencies: tuple[str, ...]
    output_value_id: str
    coefficients: tuple[int, ...]
    term_index: int | None
    output_block: int | None
    element_count: int
    byte_count: int
    dtype: str
    materializable: bool
    arithmetic_operations: int


@dataclass(frozen=True)
class TermPlan:
    term_index: int
    u_source: str
    v_source: str
    product_id: str
    accumulation_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionDag:
    algorithm: str
    decomposition_id: str
    config: BlockConfig
    actions: tuple[RankOneAction, ...]
    nodes: tuple[DagNode, ...]
    terms: tuple[TermPlan, ...]
    output_ids: tuple[str, ...]
    dag_hash: str


@dataclass(frozen=True)
class ScheduledOp:
    node_id: str
    location: str


@dataclass(frozen=True)
class LaunchGroup:
    group_id: str
    operations: tuple[ScheduledOp, ...]


@dataclass(frozen=True)
class Schedule:
    schedule_id: str
    policy: str
    dag_hash: str
    groups: tuple[LaunchGroup, ...]
    recompute_allowed: tuple[str, ...]
    schedule_hash: str


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    node_id: str | None = None
    group_id: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...]


def _node(node_id: str, operation: str, dependencies: Sequence[str], coefficients: Sequence[int],
          term_index: int | None, output_block: int | None, config: BlockConfig,
          dtype: str, materializable: bool, arithmetic_operations: int) -> DagNode:
    return DagNode(node_id, operation, tuple(dependencies), f"value:{node_id}", tuple(coefficients),
                   term_index, output_block, config.block_elements, config.block_bytes(dtype),
                   dtype, materializable, arithmetic_operations)


def _dag_payload(dag: ExecutionDag) -> dict:
    return {"algorithm": dag.algorithm, "decomposition_id": dag.decomposition_id,
            "config": asdict(dag.config), "actions": actions_payload(dag.actions),
            "nodes": [asdict(node) for node in dag.nodes],
            "terms": [asdict(term) for term in dag.terms], "output_ids": list(dag.output_ids)}


def build_execution_dag(algorithm: str, actions: Sequence[RankOneAction],
                        config: BlockConfig = BlockConfig()) -> ExecutionDag:
    """Build the exact one-level 2x2 block DAG from existing u/v/w factors."""
    frozen = tuple(actions)
    validate_decomposition(frozen)
    nodes: list[DagNode] = []
    for side in ("A", "B"):
        for block in range(4):
            nodes.append(_node(f"{side}{block}", "INPUT", (), (), None, block,
                               config, config.input_dtype, False, 0))
    forms: dict[tuple[str, tuple[int, ...]], str] = {}
    sources: list[tuple[str, str]] = []
    for action in frozen:
        pair = []
        for side, vector in (("A", action.u), ("B", action.v)):
            if vector.count(1) == 1 and sum(value != 0 for value in vector) == 1:
                pair.append(f"{side}{vector.index(1)}")
                continue
            key = (side, vector)
            if key not in forms:
                node_id = f"{side}_FORM{len([key for key in forms if key[0] == side]):02d}"
                forms[key] = node_id
                dependencies = [f"{side}{index}" for index, coefficient in enumerate(vector) if coefficient]
                cost = linear_combination_cost(vector)
                nodes.append(_node(node_id, "LINEAR_FORM", dependencies, vector,
                                   None, None, config, config.input_dtype, True,
                                   (cost.additions + cost.nonunit_coefficients) * config.block_elements))
            pair.append(forms[key])
        sources.append((pair[0], pair[1]))
    for index, (u_source, v_source) in enumerate(sources):
        nodes.append(_node(f"M{index:02d}", "MATMUL", (u_source, v_source), (),
                           index, None, config, config.accumulation_dtype, True, 0))
    by_term: list[list[str]] = [[] for _ in frozen]
    output_ids: list[str] = []
    for output in range(4):
        previous = None
        for index, action in enumerate(frozen):
            coefficient = action.w[output]
            if not coefficient:
                continue
            node_id = f"R{output}_{index:02d}"
            dependencies = (f"M{index:02d}",) if previous is None else (previous, f"M{index:02d}")
            operations = int(previous is not None) + int(abs(coefficient) > 1)
            nodes.append(_node(node_id, "ACCUMULATE", dependencies, (coefficient,),
                               index, output, config, config.accumulation_dtype, True,
                               operations * config.block_elements))
            by_term[index].append(node_id)
            previous = node_id
        if previous is None:
            raise ValueError("exact decomposition has an output block with no contributions")
        output_id = f"O{output}"
        nodes.append(_node(output_id, "OUTPUT", (previous,), (), None, output,
                           config, config.output_dtype, False, 0))
        output_ids.append(output_id)
    terms = tuple(TermPlan(index, u, v, f"M{index:02d}", tuple(by_term[index]))
                  for index, (u, v) in enumerate(sources))
    temporary = ExecutionDag(algorithm, _digest(actions_payload(frozen)), config, frozen,
                             tuple(nodes), terms, tuple(output_ids), "")
    dag = ExecutionDag(algorithm, temporary.decomposition_id, config, frozen, tuple(nodes), terms,
                       tuple(output_ids), _digest(_dag_payload(temporary)))
    result = validate_dag(dag)
    if not result.valid:
        raise ValueError(f"invalid generated DAG: {[asdict(issue) for issue in result.issues]}")
    return dag


def reference_dag(name: str, config: BlockConfig = BlockConfig()) -> ExecutionDag:
    lowering = reference_lowering(name)  # Exact validation, no runtime import.
    return build_execution_dag(name, lowering.actions, config)


def validate_dag(dag: ExecutionDag) -> ValidationResult:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    allowed = {"INPUT", "LINEAR_FORM", "MATMUL", "ACCUMULATE", "OUTPUT"}
    for node in dag.nodes:
        if node.node_id in seen:
            issues.append(ValidationIssue("DUPLICATE_NODE", "node ID repeats", node.node_id))
        if node.operation not in allowed:
            issues.append(ValidationIssue("UNKNOWN_OPERATION", "unknown DAG operation", node.node_id))
        for dependency in node.dependencies:
            if dependency not in seen:
                issues.append(ValidationIssue("NON_TOPOLOGICAL_DEPENDENCY", f"dependency {dependency} is not earlier", node.node_id))
        if node.output_value_id != f"value:{node.node_id}" or node.byte_count <= 0:
            issues.append(ValidationIssue("INVALID_VALUE", "invalid output identity or size", node.node_id))
        seen.add(node.node_id)
    if len(dag.output_ids) != 4 or any(value not in seen for value in dag.output_ids):
        issues.append(ValidationIssue("MISSING_OUTPUT", "DAG must contain four output blocks"))
    if sum(node.operation == "MATMUL" for node in dag.nodes) != len(dag.actions):
        issues.append(ValidationIssue("MATMUL_COUNT", "one MATMUL is required per rank term"))
    if dag.dag_hash != _digest(_dag_payload(dag)):
        issues.append(ValidationIssue("DAG_HASH", "DAG content hash mismatch"))
    return ValidationResult(not issues, tuple(issues))


def _schedule_payload(schedule: Schedule) -> dict:
    return {"schedule_id": schedule.schedule_id, "policy": schedule.policy,
            "dag_hash": schedule.dag_hash, "groups": [asdict(group) for group in schedule.groups],
            "recompute_allowed": list(schedule.recompute_allowed)}


def make_schedule(dag: ExecutionDag, policy: str) -> Schedule:
    if policy not in POLICIES:
        raise ValueError(f"unknown schedule policy: {policy}")
    node_map = {node.node_id: node for node in dag.nodes}
    forms = [node for node in dag.nodes if node.operation == "LINEAR_FORM"]
    form_uses = Counter(source for term in dag.terms for source in (term.u_source, term.v_source)
                        if node_map[source].operation == "LINEAR_FORM")
    fuse_input = policy in {"FUSE_INPUT_PREPARATION", "FUSE_INPUT_AND_OUTPUT", "RECOMPUTE_CHEAP_FORMS"}
    fuse_output = policy in {"FUSE_OUTPUT_RECONSTRUCTION", "FUSE_INPUT_AND_OUTPUT", "RECOMPUTE_CHEAP_FORMS"}
    recompute = {node.node_id for node in forms if policy == "RECOMPUTE_CHEAP_FORMS"
                 and form_uses[node.node_id] >= 2
                 and 0 < node.arithmetic_operations <= 2 * node.element_count}
    groups: list[LaunchGroup] = []

    def append(operations: list[ScheduledOp]) -> None:
        groups.append(LaunchGroup(f"G{len(groups):03d}", tuple(operations)))

    for form in forms:
        if form.node_id in recompute:
            continue
        if fuse_input and form_uses[form.node_id] == 1:
            continue
        append([ScheduledOp(form.node_id, "HBM")])
    final_accum = {node.dependencies[0]: node.node_id for node in dag.nodes if node.operation == "OUTPUT"}
    if fuse_output:
        for term in dag.terms:
            steps: list[ScheduledOp] = []
            for source in (term.u_source, term.v_source):
                if source in recompute:
                    steps.append(ScheduledOp(source, "EPHEMERAL"))
                elif fuse_input and node_map[source].operation == "LINEAR_FORM" and form_uses[source] == 1:
                    steps.append(ScheduledOp(source, "ON_CHIP"))
            steps.append(ScheduledOp(term.product_id, "ON_CHIP"))
            for accumulator in term.accumulation_ids:
                output = final_accum.get(accumulator)
                steps.append(ScheduledOp(accumulator, "ON_CHIP" if output else "HBM"))
                if output:
                    steps.append(ScheduledOp(output, "HBM"))
            append(steps)
    else:
        for term in dag.terms:
            steps = []
            if fuse_input:
                for source in (term.u_source, term.v_source):
                    if node_map[source].operation == "LINEAR_FORM" and form_uses[source] == 1:
                        steps.append(ScheduledOp(source, "ON_CHIP"))
            steps.append(ScheduledOp(term.product_id, "HBM"))
            append(steps)
        for node in dag.nodes:
            if node.operation == "ACCUMULATE":
                append([ScheduledOp(node.node_id, "HBM")])
        for output in dag.output_ids:
            append([ScheduledOp(output, "HBM")])
    temporary = Schedule(f"{dag.algorithm}:{policy}", policy, dag.dag_hash,
                         tuple(groups), tuple(sorted(recompute)), "")
    return Schedule(temporary.schedule_id, policy, dag.dag_hash, tuple(groups),
                    temporary.recompute_allowed, _digest(_schedule_payload(temporary)))


@dataclass(frozen=True)
class _Execution:
    key: str
    node_id: str
    group_index: int
    location: str
    dependencies: tuple[str, ...]


def _resolve_schedule(dag: ExecutionDag, schedule: Schedule) -> tuple[ValidationResult, tuple[_Execution, ...]]:
    issues = list(validate_dag(dag).issues)
    node_map = {node.node_id: node for node in dag.nodes}
    if schedule.dag_hash != dag.dag_hash:
        issues.append(ValidationIssue("DAG_MISMATCH", "schedule targets a different DAG"))
    if schedule.schedule_hash != _digest(_schedule_payload(schedule)):
        issues.append(ValidationIssue("SCHEDULE_HASH", "schedule content hash mismatch"))
    if schedule.policy not in POLICIES:
        issues.append(ValidationIssue("UNKNOWN_POLICY", "unknown schedule policy"))
    allowed = set(schedule.recompute_allowed)
    for node_id in allowed:
        if node_id not in node_map or node_map[node_id].operation != "LINEAR_FORM":
            issues.append(ValidationIssue("INVALID_RECOMPUTE_PERMISSION", "only a LINEAR_FORM may be recomputed", node_id))
    persistent: dict[str, str] = {}
    executions: list[_Execution] = []
    counts: Counter[str] = Counter()
    group_ids: set[str] = set()
    consumed: Counter[str] = Counter()
    for group_index, group in enumerate(schedule.groups):
        if group.group_id in group_ids:
            issues.append(ValidationIssue("DUPLICATE_GROUP", "group ID repeats", group_id=group.group_id))
        group_ids.add(group.group_id)
        local: dict[str, str] = {}
        ephemeral: dict[str, str] = {}
        for op in group.operations:
            node = node_map.get(op.node_id)
            if node is None:
                issues.append(ValidationIssue("UNKNOWN_NODE", "scheduled node is absent from DAG", op.node_id, group.group_id))
                continue
            if node.operation == "INPUT":
                issues.append(ValidationIssue("SCHEDULED_INPUT", "input blocks exist before the schedule", op.node_id, group.group_id))
                continue
            if op.location not in LOCATIONS or (node.operation == "OUTPUT" and op.location != "HBM"):
                issues.append(ValidationIssue("INVALID_LOCATION", "unsupported location or non-HBM final output", op.node_id, group.group_id))
                continue
            if counts[op.node_id] and (op.node_id not in allowed or op.location != "EPHEMERAL"):
                issues.append(ValidationIssue("ILLEGAL_RECOMPUTE", "duplicate execution lacks ephemeral recompute permission", op.node_id, group.group_id))
            if op.node_id in allowed and op.location != "EPHEMERAL":
                issues.append(ValidationIssue("INVALID_RECOMPUTE_LOCATION", "recomputed forms must be ephemeral", op.node_id, group.group_id))
            dependencies: list[str] = []
            for dep in node.dependencies:
                producer = node_map.get(dep)
                if producer is None:
                    issues.append(ValidationIssue("UNKNOWN_DEPENDENCY", f"DAG dependency {dep} is absent", op.node_id, group.group_id))
                elif producer.operation == "INPUT":
                    dependencies.append(f"input:{dep}")
                elif dep in ephemeral:
                    dependencies.append(ephemeral.pop(dep))
                elif dep in local:
                    dependencies.append(local[dep])
                elif dep in persistent:
                    dependencies.append(persistent[dep])
                else:
                    issues.append(ValidationIssue("UNAVAILABLE_VALUE", f"dependency {dep} is not available in this group", op.node_id, group.group_id))
            key = f"execution:{len(executions)}"
            executions.append(_Execution(key, op.node_id, group_index, op.location, tuple(dependencies)))
            for dependency in dependencies:
                consumed[dependency] += 1
            counts[op.node_id] += 1
            if op.location == "HBM":
                persistent[op.node_id] = key
            elif op.location == "ON_CHIP":
                local[op.node_id] = key
            else:
                ephemeral[op.node_id] = key
        for key in ephemeral.values():
            issues.append(ValidationIssue("UNCONSUMED_EPHEMERAL", "ephemeral value was not consumed", executions[int(key.split(":")[1])].node_id, group.group_id))
    for node in dag.nodes:
        if node.operation != "INPUT" and counts[node.node_id] == 0:
            issues.append(ValidationIssue("MISSING_REQUIRED_NODE", "required DAG node was not executed", node.node_id))
        if node.operation == "OUTPUT" and (counts[node.node_id] != 1 or node.node_id not in persistent):
            issues.append(ValidationIssue("MISSING_OUTPUT", "final output must be produced once in HBM", node.node_id))
    for execution in executions:
        if node_map[execution.node_id].operation != "OUTPUT" and consumed[execution.key] == 0:
            issues.append(ValidationIssue("ORPHAN_EXECUTION", "executed intermediate is never consumed", execution.node_id,
                                          schedule.groups[execution.group_index].group_id))
    return ValidationResult(not issues, tuple(issues)), tuple(executions)


def validate_schedule(dag: ExecutionDag, schedule: Schedule) -> ValidationResult:
    """Check data availability, group lifetimes, exact outputs, and recomputation."""
    return _resolve_schedule(dag, schedule)[0]


def evaluate_schedule(dag: ExecutionDag, schedule: Schedule) -> dict:
    """Return integer static counts only; reject an invalid schedule."""
    validation, executions = _resolve_schedule(dag, schedule)
    if not validation.valid:
        raise ValueError(f"invalid schedule: {[asdict(issue) for issue in validation.issues]}")
    nodes = {node.node_id: node for node in dag.nodes}
    by_key = {execution.key: execution for execution in executions}
    remaining = Counter(dep for execution in executions for dep in execution.dependencies
                        if dep.startswith("execution:"))
    live_hbm: set[str] = set()
    live_on_chip: set[str] = set()
    live_ephemeral: set[str] = set()
    metrics = Counter()
    group_peaks: list[dict] = []
    previous_group = -1
    group_peak = 0

    def live_size(keys: set[str]) -> int:
        return sum(nodes[by_key[key].node_id].byte_count for key in keys)

    def update_peaks() -> None:
        metrics["peak_temp_hbm_bytes"] = max(metrics["peak_temp_hbm_bytes"], live_size(live_hbm))
        metrics["peak_on_chip_live_bytes"] = max(metrics["peak_on_chip_live_bytes"], live_size(live_on_chip))
        metrics["peak_simultaneously_live_intermediates"] = max(
            metrics["peak_simultaneously_live_intermediates"],
            len(live_hbm | live_on_chip | live_ephemeral))

    for execution in executions:
        if execution.group_index != previous_group:
            if previous_group >= 0:
                group_peaks.append({"group_id": schedule.groups[previous_group].group_id,
                                    "peak_on_chip_live_bytes": group_peak})
                if live_on_chip or live_ephemeral:
                    raise AssertionError("validated group retained local values")
            previous_group = execution.group_index
            group_peak = 0
        node = nodes[execution.node_id]
        for dependency in execution.dependencies:
            if dependency.startswith("input:"):
                metrics["hbm_read_bytes"] += nodes[dependency[6:]].byte_count
            else:
                producer = by_key[dependency]
                if producer.location == "HBM":
                    amount = nodes[producer.node_id].byte_count
                    metrics["hbm_read_bytes"] += amount
                    if nodes[producer.node_id].operation != "OUTPUT":
                        metrics["temporary_hbm_bytes_read"] += amount
        if node.operation == "MATMUL":
            metrics["logical_gemm_count"] += 1
        elif node.operation == "LINEAR_FORM":
            metrics["linear_form_arithmetic_operations"] += node.arithmetic_operations
        elif node.operation == "ACCUMULATE":
            metrics["reconstruction_arithmetic_operations"] += node.arithmetic_operations
        if execution.location == "HBM":
            metrics["hbm_write_bytes"] += node.byte_count
            if node.operation == "OUTPUT":
                metrics["final_output_bytes"] += node.byte_count
            else:
                metrics["temporary_hbm_bytes_written"] += node.byte_count
                metrics["number_of_materialized_intermediates"] += 1
                live_hbm.add(execution.key)
        elif execution.location == "ON_CHIP":
            live_on_chip.add(execution.key)
        else:
            live_ephemeral.add(execution.key)
        update_peaks()
        group_peak = max(group_peak, live_size(live_on_chip))
        for dependency in execution.dependencies:
            if dependency.startswith("input:"):
                continue
            remaining[dependency] -= 1
            if remaining[dependency] == 0:
                live_hbm.discard(dependency)
                live_on_chip.discard(dependency)
                live_ephemeral.discard(dependency)
        if node.operation != "OUTPUT" and remaining[execution.key] == 0:
            live_hbm.discard(execution.key)
            live_on_chip.discard(execution.key)
            live_ephemeral.discard(execution.key)
    if previous_group >= 0:
        group_peaks.append({"group_id": schedule.groups[previous_group].group_id,
                            "peak_on_chip_live_bytes": group_peak})
    counts = Counter(execution.node_id for execution in executions)
    for node_id in schedule.recompute_allowed:
        if counts[node_id] > 1:
            node = nodes[node_id]
            extra = counts[node_id] - 1
            metrics["number_of_recomputed_intermediates"] += extra
            metrics["recomputation_arithmetic_operations"] += extra * node.arithmetic_operations
            metrics["recomputed_hbm_write_bytes_avoided"] += node.byte_count
            metrics["recomputed_hbm_reload_bytes_avoided"] += counts[node_id] * node.byte_count
            metrics["extra_source_read_bytes"] += extra * sum(nodes[dep].byte_count for dep in node.dependencies)
    metrics["temporary_hbm_traffic_bytes"] = metrics["temporary_hbm_bytes_read"] + metrics["temporary_hbm_bytes_written"]
    metrics["largest_single_on_chip_value_set_bytes"] = metrics["peak_on_chip_live_bytes"]
    result = {
        "algorithm": dag.algorithm, "decomposition_id": dag.decomposition_id,
        "matrix_size": dag.config.matrix_size, "block_dimension": dag.config.matrix_size // 2,
        "block_elements": dag.config.block_elements, "input_dtype": dag.config.input_dtype,
        "accumulation_dtype": dag.config.accumulation_dtype, "output_dtype": dag.config.output_dtype,
        "input_block_bytes": dag.config.block_bytes(dag.config.input_dtype),
        "product_accumulation_block_bytes": dag.config.block_bytes(dag.config.accumulation_dtype),
        "output_block_bytes": dag.config.block_bytes(dag.config.output_dtype),
        "schedule_id": schedule.schedule_id, "policy": schedule.policy,
        "schedule_hash": schedule.schedule_hash, "dag_hash": dag.dag_hash,
        "validation_status": "PASS", "logical_launch_groups": len(schedule.groups),
        "logical_group_boundaries": max(0, len(schedule.groups) - 1),
        "explicit_synchronization_boundaries": 0,
        "total_dag_nodes_executed": len(executions) + 8,
        "total_value_materializations": sum(execution.location != "EPHEMERAL" for execution in executions),
        "peak_on_chip_live_bytes_per_group": group_peaks,
    }
    for name in ("logical_gemm_count", "linear_form_arithmetic_operations", "reconstruction_arithmetic_operations",
                 "recomputation_arithmetic_operations", "hbm_read_bytes", "hbm_write_bytes",
                 "temporary_hbm_bytes_written", "temporary_hbm_bytes_read", "temporary_hbm_traffic_bytes",
                 "final_output_bytes", "peak_temp_hbm_bytes", "peak_on_chip_live_bytes",
                 "largest_single_on_chip_value_set_bytes", "peak_simultaneously_live_intermediates",
                 "number_of_materialized_intermediates", "number_of_recomputed_intermediates",
                 "recomputed_hbm_write_bytes_avoided", "recomputed_hbm_reload_bytes_avoided",
                 "extra_source_read_bytes"):
        result[name] = metrics[name]
    return result
