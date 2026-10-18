"""Optional CPU-only CP-SAT search over a fixed exact execution-DAG order.

The HA-Schedule-1 validator and evaluator remain independent recovery gates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import time

from .tensor_game_schedule_model import (
    ExecutionDag, LaunchGroup, Schedule, ScheduledOp, _digest, _schedule_payload,
    evaluate_schedule, validate_schedule,
)

OBJECTIVES = ("MIN_TEMP_HBM_TRAFFIC", "MIN_HBM_WRITES", "MIN_PEAK_HBM_TEMP",
              "MIN_LAUNCH_GROUPS", "MIN_RECOMPUTE_OPS", "LEX_TEMP_RECOMPUTE_LAUNCH")


@dataclass(frozen=True)
class SearchConfig:
    on_chip_capacity_bytes: int
    max_groups: int = 40
    max_matmuls_per_group: int = 1
    recomputation_budget_operations: int = 0
    objective: str = "LEX_TEMP_RECOMPUTE_LAUNCH"
    temporary_traffic_cap_bytes: int | None = None
    seed: int = 20261018
    workers: int = 1
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if (self.on_chip_capacity_bytes < 0 or self.max_groups < 1
                or self.max_matmuls_per_group < 1 or self.recomputation_budget_operations < 0):
            raise ValueError("capacity, group count, and recomputation budget must be nonnegative")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective: {self.objective}")
        if self.objective == "MIN_RECOMPUTE_OPS" and self.temporary_traffic_cap_bytes is None:
            raise ValueError("MIN_RECOMPUTE_OPS needs a temporary traffic cap")
        if self.temporary_traffic_cap_bytes is not None and self.temporary_traffic_cap_bytes < 0:
            raise ValueError("temporary traffic cap must be nonnegative")
        if self.workers < 1 or self.timeout_seconds <= 0:
            raise ValueError("workers and timeout must be positive")


@dataclass(frozen=True)
class SearchOutcome:
    status: str
    schedule: Schedule | None
    metrics: dict | None
    predicted: dict | None
    agreement: str
    stages: tuple[dict, ...]
    solver_summary: dict
    diagnostics: tuple[str, ...] = ()


def fixed_operation_order(dag: ExecutionDag) -> tuple[str, ...]:
    """Term-major deterministic topological order; forms appear at first use."""
    nodes = {node.node_id: node for node in dag.nodes}
    output_for_acc = {node.dependencies[0]: node.node_id for node in dag.nodes
                      if node.operation == "OUTPUT"}
    seen: set[str] = set()
    order: list[str] = []
    for term in dag.terms:
        for source in (term.u_source, term.v_source):
            if nodes[source].operation == "LINEAR_FORM" and source not in seen:
                order.append(source)
                seen.add(source)
        order.append(term.product_id)
        seen.add(term.product_id)
        for accumulator in term.accumulation_ids:
            order.append(accumulator)
            seen.add(accumulator)
            if accumulator in output_for_acc:
                output = output_for_acc[accumulator]
                order.append(output)
                seen.add(output)
    required = {node.node_id for node in dag.nodes if node.operation != "INPUT"}
    if seen != required or len(order) != len(seen):
        raise ValueError("fixed order does not cover the executable DAG exactly")
    available = {node.node_id for node in dag.nodes if node.operation == "INPUT"}
    for node_id in order:
        if any(dep not in available for dep in nodes[node_id].dependencies):
            raise ValueError(f"fixed order is not topological at {node_id}")
        available.add(node_id)
    return tuple(order)


def eligible_recompute_forms(dag: ExecutionDag) -> tuple[str, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    uses = Counter(dep for node in dag.nodes for dep in node.dependencies)
    return tuple(node.node_id for node in dag.nodes
                 if node.operation == "LINEAR_FORM" and uses[node.node_id] >= 2
                 and 0 < node.arithmetic_operations <= 2 * node.element_count
                 and all(nodes[dep].operation == "INPUT" for dep in node.dependencies))


def rebind_schedule(source_dag: ExecutionDag, target_dag: ExecutionDag,
                    schedule: Schedule) -> Schedule:
    """Reuse structural decisions at another N with identical dtype ratios."""
    if (source_dag.actions != target_dag.actions or source_dag.algorithm != target_dag.algorithm
            or source_dag.config.input_dtype != target_dag.config.input_dtype
            or source_dag.config.accumulation_dtype != target_dag.config.accumulation_dtype
            or source_dag.config.output_dtype != target_dag.config.output_dtype
            or tuple((node.node_id, node.operation, node.dependencies) for node in source_dag.nodes)
            != tuple((node.node_id, node.operation, node.dependencies) for node in target_dag.nodes)):
        raise ValueError("DAGs do not have the same normalized schedule structure")
    if schedule.dag_hash != source_dag.dag_hash:
        raise ValueError("source schedule does not match source DAG")
    pending = Schedule(schedule.schedule_id, schedule.policy, target_dag.dag_hash,
                       schedule.groups, schedule.recompute_allowed, "")
    result = Schedule(pending.schedule_id, pending.policy, pending.dag_hash,
                      pending.groups, pending.recompute_allowed, _digest(_schedule_payload(pending)))
    if not validate_schedule(target_dag, result).valid:
        raise ValueError("normalized schedule is invalid at target matrix size")
    return result


def _recovered_schedule(dag: ExecutionDag, order: tuple[str, ...], groups: list[int],
                        materialized: dict[str, bool], recompute: dict[str, bool]) -> Schedule:
    nodes = {node.node_id: node for node in dag.nodes}
    seen_consumers: Counter[str] = Counter()
    grouped: list[list[ScheduledOp]] = [[] for _ in range(max(groups) + 1)]
    for position, node_id in enumerate(order):
        node = nodes[node_id]
        for dependency in node.dependencies:
            if dependency in recompute:
                if seen_consumers[dependency] and recompute[dependency]:
                    grouped[groups[position]].append(ScheduledOp(dependency, "EPHEMERAL"))
                seen_consumers[dependency] += 1
        if node.operation == "OUTPUT" or materialized.get(node_id, False):
            location = "HBM"
        elif recompute.get(node_id, False):
            location = "EPHEMERAL"
        else:
            location = "ON_CHIP"
        grouped[groups[position]].append(ScheduledOp(node_id, location))
    launches = tuple(LaunchGroup(f"G{index:03d}", tuple(operations))
                     for index, operations in enumerate(grouped))
    if any(not group.operations for group in launches):
        raise ValueError("solver recovery produced an empty launch group")
    # Provenance lives beside the schedule; its content hash is purely semantic.
    schedule_id = f"{dag.algorithm}:SOLVER_SEARCH"
    pending = Schedule(schedule_id, "SOLVER_SEARCH", dag.dag_hash, launches,
                       tuple(sorted(node_id for node_id, enabled in recompute.items() if enabled)), "")
    return Schedule(pending.schedule_id, pending.policy, pending.dag_hash,
                    pending.groups, pending.recompute_allowed, _digest(_schedule_payload(pending)))


def solve_schedule(dag: ExecutionDag, config: SearchConfig) -> SearchOutcome:
    """Optimize one fixed-order schedule and gate it with independent accounting."""
    try:
        import ortools
        from ortools.sat.python import cp_model
    except ImportError:
        return SearchOutcome("DEPENDENCY_MISSING", None, None, None, "NOT_RUN", (),
                             {"solver": "OR-Tools CP-SAT", "version": None},
                             ("Install the optional constraint-search extra (OR-Tools).",))
    order = fixed_operation_order(dag)
    nodes = {node.node_id: node for node in dag.nodes}
    position = {node_id: index for index, node_id in enumerate(order)}
    consumers: dict[str, list[str]] = defaultdict(list)
    for node_id in order:
        for dependency in nodes[node_id].dependencies:
            if dependency in position:
                consumers[dependency].append(node_id)
    eligible = eligible_recompute_forms(dag)
    model = cp_model.CpModel()
    group = [model.NewIntVar(0, config.max_groups - 1, f"group_{index}") for index in range(len(order))]
    model.Add(group[0] == 0)
    for index in range(1, len(order)):
        model.Add(group[index] >= group[index - 1])
        model.Add(group[index] <= group[index - 1] + 1)
    matmul_positions = [index for index, node_id in enumerate(order)
                        if nodes[node_id].operation == "MATMUL"]
    for index in range(config.max_matmuls_per_group, len(matmul_positions)):
        model.Add(group[matmul_positions[index]] >
                  group[matmul_positions[index - config.max_matmuls_per_group]])
    group_count = model.NewIntVar(1, config.max_groups, "group_count")
    model.Add(group_count == group[-1] + 1)
    materialized = {node_id: model.NewBoolVar(f"hbm_{node_id}") for node_id in order
                    if nodes[node_id].materializable}
    recompute = {node_id: model.NewBoolVar(f"recompute_{node_id}") for node_id in eligible}
    for node_id, decision in recompute.items():
        model.Add(materialized[node_id] + decision <= 1)
        model.Add(group[position[node_id]] < group[position[consumers[node_id][-1]]]).OnlyEnforceIf(decision)
    recompute_ops = sum((len(consumers[node_id]) - 1) * nodes[node_id].arithmetic_operations * decision
                        for node_id, decision in recompute.items())
    model.Add(recompute_ops <= config.recomputation_budget_operations)
    local_uses: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for node_id in order:
        for use_index, consumer_id in enumerate(consumers[node_id]):
            same = model.NewBoolVar(f"same_{node_id}_{consumer_id}")
            model.Add(group[position[node_id]] == group[position[consumer_id]]).OnlyEnforceIf(same)
            model.Add(group[position[node_id]] != group[position[consumer_id]]).OnlyEnforceIf(same.Not())
            rec = recompute.get(node_id) if use_index else None
            material = materialized[node_id]
            model.Add(material + same + (rec if rec is not None else 0) >= 1)
            local = model.NewBoolVar(f"local_{node_id}_{consumer_id}")
            model.Add(local <= same)
            model.Add(local + material <= 1)
            if rec is not None:
                model.Add(local + rec <= 1)
                model.Add(local >= same - material - rec)
            else:
                model.Add(local >= same - material)
            local_uses[node_id].append((position[consumer_id], local))
    traffic = sum(nodes[node_id].byte_count * (1 + len(consumers[node_id])) * decision
                  for node_id, decision in materialized.items())
    writes = sum(nodes[node_id].byte_count * decision
                 for node_id, decision in materialized.items())
    if config.temporary_traffic_cap_bytes is not None:
        model.Add(traffic <= config.temporary_traffic_cap_bytes)
    max_local = sum(nodes[node_id].byte_count for node_id in materialized)
    local_loads = []
    for t, current in enumerate(order):
        live_terms = []
        for node_id in materialized:
            if position[node_id] <= t:
                eligible_uses = [local for use_pos, local in local_uses[node_id] if use_pos >= t]
                if eligible_uses:
                    live = model.NewBoolVar(f"live_{node_id}_{t}")
                    model.AddMaxEquality(live, eligible_uses)
                    live_terms.append(nodes[node_id].byte_count * live)
        if current in {consumer for node_id in eligible for consumer in consumers[node_id][1:]}:
            for node_id in eligible:
                if current in consumers[node_id][1:]:
                    live_terms.append(nodes[node_id].byte_count * recompute[node_id])
        load = model.NewIntVar(0, max_local, f"local_load_{t}")
        model.Add(load == sum(live_terms))
        model.Add(load <= config.on_chip_capacity_bytes)
        local_loads.append(load)
    peak_local = model.NewIntVar(0, max_local, "peak_local")
    model.AddMaxEquality(peak_local, local_loads)
    hbm_loads = []
    for t in range(len(order)):
        expression = sum(nodes[node_id].byte_count * materialized[node_id]
                         for node_id in materialized
                         if position[node_id] <= t <= max(position[c] for c in consumers[node_id]))
        load = model.NewIntVar(0, max_local, f"hbm_load_{t}")
        model.Add(load == expression)
        hbm_loads.append(load)
    peak_hbm = model.NewIntVar(0, max_local, "peak_hbm")
    model.AddMaxEquality(peak_hbm, hbm_loads)
    stages = []
    if config.objective == "LEX_TEMP_RECOMPUTE_LAUNCH":
        objectives = (("temporary_hbm_traffic_bytes", traffic),
                      ("recomputation_arithmetic_operations", recompute_ops),
                      ("logical_launch_groups", group_count))
    else:
        objectives = ((config.objective, {
            "MIN_TEMP_HBM_TRAFFIC": traffic,
            "MIN_HBM_WRITES": writes,
            "MIN_PEAK_HBM_TEMP": peak_hbm,
            "MIN_LAUNCH_GROUPS": group_count,
            "MIN_RECOMPUTE_OPS": recompute_ops,
        }[config.objective]),)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = config.workers
    solver.parameters.random_seed = config.seed
    solver.parameters.randomize_search = False
    solver.parameters.log_search_progress = False
    started = time.monotonic()
    status = "NOT_STARTED"
    for stage_name, expression in objectives:
        remaining = config.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            status = "TIME_LIMIT"
            break
        model.Minimize(expression)
        solver.parameters.max_time_in_seconds = remaining
        code = solver.Solve(model)
        status = solver.StatusName(code)
        stage = {"objective": stage_name, "status": status,
                 "wall_seconds": solver.WallTime(), "branches": solver.NumBranches(),
                 "conflicts": solver.NumConflicts(),
                 "value": int(solver.Value(expression)) if code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None}
        stages.append(stage)
        if code != cp_model.OPTIMAL:
            break
        model.Add(expression == stage["value"])
    summary = {"solver": "OR-Tools CP-SAT", "version": ortools.__version__,
               "seed": config.seed, "workers": config.workers, "timeout_seconds": config.timeout_seconds,
               "max_matmuls_per_group": config.max_matmuls_per_group,
               "deterministic_settings": {"randomize_search": False, "single_worker": config.workers == 1},
               "variables": len(model.Proto().variables), "constraints": len(model.Proto().constraints),
               "branches": sum(stage["branches"] for stage in stages),
               "conflicts": sum(stage["conflicts"] for stage in stages),
               "solve_wall_seconds": time.monotonic() - started,
               "eligible_recompute_forms": eligible, "fixed_order": order}
    if status not in ("OPTIMAL", "FEASIBLE"):
        return SearchOutcome(status, None, None, None, "NOT_RUN", tuple(stages), summary)
    if config.objective == "LEX_TEMP_RECOMPUTE_LAUNCH" and len(stages) != len(objectives):
        return SearchOutcome("PARTIAL", None, None, None, "NOT_RUN", tuple(stages), summary,
                             ("lexicographic stages did not all complete",))
    selected_groups = [solver.Value(variable) for variable in group]
    selected_hbm = {node_id: bool(solver.Value(variable)) for node_id, variable in materialized.items()}
    selected_recompute = {node_id: bool(solver.Value(variable)) for node_id, variable in recompute.items()}
    predicted = {"temporary_hbm_traffic_bytes": solver.Value(traffic),
                 "temporary_hbm_bytes_written": solver.Value(writes),
                 "recomputation_arithmetic_operations": solver.Value(recompute_ops),
                 "number_of_recomputed_intermediates": sum((len(consumers[node_id]) - 1) * enabled
                                                              for node_id, enabled in selected_recompute.items()),
                 "logical_launch_groups": solver.Value(group_count),
                 "peak_abstract_local_bytes": solver.Value(peak_local),
                 "peak_temp_hbm_bytes": solver.Value(peak_hbm)}
    try:
        schedule = _recovered_schedule(dag, order, selected_groups, selected_hbm, selected_recompute)
        validation = validate_schedule(dag, schedule)
        if not validation.valid:
            raise ValueError(f"validator issues: {[asdict(issue) for issue in validation.issues]}")
        metrics = evaluate_schedule(dag, schedule)
    except ValueError as exc:
        return SearchOutcome("MODEL_INVALID", None, None, predicted, "FAIL", tuple(stages), summary, (str(exc),))
    mismatches = tuple(f"{name}: solver={value}, evaluator={metrics[name]}"
                       for name, value in predicted.items() if metrics[name] != value)
    if metrics["peak_abstract_local_bytes"] > config.on_chip_capacity_bytes:
        mismatches += ("independent peak exceeds capacity",)
    if mismatches:
        return SearchOutcome("MODEL_INVALID", None, None, predicted, "FAIL", tuple(stages), summary, mismatches)
    return SearchOutcome(status, schedule, metrics, predicted, "PASS", tuple(stages), summary)
