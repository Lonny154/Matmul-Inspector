"""CPU-only bounded HBM lifetime contract for retained execution schedules.

The contract records intentional HBM values, not inferred GPU cache traffic.
Only singleton groups, one-GEMM groups, and one accumulation/finalization
group are executable. Existing HA-6/7 plans retain their original identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .tensor_game_schedule_calibration import stable_hash
from .tensor_game_schedule_lowering import ScheduleLoweringPlan
from .tensor_game_schedule_mixed_lowering import classify_mixed_schedule, mixed_capability_manifest
from .tensor_game_schedule_model import ExecutionDag, Schedule, validate_schedule

FAMILY_HBM = "HBM_BACKED_MIXED_DAG"
HBM_VERSION = "ha-schedule-8-v1"
VALUE_CLASSES = {"LINEAR_FORM": "PREPARED_FORM", "MATMUL": "GEMM_PRODUCT",
                 "ACCUMULATE": "PARTIAL_OUTPUT", "OUTPUT": "FINAL_OUTPUT"}
READ_TYPES = {"PREPARED_FORM": "READ_FORM", "GEMM_PRODUCT": "READ_PRODUCT",
              "PARTIAL_OUTPUT": "READ_PARTIAL"}


@dataclass(frozen=True)
class HBMValueLifetime:
    value_id: str
    value_class: str
    producer_group: str
    producer_launch_role: str
    first_consumer_group: str | None
    consumer_groups: tuple[str, ...]
    consumer_node_ids: tuple[str, ...]
    read_semantics: tuple[str, ...]
    final_consumer_group: str | None
    byte_size: int
    dtype: str
    write_semantics: str
    accumulation_state: str
    finalization_state: str
    source_schedule_hash: str
    lowering_plan_hash: str
    provenance: str


@dataclass(frozen=True)
class HBMBufferPlan:
    value_id: str
    buffer_id: str
    byte_size: int
    dtype: str
    first_write_group: str
    last_use_group: str
    output_block: int | None


@dataclass(frozen=True)
class HBMBackedPlan:
    schedule_plan: ScheduleLoweringPlan
    hbm_lifetimes: tuple[HBMValueLifetime, ...]
    buffer_plan: tuple[HBMBufferPlan, ...]
    contribution_order: tuple[dict, ...]
    lowering_plan_hash: str


def hbm_capability_manifest() -> dict:
    base = mixed_capability_manifest()
    return {"version": HBM_VERSION, "families": [*base["families"], {
        "family_id": FAMILY_HBM,
        "supported_algorithms": ["conventional", "strassen", "project-rank8"],
        "supported_features": ["intentional FP16 prepared forms in HBM",
                               "intentional FP32 products in HBM", "FP32 partial-output chains",
                               "same-launch HBM partial plus final output write",
                               "one GEMM and at most two direct reconstruction targets per group"],
        "forbidden_features": ["multiple GEMMs per group", "nonlocal group work",
                               "more than two direct targets", "duplicate output targets",
                               "in-place buffer reuse", "concurrent partial writes"],
        "mapping_capability": "EXACT_MAPPING",
        "implementation": "matmul_inspector.tensor_game_schedule_hbm_runtime.execute_hbm",
        "tested_sizes": [128, 256, 512, 1024], "tested_dtypes": ["fp16/fp32/fp16"],
    }]}


def _target(nodes: dict, accumulation, final_id: str | None) -> dict:
    node = nodes[accumulation.node_id]
    previous = node.dependencies[0] if len(node.dependencies) == 2 else None
    return {"accumulation_id": node.node_id, "output_block": node.output_block,
            "coefficient": node.coefficients[0], "product_id": node.dependencies[-1],
            "previous_id": previous, "mode": "READ_MODIFY_WRITE" if previous else "INITIALIZE",
            "write_partial": accumulation.location == "HBM", "write_final": final_id is not None,
            "output_id": final_id}


def hbm_group_specs(dag: ExecutionDag, schedule: Schedule) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    """Recognize only physically bounded group shapes, retaining exact order."""
    validation = validate_schedule(dag, schedule)
    if not validation.valid:
        raise ValueError(f"illegal schedule: {validation.issues}")
    nodes = {node.node_id: node for node in dag.nodes}
    reasons: list[str] = []
    if (dag.config.input_dtype, dag.config.accumulation_dtype, dag.config.output_dtype) != (
            "fp16", "fp32", "fp16"):
        reasons.append("DTYPE_TEMPLATE_MISSING")
    seen_hbm = set()
    launches = []
    for group in schedule.groups:
        ops = group.operations
        kinds = [nodes[op.node_id].operation for op in ops]
        forms = [index for index, kind in enumerate(kinds) if kind == "LINEAR_FORM"]
        products = [index for index, kind in enumerate(kinds) if kind == "MATMUL"]
        accumulations = [index for index, kind in enumerate(kinds) if kind == "ACCUMULATE"]
        outputs = [index for index, kind in enumerate(kinds) if kind == "OUTPUT"]
        if len(ops) == 1 and ops[0].location == "HBM":
            role = kinds[0]
            launches.append({"group_id": group.group_id, "role": role, "node_id": ops[0].node_id,
                             "kernel": {"LINEAR_FORM": "_materialize_form_kernel",
                                        "MATMUL": "_fp32_gemm_kernel",
                                        "ACCUMULATE": "_accumulate_kernel",
                                        "OUTPUT": "_output_kernel"}[role]})
        elif not products:
            if (len(ops) != 2 or len(accumulations) != 1 or len(outputs) != 1 or
                    accumulations[0] >= outputs[0] or
                    nodes[ops[outputs[0]].node_id].dependencies != (ops[accumulations[0]].node_id,)):
                reasons.append("NON_GEMM_GROUP_SHAPE_UNSUPPORTED")
            else:
                accumulation = ops[accumulations[0]]
                node = nodes[accumulation.node_id]
                if node.dependencies[-1] not in seen_hbm or (
                        len(node.dependencies) == 2 and node.dependencies[0] not in seen_hbm):
                    reasons.append("NONLOCAL_RECONSTRUCTION_UNSUPPORTED")
                launches.append({"group_id": group.group_id, "role": "HBM_FINALIZE",
                                 "node_id": accumulation.node_id, "kernel": "_hbm_accumulate_output_kernel",
                                 "target": _target(nodes, accumulation, ops[outputs[0]].node_id)})
        elif len(products) != 1:
            reasons.append("MULTI_GEMM_GROUP_UNSUPPORTED")
        else:
            product_index = products[0]
            product = ops[product_index]
            product_node = nodes[product.node_id]
            local_forms = {ops[index].node_id: ops[index] for index in forms}
            if (len(forms) > 2 or any(index >= product_index for index in forms) or
                    any(source not in product_node.dependencies for source in local_forms)):
                reasons.append("NONLOCAL_FORM_WORK_UNSUPPORTED")
            if any(index <= product_index for index in accumulations):
                reasons.append("RECONSTRUCTION_BEFORE_GEMM_UNSUPPORTED")
            if len(accumulations) > 2:
                reasons.append("RECONSTRUCTION_FANOUT_GT2_UNSUPPORTED")
            if any(product.node_id not in nodes[ops[index].node_id].dependencies
                   for index in accumulations):
                reasons.append("NONLOCAL_RECONSTRUCTION_UNSUPPORTED")
            if len({nodes[ops[index].node_id].output_block for index in accumulations}) != len(accumulations):
                reasons.append("DUPLICATE_OUTPUT_TARGET_UNSUPPORTED")
            if not accumulations and product.location != "HBM":
                reasons.append("UNMATERIALIZED_UNCONSUMED_PRODUCT_UNSUPPORTED")
            accumulation_ids = {ops[index].node_id for index in accumulations}
            if any(index <= product_index or nodes[ops[index].node_id].dependencies[0] not in accumulation_ids
                   for index in outputs):
                reasons.append("NONLOCAL_OUTPUT_UNSUPPORTED")
            if len(ops) != len(forms) + 1 + len(accumulations) + len(outputs):
                reasons.append("OTHER_GROUP_WORK_UNSUPPORTED")
            source_modes = []
            for source in product_node.dependencies:
                if source in local_forms:
                    source_modes.append("LOCAL_FORM")
                elif nodes[source].operation == "INPUT":
                    source_modes.append("INPUT")
                elif source in seen_hbm:
                    source_modes.append("HBM_FORM")
                else:
                    reasons.append("MISSING_PREPARED_FORM_UNSUPPORTED")
                    source_modes.append("UNAVAILABLE")
            targets = []
            for index in accumulations:
                accumulation = ops[index]
                node = nodes[accumulation.node_id]
                final = next((ops[oi].node_id for oi in outputs if oi > index and
                              nodes[ops[oi].node_id].dependencies == (node.node_id,)), None)
                if accumulation.location == "ON_CHIP" and final is None:
                    reasons.append("UNFINALIZED_LOCAL_PARTIAL_UNSUPPORTED")
                if len(node.dependencies) == 2 and node.dependencies[0] not in seen_hbm:
                    reasons.append("NONLOCAL_PARTIAL_UNSUPPORTED")
                targets.append(_target(nodes, accumulation, final))
            launches.append({"group_id": group.group_id, "role": FAMILY_HBM,
                             "node_id": product.node_id, "kernel": "_hbm_gemm_group_kernel",
                             "fused_gemm_id": product.node_id,
                             "input_sources": product_node.dependencies,
                             "input_source_modes": tuple(source_modes),
                             "local_forms": tuple(op.node_id for op in local_forms.values()),
                             "form_writes": tuple(op.node_id for op in local_forms.values() if op.location == "HBM"),
                             "product_write": product.location == "HBM",
                             "reconstruction_targets": tuple(targets)})
        seen_hbm.update(op.node_id for op in ops if op.location == "HBM")
    return tuple(launches), tuple(dict.fromkeys(reasons))


def _contribution_order(dag: ExecutionDag, schedule: Schedule) -> tuple[dict, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    result = []
    for block in range(4):
        expected = [node.node_id for node in dag.nodes if node.operation == "ACCUMULATE" and node.output_block == block]
        actual = [op.node_id for group in schedule.groups for op in group.operations
                  if nodes[op.node_id].operation == "ACCUMULATE" and nodes[op.node_id].output_block == block]
        if actual != expected:
            raise ValueError(f"noncanonical contribution order for output block {block}")
        result.append({"output_block": block, "accumulation_ids": tuple(actual),
                       "term_indices": tuple(nodes[node_id].term_index for node_id in actual),
                       "coefficients": tuple(nodes[node_id].coefficients[0] for node_id in actual)})
    return tuple(result)


def _lifetime_payload(lifetime: HBMValueLifetime) -> dict:
    result = asdict(lifetime)
    result.pop("lowering_plan_hash")
    return result


def _plan_payload(schedule_plan: ScheduleLoweringPlan, lifetimes, buffers, contributions) -> dict:
    return {"schedule_plan_hash": schedule_plan.lowering_plan_hash,
            "lifetimes": [_lifetime_payload(lifetime) for lifetime in lifetimes],
            "buffers": [asdict(buffer) for buffer in buffers],
            "contribution_order": list(contributions), "version": HBM_VERSION}


def _build_lifetimes(dag: ExecutionDag, schedule: Schedule, launches: tuple[dict, ...],
                     schedule_plan: ScheduleLoweringPlan) -> HBMBackedPlan:
    nodes = {node.node_id: node for node in dag.nodes}
    group_index = {group.group_id: index for index, group in enumerate(schedule.groups)}
    produced = {op.node_id: (group.group_id, launch["role"])
                for group, launch in zip(schedule.groups, launches) for op in group.operations if op.location == "HBM"}
    uses = {node_id: [] for node_id in produced}
    for group in schedule.groups:
        for op in group.operations:
            for dependency in nodes[op.node_id].dependencies:
                if dependency in produced:
                    owner = produced[dependency][0]
                    value_class = VALUE_CLASSES[nodes[dependency].operation]
                    semantics = "LOCAL_FORWARD" if owner == group.group_id else READ_TYPES[value_class]
                    uses[dependency].append((group.group_id, op.node_id, semantics))
    lifetimes = []
    buffers = []
    for group in schedule.groups:
        for op in group.operations:
            if op.location != "HBM":
                continue
            node = nodes[op.node_id]
            value_class = VALUE_CLASSES[node.operation]
            consumer = uses[node.node_id]
            if node.operation == "ACCUMULATE":
                write = "WRITE_PARTIAL" if len(node.dependencies) == 1 else "UPDATE_PARTIAL"
            elif node.operation == "OUTPUT":
                write = "WRITE_FINAL"
            else:
                write = "WRITE_NEW"
            lifetime = HBMValueLifetime(node.node_id, value_class, group.group_id,
                produced[node.node_id][1], consumer[0][0] if consumer else None,
                tuple(use[0] for use in consumer), tuple(use[1] for use in consumer),
                tuple(use[2] for use in consumer), consumer[-1][0] if consumer else None,
                node.byte_count, node.dtype, write,
                "PARTIAL" if value_class == "PARTIAL_OUTPUT" else
                "FINAL" if value_class == "FINAL_OUTPUT" else "NONE",
                "FINALIZED" if value_class == "FINAL_OUTPUT" else "NOT_FINALIZED",
                schedule.schedule_hash, "", dag.dag_hash)
            lifetimes.append(lifetime)
            buffers.append(HBMBufferPlan(node.node_id, "output" if node.operation == "OUTPUT" else
                                         f"buf:{node.node_id}", node.byte_count, node.dtype,
                                         group.group_id, consumer[-1][0] if consumer else group.group_id,
                                         node.output_block if node.operation == "OUTPUT" else None))
    contributions = _contribution_order(dag, schedule)
    digest = stable_hash(_plan_payload(schedule_plan, lifetimes, buffers, contributions))
    linked = tuple(replace(lifetime, lowering_plan_hash=digest) for lifetime in lifetimes)
    result = HBMBackedPlan(schedule_plan, linked, tuple(buffers), contributions, digest)
    validate_hbm_plan(dag, schedule, result)
    return result


def classify_hbm_schedule(dag: ExecutionDag, schedule: Schedule, *, lowering_name: str | None = None,
                          static_metrics: dict | None = None,
                          calibration_features: dict | None = None) -> ScheduleLoweringPlan | HBMBackedPlan:
    """Keep old exact/approximate plans byte-for-byte; add one HBM family."""
    previous = classify_mixed_schedule(dag, schedule, lowering_name=lowering_name,
                                       static_metrics=static_metrics,
                                       calibration_features=calibration_features)
    if previous.mapping_status != "UNSUPPORTED":
        return previous
    launches, reasons = hbm_group_specs(dag, schedule)
    if reasons:
        payload = {key: value for key, value in asdict(previous).items() if key != "lowering_plan_hash"}
        payload["unsupported_features"] = reasons
        payload["lowering_version"] = HBM_VERSION
        return ScheduleLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))
    nodes = {node.node_id: node for node in dag.nodes}
    ops = [op for group in schedule.groups for op in group.operations]
    payload = {"schedule_hash": schedule.schedule_hash, "algorithm": dag.algorithm,
               "decomposition_id": dag.decomposition_id, "family_id": FAMILY_HBM,
               "support_status": "LOWERABLE_WITH_TEMPLATE", "mapping_status": "EXACT_MAPPING",
               "launch_sequence": launches,
               "materialized_values": tuple(op.node_id for op in ops if op.location == "HBM" and
                                            nodes[op.node_id].operation != "OUTPUT"),
               "recomputed_values": tuple(sorted(schedule.recompute_allowed)),
               "fused_forms": tuple(op.node_id for op in ops if nodes[op.node_id].operation == "LINEAR_FORM" and
                                    op.location in ("ON_CHIP", "EPHEMERAL")),
               "reconstruction_strategy": "HBM_BACKED_BOUNDED_GROUPS",
               "expected_launch_count": len(launches), "unsupported_features": (),
               "lowering_version": HBM_VERSION,
               "static_metrics_hash": stable_hash(static_metrics) if static_metrics is not None else None,
               "calibration_features_hash": stable_hash(calibration_features) if calibration_features is not None else None}
    base = ScheduleLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))
    return _build_lifetimes(dag, schedule, launches, base)


def validate_hbm_plan(dag: ExecutionDag, schedule: Schedule, plan: HBMBackedPlan) -> dict:
    """Check typed producer/consumer order, buffer identity, and finalization."""
    if plan.schedule_plan.schedule_hash != schedule.schedule_hash or plan.schedule_plan.family_id != FAMILY_HBM:
        raise ValueError("HBM plan provenance or family mismatch")
    expected_launches, reasons = hbm_group_specs(dag, schedule)
    if reasons or plan.schedule_plan.launch_sequence != expected_launches:
        raise ValueError("HBM launch writes or group topology differ from schedule")
    if plan.lowering_plan_hash != stable_hash(_plan_payload(
            plan.schedule_plan, plan.hbm_lifetimes, plan.buffer_plan, plan.contribution_order)):
        raise ValueError("HBM lifetime plan hash mismatch")
    index = {group.group_id: i for i, group in enumerate(schedule.groups)}
    nodes = {node.node_id: node for node in dag.nodes}
    expected = {op.node_id for group in schedule.groups for op in group.operations if op.location == "HBM"}
    expected_intermediates = tuple(op.node_id for group in schedule.groups for op in group.operations
                                   if op.location == "HBM" and nodes[op.node_id].operation != "OUTPUT")
    if plan.schedule_plan.materialized_values != expected_intermediates:
        raise ValueError("hidden or missing HBM materialization")
    if {life.value_id for life in plan.hbm_lifetimes} != expected or len(plan.hbm_lifetimes) != len(expected):
        raise ValueError("planned HBM values differ from schedule")
    if {buffer.value_id for buffer in plan.buffer_plan} != expected or len(plan.buffer_plan) != len(expected):
        raise ValueError("buffer plan differs from HBM values")
    expected_uses = {node_id: [] for node_id in expected}
    producers = {op.node_id: group.group_id for group in schedule.groups
                 for op in group.operations if op.location == "HBM"}
    for group in schedule.groups:
        for op in group.operations:
            for dependency in nodes[op.node_id].dependencies:
                if dependency in expected:
                    owner = producers[dependency]
                    value_class = VALUE_CLASSES[nodes[dependency].operation]
                    read = "LOCAL_FORWARD" if owner == group.group_id else READ_TYPES[value_class]
                    expected_uses[dependency].append((group.group_id, op.node_id, read))
    seen_buffers: dict[str, list[HBMBufferPlan]] = {}
    for lifetime, buffer in zip(plan.hbm_lifetimes, plan.buffer_plan):
        if lifetime.value_id != buffer.value_id or lifetime.lowering_plan_hash != plan.lowering_plan_hash:
            raise ValueError("lifetime/buffer identity or hash mismatch")
        if lifetime.source_schedule_hash != schedule.schedule_hash or lifetime.provenance != dag.dag_hash:
            raise ValueError("lifetime provenance mismatch")
        if len(lifetime.consumer_groups) != len(lifetime.consumer_node_ids) or len(
                lifetime.consumer_groups) != len(lifetime.read_semantics):
            raise ValueError("consumer semantics length mismatch")
        recorded_uses = tuple(zip(lifetime.consumer_groups, lifetime.consumer_node_ids,
                                  lifetime.read_semantics))
        if recorded_uses != tuple(expected_uses[lifetime.value_id]):
            raise ValueError("planned HBM consumers differ from schedule")
        if lifetime.first_consumer_group != (recorded_uses[0][0] if recorded_uses else None) or (
                lifetime.final_consumer_group != (recorded_uses[-1][0] if recorded_uses else None)):
            raise ValueError("first or final HBM consumer mismatch")
        if buffer.first_write_group != lifetime.producer_group or buffer.byte_size != lifetime.byte_size:
            raise ValueError("buffer first write or size mismatch")
        if buffer.last_use_group != (lifetime.final_consumer_group or lifetime.producer_group):
            raise ValueError("buffer last-use mismatch")
        for group_id, read in zip(lifetime.consumer_groups, lifetime.read_semantics):
            if index[group_id] < index[lifetime.producer_group]:
                raise ValueError("consumer precedes producer")
            expected_read = ("LOCAL_FORWARD" if group_id == lifetime.producer_group else
                             READ_TYPES.get(lifetime.value_class))
            if read != expected_read:
                raise ValueError("typed HBM read mismatch")
        for old in seen_buffers.get(buffer.buffer_id, []):
            disjoint_outputs = (buffer.buffer_id == "output" and old.output_block is not None and
                                buffer.output_block is not None and old.output_block != buffer.output_block)
            nonoverlap = (index[old.last_use_group] < index[buffer.first_write_group] or
                          index[buffer.last_use_group] < index[old.first_write_group])
            if not (disjoint_outputs or nonoverlap):
                raise ValueError("overlapping HBM buffer reuse")
        seen_buffers.setdefault(buffer.buffer_id, []).append(buffer)
        if lifetime.value_class == "FINAL_OUTPUT":
            if lifetime.write_semantics != "WRITE_FINAL" or lifetime.consumer_groups:
                raise ValueError("final output was modified or consumed")
        elif lifetime.value_class == "PARTIAL_OUTPUT":
            if lifetime.write_semantics not in ("WRITE_PARTIAL", "UPDATE_PARTIAL"):
                raise ValueError("invalid partial write")
        elif lifetime.write_semantics != "WRITE_NEW":
            raise ValueError("invalid form/product write")
    checks = reference_transition_checks(dag, schedule, plan)
    if not all(check["passed"] for check in checks):
        raise ValueError("contribution state machine failed")
    return {"planned_hbm_values": len(plan.hbm_lifetimes),
            "planned_logical_buffers": len({buffer.buffer_id for buffer in plan.buffer_plan}),
            "contribution_checks": len(checks), "status": "PASS"}


def reference_transition_checks(dag: ExecutionDag, schedule: Schedule, plan: HBMBackedPlan) -> tuple[dict, ...]:
    """Independent scalar reference for exact contribution order and states."""
    nodes = {node.node_id: node for node in dag.nodes}
    recorded = {entry["output_block"]: entry for entry in plan.contribution_order}
    checks = []
    for block in range(4):
        expected = [node for node in dag.nodes if node.operation == "ACCUMULATE" and node.output_block == block]
        actual = [nodes[op.node_id] for group in schedule.groups for op in group.operations
                  if nodes[op.node_id].operation == "ACCUMULATE" and nodes[op.node_id].output_block == block]
        entry = recorded.get(block, {})
        expected_scalar = sum(node.coefficients[0] * (node.term_index + 1) for node in expected)
        value = 0
        state = "UNINITIALIZED"
        partials = []
        previous = None
        for node in actual:
            if state == "FINAL" or (previous is None and len(node.dependencies) != 1) or (
                    previous is not None and node.dependencies[0] != previous):
                state = "INVALID"
                break
            value += node.coefficients[0] * (node.term_index + 1)
            partials.append(value)
            previous = node.node_id
            state = "PARTIAL"
        output = nodes[dag.output_ids[block]]
        output_ops = [op for group in schedule.groups for op in group.operations if op.node_id == output.node_id]
        if len(output_ops) == 1 and output.dependencies == (previous,) and state == "PARTIAL":
            state = "FINAL"
        passed = (state == "FINAL" and actual == expected and value == expected_scalar and
                  tuple(node.node_id for node in actual) == tuple(entry.get("accumulation_ids", ())) and
                  tuple(node.coefficients[0] for node in actual) == tuple(entry.get("coefficients", ())))
        checks.append({"output_block": block, "state": state, "contribution_ids": tuple(node.node_id for node in actual),
                       "synthetic_partial_values": tuple(partials), "synthetic_final_value": value,
                       "expected_synthetic_final_value": expected_scalar, "passed": passed})
    return tuple(checks)


def hbm_lowerability_query(dag: ExecutionDag, schedule: Schedule, **kwargs) -> dict:
    result = classify_hbm_schedule(dag, schedule, **kwargs)
    base = result.schedule_plan if isinstance(result, HBMBackedPlan) else result
    return {"supported": base.mapping_status != "UNSUPPORTED", "exact": base.mapping_status == "EXACT_MAPPING",
            "family_id": base.family_id, "schedule_hash": base.schedule_hash,
            "lowering_plan_hash": result.lowering_plan_hash,
            "reasons": list(base.unsupported_features)}
