"""Bounded one-GEMM/one-reconstruction-stage schedule lowering (CPU only).

No arbitrary DAG fusion is attempted. A supported mixed group contains one
ON_CHIP product, at most two direct accumulation targets, and no HBM form
preparation or independent work in the same launch.
"""

from __future__ import annotations

from dataclasses import asdict

from .tensor_game_schedule_calibration import stable_hash
from .tensor_game_schedule_lowering import (
    ScheduleLoweringPlan, capability_manifest, classify_schedule,
)
from .tensor_game_schedule_model import ExecutionDag, Schedule, validate_schedule

FAMILY_MIXED = "FUSED_GEMM_RECONSTRUCTION"
MIXED_VERSION = "ha-schedule-7-v1"


def mixed_capability_manifest() -> dict:
    base = capability_manifest()
    return {"version": MIXED_VERSION, "families": [*base["families"], {
        "family_id": FAMILY_MIXED,
        "supported_algorithms": ["conventional", "strassen", "project-rank8"],
        "supported_features": ["one ON_CHIP GEMM", "one or two immediate accumulation targets",
                               "direct FP16-rounded input forms", "HBM prior partial reads",
                               "HBM partial or final FP16 output writes"],
        "forbidden_features": ["multi-GEMM group", "co-located HBM form preparation",
                               "more than two reconstruction targets", "independent operations",
                               "nested recomputation", "HBM product in mixed group"],
        "mapping_capability": "EXACT_MAPPING",
        "implementation": "matmul_inspector.tensor_game_schedule_mixed_runtime.execute_mixed",
        "tested_sizes": [128, 256, 512, 1024],
        "tested_dtypes": ["fp16/fp32/fp16"],
    }]}


def mixed_group_specs(dag: ExecutionDag, schedule: Schedule) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    """Describe every launch or return explicit unsupported semantic reasons."""
    validation = validate_schedule(dag, schedule)
    if not validation.valid:
        raise ValueError(f"illegal schedule: {validation.issues}")
    nodes = {node.node_id: node for node in dag.nodes}
    reasons: list[str] = []
    if (dag.config.input_dtype, dag.config.accumulation_dtype, dag.config.output_dtype) != (
            "fp16", "fp32", "fp16"):
        reasons.append("DTYPE_TEMPLATE_MISSING")
    specifications = []
    mixed_count = 0
    for group in schedule.groups:
        ops = group.operations
        if len(ops) == 1 and ops[0].location == "HBM":
            specifications.append({"group_id": group.group_id, "role": nodes[ops[0].node_id].operation,
                                   "node_id": ops[0].node_id,
                                   "kernel": {"LINEAR_FORM": "_materialize_form_kernel",
                                              "MATMUL": "_fp32_gemm_kernel",
                                              "ACCUMULATE": "_accumulate_kernel",
                                              "OUTPUT": "_output_kernel"}[nodes[ops[0].node_id].operation]})
            continue
        products = [op for op in ops if nodes[op.node_id].operation == "MATMUL"]
        accumulations = [op for op in ops if nodes[op.node_id].operation == "ACCUMULATE"]
        forms = [op for op in ops if nodes[op.node_id].operation == "LINEAR_FORM"]
        outputs = [op for op in ops if nodes[op.node_id].operation == "OUTPUT"]
        if len(products) != 1 or not accumulations:
            reasons.append("NON_GEMM_MULTI_OPERATION_GROUP_UNSUPPORTED")
            continue
        mixed_count += 1
        product = products[0]
        product_node = nodes[product.node_id]
        if product.location != "ON_CHIP":
            reasons.append("MIXED_HBM_PRODUCT_UNSUPPORTED")
        if not 1 <= len(accumulations) <= 2:
            reasons.append("RECONSTRUCTION_FANOUT_GT2_UNSUPPORTED")
        if len(forms) > 2 or any(op.location == "HBM" for op in forms):
            reasons.append("COLOCATED_FORM_MATERIALIZATION_UNSUPPORTED")
        if any(op.node_id not in product_node.dependencies for op in forms):
            reasons.append("NON_OPERAND_FORM_IN_GEMM_GROUP_UNSUPPORTED")
        if any(nodes[op.node_id].operation == "LINEAR_FORM" and
               any(nodes[dep].operation != "INPUT" for dep in nodes[op.node_id].dependencies)
               for op in forms):
            reasons.append("NESTED_RECOMPUTATION_UNSUPPORTED")
        if any(product.node_id not in nodes[op.node_id].dependencies for op in accumulations):
            reasons.append("NONLOCAL_RECONSTRUCTION_UNSUPPORTED")
        if len({nodes[op.node_id].output_block for op in accumulations}) != len(accumulations):
            reasons.append("DUPLICATE_OUTPUT_TARGET_UNSUPPORTED")
        if any(nodes[op.node_id].dependencies[0] not in {item.node_id for item in accumulations}
               for op in outputs):
            reasons.append("NONLOCAL_OUTPUT_UNSUPPORTED")
        if len(ops) != len(products) + len(accumulations) + len(forms) + len(outputs):
            reasons.append("UNKNOWN_MIXED_OPERATION_UNSUPPORTED")
        if len(outputs) > 2:
            reasons.append("OUTPUT_FANOUT_GT2_UNSUPPORTED")
        targets = []
        for accumulation in accumulations:
            node = nodes[accumulation.node_id]
            final = next((op for op in outputs if nodes[op.node_id].dependencies == (node.node_id,)), None)
            if (accumulation.location == "ON_CHIP") != (final is not None):
                reasons.append("PARTIAL_OUTPUT_LIFETIME_UNSUPPORTED")
            if final is not None and final.location != "HBM":
                reasons.append("OUTPUT_LOCATION_UNSUPPORTED")
            previous = node.dependencies[0] if len(node.dependencies) == 2 else None
            if previous is not None and nodes[previous].operation != "ACCUMULATE":
                reasons.append("NONLOCAL_RECONSTRUCTION_UNSUPPORTED")
            targets.append({"accumulation_id": node.node_id, "output_block": node.output_block,
                            "coefficient": node.coefficients[0], "previous_id": previous,
                            "mode": "READ_MODIFY_WRITE" if previous else "INITIALIZE",
                            "write": "FINAL_FP16" if final else "PARTIAL_FP32",
                            "output_id": final.node_id if final else None})
        form_ids = {op.node_id for op in forms}
        for source in product_node.dependencies:
            if nodes[source].operation == "LINEAR_FORM" and source not in form_ids:
                producers = [op for prior in schedule.groups[:schedule.groups.index(group)]
                             for op in prior.operations if op.node_id == source and op.location == "HBM"]
                if not producers:
                    reasons.append("NONLOCAL_FORM_LIFETIME_UNSUPPORTED")
        specifications.append({"group_id": group.group_id, "role": FAMILY_MIXED,
                               "node_id": product.node_id, "kernel": "_fused_gemm_reconstruction_kernel",
                               "fused_gemm_id": product.node_id, "input_sources": product_node.dependencies,
                               "local_forms": tuple(op.node_id for op in forms),
                               "reconstruction_targets": tuple(targets),
                               "product_materialized": False,
                               "expected_launch_role": FAMILY_MIXED})
    if not mixed_count:
        reasons.append("NO_BOUNDED_MIXED_GROUP")
    return tuple(specifications), tuple(dict.fromkeys(reasons))


def classify_mixed_schedule(dag: ExecutionDag, schedule: Schedule, *, lowering_name: str | None = None,
                            static_metrics: dict | None = None,
                            calibration_features: dict | None = None) -> ScheduleLoweringPlan:
    """Prefer the HA-6 mapping for existing hashes; add one exact family."""
    previous = classify_schedule(dag, schedule, lowering_name=lowering_name,
                                 static_metrics=static_metrics, calibration_features=calibration_features)
    if previous.mapping_status != "UNSUPPORTED":
        return previous
    launches, reasons = mixed_group_specs(dag, schedule)
    if reasons:
        payload = {key: value for key, value in asdict(previous).items() if key != "lowering_plan_hash"}
        # HA-6's generic template-missing labels are superseded by these
        # concrete semantic blockers for the mixed family.
        payload["unsupported_features"] = reasons
        payload["lowering_version"] = MIXED_VERSION
        return ScheduleLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))
    nodes = {node.node_id: node for node in dag.nodes}
    ops = [op for group in schedule.groups for op in group.operations]
    payload = {"schedule_hash": schedule.schedule_hash, "algorithm": dag.algorithm,
               "decomposition_id": dag.decomposition_id, "family_id": FAMILY_MIXED,
               "support_status": "LOWERABLE_WITH_TEMPLATE", "mapping_status": "EXACT_MAPPING",
               "launch_sequence": launches,
               "materialized_values": tuple(op.node_id for op in ops if op.location == "HBM" and
                                            nodes[op.node_id].operation != "OUTPUT"),
               "recomputed_values": tuple(sorted(schedule.recompute_allowed)),
               "fused_forms": tuple(op.node_id for op in ops if nodes[op.node_id].operation == "LINEAR_FORM" and
                                    op.location in ("ON_CHIP", "EPHEMERAL")),
               "reconstruction_strategy": "DIRECT_GEMM_EPILOGUE_PARTIAL_OR_FINAL",
               "expected_launch_count": len(launches), "unsupported_features": (),
               "lowering_version": MIXED_VERSION,
               "static_metrics_hash": stable_hash(static_metrics) if static_metrics is not None else None,
               "calibration_features_hash": stable_hash(calibration_features) if calibration_features is not None else None}
    return ScheduleLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))


def mixed_lowerability_query(dag: ExecutionDag, schedule: Schedule, **kwargs) -> dict:
    plan = classify_mixed_schedule(dag, schedule, **kwargs)
    return {"supported": plan.mapping_status != "UNSUPPORTED", "exact": plan.mapping_status == "EXACT_MAPPING",
            "family_id": plan.family_id, "schedule_hash": plan.schedule_hash,
            "lowering_plan_hash": plan.lowering_plan_hash, "reasons": list(plan.unsupported_features)}
