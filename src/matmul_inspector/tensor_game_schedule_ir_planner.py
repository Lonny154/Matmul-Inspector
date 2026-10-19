"""Deterministic primitive planning over validated schedule IR groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .tensor_game_schedule_calibration import stable_hash
from .tensor_game_schedule_hbm_lowering import HBMBackedPlan, classify_hbm_schedule
from .tensor_game_schedule_ir import ScheduleLoweringIR, translate_schedule, validate_ir
from .tensor_game_schedule_model import ExecutionDag, Schedule

PLANNER_VERSION = "ha-schedule-9-planner-v1"


@dataclass(frozen=True)
class IRLoweringPlan:
    source_schedule_hash: str
    ir_hash: str
    decomposition_id: str
    backend: str
    mapping_status: str
    primitive_ids: tuple[str, ...]
    launch_sequence: tuple[dict, ...]
    buffer_plan: tuple[dict, ...]
    recomputed_values: tuple[str, ...]
    output_contributions: tuple[dict, ...]
    expected_launch_count: int
    unsupported_reasons: tuple[str, ...]
    legacy_plan_hash: str | None
    version: str
    lowering_plan_hash: str


def primitive_registry() -> tuple[dict, ...]:
    common = {"dtypes": ["fp16/fp32/fp16"], "coefficients": "integer decomposition coefficients",
              "hbm_inputs": True, "recomputation": False, "expected_launches": 1}
    return (
        {**common, "primitive_id": "SINGLETON_FORM", "pattern": ["PREPARE_FORM"],
         "value_classes": ["PREPARED_FORM"], "max_fan_in": 4, "max_fan_out": 1,
         "partial_outputs": False, "hbm_inputs": False,
         "implementation": "_materialize_form_kernel"},
        {**common, "primitive_id": "SINGLETON_GEMM", "pattern": ["GEMM"],
         "value_classes": ["GEMM_PRODUCT"], "max_fan_in": 2, "max_fan_out": 1,
         "partial_outputs": False, "implementation": "_fp32_gemm_kernel"},
        {**common, "primitive_id": "SINGLETON_RECON", "pattern": ["RECONSTRUCT"],
         "value_classes": ["GEMM_PRODUCT", "PARTIAL_OUTPUT"], "max_fan_in": 2, "max_fan_out": 1,
         "partial_outputs": True, "implementation": "_accumulate_kernel"},
        {**common, "primitive_id": "SINGLETON_FINAL", "pattern": ["FINALIZE_OUTPUT"],
         "value_classes": ["PARTIAL_OUTPUT", "FINAL_OUTPUT"], "max_fan_in": 1, "max_fan_out": 1,
         "partial_outputs": True, "implementation": "_output_kernel"},
        {**common, "primitive_id": "ORDERED_ONE_GEMM", "pattern": ["<=8 ordered form/reconstruct/final ops", "one GEMM"],
         "value_classes": ["PREPARED_FORM", "GEMM_PRODUCT", "PARTIAL_OUTPUT", "FINAL_OUTPUT"],
         "max_fan_in": 2, "max_fan_out": 4, "max_operations": 8, "max_reconstructions": 4,
         "partial_outputs": True, "recomputation": True,
         "implementation": "_ordered_group_kernel"},
        {**common, "primitive_id": "ORDERED_RECON_FINAL", "pattern": ["RECONSTRUCT", "FINALIZE_OUTPUT"],
         "value_classes": ["GEMM_PRODUCT", "PARTIAL_OUTPUT", "FINAL_OUTPUT"],
         "max_fan_in": 2, "max_fan_out": 1, "partial_outputs": True,
         "implementation": "_ordered_group_kernel"},
    )


def match_group(ir: ScheduleLoweringIR, group_id: str) -> tuple[str | None, str | None]:
    """Match only retained bounded group shapes; never split a requested group."""
    group = next(group for group in ir.groups if group.group_id == group_id)
    by_id = {op.op_id: op for op in ir.operations}
    kinds = [by_id[op_id].kind for op_id in group.op_ids]
    if group.intended_strategy == "UNSUPPORTED":
        return None, "GROUP_STRATEGY_UNSUPPORTED"
    if len(kinds) == 1:
        return {"PREPARE_FORM": "SINGLETON_FORM",
                "GEMM": "SINGLETON_GEMM", "RECONSTRUCT": "SINGLETON_RECON",
                "FINALIZE_OUTPUT": "SINGLETON_FINAL"}.get(kinds[0]), (
                    "ORPHAN_RECOMPUTE_FORM" if kinds[0] == "RECOMPUTE_FORM" else None)
    if kinds == ["RECONSTRUCT", "FINALIZE_OUTPUT"]:
        return "ORDERED_RECON_FINAL", None
    if len(kinds) > 8:
        return None, "GROUP_LENGTH_GT8"
    if kinds.count("GEMM") != 1:
        return None, "GROUP_GEMM_COUNT_NOT_ONE"
    if kinds.count("RECONSTRUCT") > 4:
        return None, "GROUP_RECONSTRUCTION_FANOUT_GT4"
    if kinds.count("FINALIZE_OUTPUT") > 2:
        return None, "GROUP_FINALIZATION_FANOUT_GT2"
    if any(kind not in ("PREPARE_FORM", "RECOMPUTE_FORM", "GEMM", "RECONSTRUCT", "FINALIZE_OUTPUT")
           for kind in kinds):
        return None, "UNKNOWN_IR_OPERATION"
    return "ORDERED_ONE_GEMM", None


def _legacy_base(legacy):
    return legacy.schedule_plan if isinstance(legacy, HBMBackedPlan) else legacy


def plan_ir(dag: ExecutionDag, schedule: Schedule, ir: ScheduleLoweringIR, *,
            lowering_name: str | None = None, static_metrics: dict | None = None,
            calibration_features: dict | None = None,
            prefer_legacy: bool = False) -> IRLoweringPlan:
    """Plan IR groups, retaining old physical paths as an opt-in compatibility control."""
    validate_ir(dag, schedule, ir)
    names, reasons = [], []
    for group in ir.groups:
        primitive, reason = match_group(ir, group.group_id)
        if primitive is None:
            reasons.append(f"{group.group_id}:{reason}")
        names.append(primitive or "UNSUPPORTED")
    dtype = (dag.config.input_dtype, dag.config.accumulation_dtype, dag.config.output_dtype)
    if dtype != ("fp16", "fp32", "fp16"):
        reasons.append("DTYPE_PRIMITIVE_UNSUPPORTED")
    legacy = classify_hbm_schedule(dag, schedule, lowering_name=lowering_name,
        static_metrics=static_metrics, calibration_features=calibration_features)
    base = _legacy_base(legacy)
    if prefer_legacy and base.mapping_status != "UNSUPPORTED":
        backend = "LEGACY_KERNEL_REUSE"
        status = base.mapping_status
        sequence = base.launch_sequence
        reasons = list(base.unsupported_features) if status == "APPROXIMATE_MAPPING" else []
    elif reasons:
        backend = "IR_PRIMITIVES"
        status = "LOWERING_UNSUPPORTED"
        sequence = ()
    else:
        backend = "IR_PRIMITIVES"
        status = "EXACT_MAPPING"
        sequence = []
        for group, primitive in zip(ir.groups, names):
            operation = next(op for op in ir.operations if op.op_id == group.op_ids[0])
            kernel = next(item["implementation"] for item in primitive_registry()
                          if item["primitive_id"] == primitive)
            sequence.append({"group_id": group.group_id, "role": primitive,
                             "node_id": operation.source_node_id, "kernel": kernel,
                             "op_ids": group.op_ids})
        sequence = tuple(sequence)
    payload = {"source_schedule_hash": schedule.schedule_hash, "ir_hash": ir.ir_hash,
        "decomposition_id": dag.decomposition_id, "backend": backend,
        "mapping_status": status, "primitive_ids": tuple(names),
        "launch_sequence": tuple(sequence),
        "buffer_plan": tuple(asdict(buffer) for buffer in ir.buffer_plan),
        "recomputed_values": tuple(sorted({op.source_node_id for op in ir.operations
                                             if op.kind == "RECOMPUTE_FORM"})),
        "output_contributions": tuple({"output_block": item["output_block"],
                                       "accumulation_ids": item["contribution_nodes"]}
                                      for item in ir.output_transitions),
        "expected_launch_count": len(sequence), "unsupported_reasons": tuple(reasons),
        "legacy_plan_hash": legacy.lowering_plan_hash if base.mapping_status != "UNSUPPORTED" else None,
        "version": PLANNER_VERSION}
    return IRLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))


def compare_legacy_plan(legacy, plan: IRLoweringPlan) -> dict:
    """Explicitly compare old and IR-derived topology and schedule semantics."""
    base = _legacy_base(legacy)
    if base.schedule_hash != plan.source_schedule_hash:
        return {"status": "ERROR", "reason": "source schedule hash differs"}
    if base.mapping_status == "UNSUPPORTED":
        return {"status": "NOT_APPLICABLE", "reason": "no legacy physical plan"}
    same_topology = tuple(item["kernel"] for item in base.launch_sequence) == tuple(
        item["kernel"] for item in plan.launch_sequence)
    same_launch_count = base.expected_launch_count == plan.expected_launch_count
    same_hbm = tuple(base.materialized_values) == tuple(
        item["value_id"] for item in plan.buffer_plan if item["output_block"] is None)
    same_recompute = tuple(base.recomputed_values) == plan.recomputed_values
    if isinstance(legacy, HBMBackedPlan):
        old_contributions = tuple((item["output_block"], tuple(item["accumulation_ids"]))
                                  for item in legacy.contribution_order)
    elif base.mapping_status == "EXACT_MAPPING":
        by_block = {block: [] for block in range(4)}
        for launch in base.launch_sequence:
            if launch["role"] == "ACCUMULATE":
                node_id = launch["node_id"]
                by_block[int(node_id.split("_")[0][1:])].append(node_id)
            for target in launch.get("reconstruction_targets", ()):
                by_block[target["output_block"]].append(target["accumulation_id"])
        old_contributions = tuple((block, tuple(by_block[block])) for block in range(4))
    else:
        old_contributions = None
    same_contributions = old_contributions == tuple((item["output_block"],
        tuple(item["accumulation_ids"])) for item in plan.output_contributions)
    if base.mapping_status == "APPROXIMATE_MAPPING" and plan.mapping_status == "EXACT_MAPPING":
        status = "DIFFERENT"
    elif same_topology and same_hbm and same_recompute and same_contributions and same_launch_count:
        status = "EQUIVALENT"
    elif same_hbm and same_recompute and same_contributions and same_launch_count:
        status = "SEMANTICALLY_EQUIVALENT_DIFFERENT_PLAN"
    else:
        status = "DIFFERENT"
    return {"status": status, "same_kernel_order": same_topology,
            "same_launch_count": same_launch_count,
            "same_hbm_materializations": same_hbm, "same_recomputation": same_recompute,
            "same_output_contribution_order": same_contributions,
            "legacy_plan_hash": legacy.lowering_plan_hash,
            "ir_plan_hash": plan.lowering_plan_hash}


def ir_lowerability_query(dag: ExecutionDag, schedule: Schedule, **kwargs) -> dict:
    ir = translate_schedule(dag, schedule)
    plan = plan_ir(dag, schedule, ir, **kwargs)
    return {"schedule_hash": schedule.schedule_hash, "ir_hash": ir.ir_hash,
            "lowering_plan_hash": plan.lowering_plan_hash,
            "ir_status": "IR_VALID", "mapping_status": plan.mapping_status,
            "reasons": list(plan.unsupported_reasons)}
