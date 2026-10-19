"""Bounded, CPU-only schedule-to-template mapping for retained HA schedules.

This is deliberately not a DAG compiler. Only the fully materialized,
single-operation launch family has an exact executable template today.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter

from .tensor_game_schedule_calibration import stable_hash
from .tensor_game_schedule_calibration import pair_direction
from .tensor_game_schedule_model import ExecutionDag, Schedule, validate_schedule

VERSION = "ha-schedule-6-v1"
FAMILY_MATERIALIZED = "MATERIALIZED_DAG"
FAMILY_LEGACY = "LEGACY_PRODUCT_RECONSTRUCTION"
SUPPORT = ("DIRECTLY_LOWERABLE", "LOWERABLE_WITH_TEMPLATE", "APPROXIMATE_ONLY", "UNSUPPORTED")
MAPPING = ("EXACT_MAPPING", "APPROXIMATE_MAPPING", "UNSUPPORTED")
LEGACY_SCHEDULE_HASHES = {
    "conventional_reference": {
        128: "496d50911dfa5b5a498f1eedfd8fff15c1278d6928675a38281c9da61fcba32e",
        256: "5ebc4315b42fe2f86953a1b0d06a038a335a61dcf1006c492db524118a121c2f",
        512: "fcd613e1b61e0c68fd4b5a44cbbe3ce4c64d84f6f59dc3f9ae40f62ed27f2eaf",
        1024: "0dfaa6df948292e9d000ee65165a0b4e267955db99cd1941fcae6ae4f956b0f5"},
    "strassen_reference": {
        128: "0bb49cb28bf3dca8af4430332f6ffb0e29625d5ebbd0037efd084c93d5bbeada",
        256: "d301348607c02ecad09eda5e78bee578cad506dc39c65d4386b948d9f4af5f70",
        512: "4494d457ce204e22fe73742e8aca13ee1b746517abe4815308462224fdbd2ac0",
        1024: "2aa01b2a211c28c89b8ba77ba7e77ec415e88529af09b64953c07af7f0e4b81d"},
    "rank8_stored": {
        128: "fb511c7879508b00cd11edbe7f30cffb9105bde817605acc6347cfd9d0268a80",
        256: "3174902e7fabc72d57e6bea2101c5339a5725044d39a889bdc4802fc94344d66",
        512: "aaa035fdffa7b627b9524df9bf3fbd4532df871ffc059fbdc188b2558a697cb2",
        1024: "7cf2f0ee2bd6f70b7088ce6001fc0448f1bdc5730b2f6ed8554f23e9de50e54f"},
    "rank8_recomputed": {
        128: "ed149344b54774282370407a329ed853b79935fa24dcd73c87f544a27899691e",
        256: "61e6aef23f76cbaf4d42b662b87a1e10adf2290dde00fe38600f075d6f856dd1",
        512: "e09975154fd3e744b4f0cd6b50e16ba2f40e7d60bfd3e54a142849f3fc4bdec7",
        1024: "d0c9eef7bedae54e49fad9185dbf9e778fc391ce5de6c50e6662951be9b4a3d8"},
}


@dataclass(frozen=True)
class ScheduleLoweringPlan:
    schedule_hash: str
    algorithm: str
    decomposition_id: str
    family_id: str
    support_status: str
    mapping_status: str
    launch_sequence: tuple[dict, ...]
    materialized_values: tuple[str, ...]
    recomputed_values: tuple[str, ...]
    fused_forms: tuple[str, ...]
    reconstruction_strategy: str
    expected_launch_count: int
    unsupported_features: tuple[str, ...]
    lowering_version: str
    static_metrics_hash: str | None
    calibration_features_hash: str | None
    lowering_plan_hash: str


def capability_manifest() -> dict:
    """Static registry; safe for an agent-style query without CUDA imports."""
    return {"version": VERSION, "families": [
        {"family_id": FAMILY_MATERIALIZED, "supported_algorithms": ["conventional", "strassen", "project-rank8"],
         "supported_features": ["singleton HBM form/product/accumulation/output groups"],
         "forbidden_features": ["ON_CHIP", "EPHEMERAL", "multi-operation groups", "recomputation"],
         "mapping_capability": "EXACT_MAPPING",
         "implementation": "matmul_inspector.tensor_game_schedule_lowering_runtime.execute_materialized",
         "tested_sizes": [128, 256, 512, 1024], "tested_dtypes": ["fp16/fp32/fp16"]},
        {"family_id": FAMILY_LEGACY, "supported_algorithms": ["conventional", "strassen", "project-rank8"],
         "supported_features": ["known HA-Schedule-3 selected hashes only"],
         "forbidden_features": ["claim of exact product or reconstruction placement"],
         "mapping_capability": "APPROXIMATE_MAPPING",
         "implementation": "matmul_inspector.tensor_game_schedule_gpu_runtime.execute_selected",
         "tested_sizes": [128, 256, 512, 1024], "tested_dtypes": ["fp16/fp32/fp16"]},
    ]}


def classify_schedule(dag: ExecutionDag, schedule: Schedule, *, lowering_name: str | None = None,
                      static_metrics: dict | None = None, calibration_features: dict | None = None) -> ScheduleLoweringPlan:
    """Classify actual placement/group semantics, never a metric-vector match."""
    check = validate_schedule(dag, schedule)
    if not check.valid:
        raise ValueError(f"illegal schedule: {check.issues}")
    if lowering_name and LEGACY_SCHEDULE_HASHES.get(lowering_name, {}).get(dag.config.matrix_size) != schedule.schedule_hash:
        raise ValueError("legacy GPU mapping does not match this exact retained schedule hash")
    nodes = {node.node_id: node for node in dag.nodes}
    ops = [op for group in schedule.groups for op in group.operations]
    materialized = tuple(op.node_id for op in ops if op.location == "HBM" and nodes[op.node_id].operation != "OUTPUT")
    fused = tuple(op.node_id for op in ops if nodes[op.node_id].operation == "LINEAR_FORM" and op.location == "ON_CHIP")
    recomputed = tuple(sorted(schedule.recompute_allowed))
    dtype_supported = (dag.config.input_dtype, dag.config.accumulation_dtype, dag.config.output_dtype) == (
        "fp16", "fp32", "fp16")
    all_materialized = dtype_supported and not recomputed and all(
        len(group.operations) == 1 and group.operations[0].location == "HBM" for group in schedule.groups)
    reasons: list[str] = []
    if not dtype_supported:
        reasons.append("DTYPE_TEMPLATE_MISSING")
    if any(sum(nodes[op.node_id].operation == "MATMUL" for op in group.operations) > 1 for group in schedule.groups):
        reasons.append("MULTI_GEMM_GROUP_UNSUPPORTED")
    if any(len(group.operations) > 1 for group in schedule.groups):
        reasons.append("MULTI_OPERATION_GROUP_TEMPLATE_MISSING")
    if any({nodes[op.node_id].operation for op in group.operations} >= {"MATMUL", "ACCUMULATE"}
           for group in schedule.groups):
        reasons.append("FUSED_GEMM_RECONSTRUCTION_TEMPLATE_MISSING")
    if any(nodes[op.node_id].operation == "LINEAR_FORM" and op.location == "HBM"
           for group in schedule.groups if len(group.operations) > 1 for op in group.operations):
        reasons.append("COLOCATED_FORM_MATERIALIZATION_UNSUPPORTED")
    if any(op.location == "ON_CHIP" for op in ops):
        reasons.append("ONCHIP_FUSION_TEMPLATE_MISSING")
    if recomputed:
        reasons.append("RECOMPUTATION_TEMPLATE_MISSING")
    if not all_materialized and not reasons:
        reasons.append("LOWERING_TEMPLATE_MISSING")
    if all_materialized:
        family, support, mapping, reasons = FAMILY_MATERIALIZED, "LOWERABLE_WITH_TEMPLATE", "EXACT_MAPPING", []
        launch_sequence = tuple({"group_id": group.group_id, "role": nodes[group.operations[0].node_id].operation,
                                 "node_id": group.operations[0].node_id,
                                 "kernel": {"LINEAR_FORM": "_materialize_form_kernel", "MATMUL": "_fp32_gemm_kernel",
                                            "ACCUMULATE": "_accumulate_kernel", "OUTPUT": "_output_kernel"}[nodes[group.operations[0].node_id].operation]}
                                for group in schedule.groups)
        reconstruction = "SEPARATE_ACCUMULATE_AND_OUTPUT"
    elif lowering_name:
        family, support, mapping = FAMILY_LEGACY, "APPROXIMATE_ONLY", "APPROXIMATE_MAPPING"
        # Existing HA-3 mapping is intentionally not reclassified as exact.
        reasons = ["PRODUCT_AND_RECONSTRUCTION_PLACEMENT_DIFFERS", *reasons]
        kernels = (["_materialize_u_form"] if lowering_name == "rank8_stored" else []) + [
            "_shared_form_gemm" if lowering_name.startswith("rank8_") and index < 2 else "_fused_input_gemm_kernel"
            for index in range(len(dag.terms))] + ["_fused_reconstruction_kernel"]
        launch_sequence = tuple({"group_id": None, "role": "LEGACY", "node_id": None, "kernel": kernel}
                                for kernel in kernels)
        reconstruction = "MATERIALIZED_PRODUCTS_FUSED_RECONSTRUCTION"
    else:
        family, support, mapping = "UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED"
        launch_sequence = ()
        reconstruction = "UNSUPPORTED"
    payload = {"schedule_hash": schedule.schedule_hash, "algorithm": dag.algorithm,
               "decomposition_id": dag.decomposition_id, "family_id": family,
               "support_status": support, "mapping_status": mapping,
               "launch_sequence": launch_sequence, "materialized_values": materialized,
               "recomputed_values": recomputed, "fused_forms": fused,
               "reconstruction_strategy": reconstruction, "expected_launch_count": len(launch_sequence),
               "unsupported_features": tuple(dict.fromkeys(reasons)), "lowering_version": VERSION,
               "static_metrics_hash": stable_hash(static_metrics) if static_metrics is not None else None,
               "calibration_features_hash": stable_hash(calibration_features) if calibration_features is not None else None}
    return ScheduleLoweringPlan(**payload, lowering_plan_hash=stable_hash(payload))


def lowerability_query(dag: ExecutionDag, schedule: Schedule, **kwargs) -> dict:
    """Return a deterministic pre-GPU decision and its provenance."""
    plan = classify_schedule(dag, schedule, **kwargs)
    return {"supported": plan.mapping_status != "UNSUPPORTED", "exact": plan.mapping_status == "EXACT_MAPPING",
            "family_id": plan.family_id, "schedule_hash": plan.schedule_hash,
            "lowering_plan_hash": plan.lowering_plan_hash, "reasons": list(plan.unsupported_features)}


def benchmarkable(plan: ScheduleLoweringPlan, correctness: list[dict]) -> bool:
    """Every required seeded check must pass before timing this plan."""
    checks = [row for row in correctness if row.get("lowering_plan_hash") == plan.lowering_plan_hash]
    return plan.mapping_status != "UNSUPPORTED" and len({row.get("seed") for row in checks}) >= 3 and all(
        row.get("passed") is True for row in checks)


def coverage_summary(plans: list[ScheduleLoweringPlan]) -> dict:
    by_algorithm: dict[str, dict] = {}
    for algorithm in sorted({plan.algorithm for plan in plans}):
        subset = [plan for plan in plans if plan.algorithm == algorithm]
        counts = Counter(plan.mapping_status for plan in subset)
        by_algorithm[algorithm] = {"legal": len(subset), **{status: counts[status] for status in MAPPING},
                                   "families": sorted({plan.family_id for plan in subset if plan.mapping_status != "UNSUPPORTED"})}
    totals = Counter(plan.mapping_status for plan in plans)
    return {"legal": len(plans), "by_algorithm": by_algorithm,
            **{status: totals[status] for status in MAPPING}}


def plan_record(plan: ScheduleLoweringPlan) -> dict:
    return asdict(plan)


def aggregate_graph_sessions(sessions: list[dict]) -> list[dict]:
    """Keep each independent session visible while summarizing its direction."""
    result = []
    for size in sorted({row["matrix_size"] for row in sessions}):
        rows = [row for row in sessions if row["matrix_size"] == size]
        if len({row["session_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate graph session ID")
        directions = [pair_direction(row) for row in rows]
        directional = set(directions) - {"PARITY"}
        if len(rows) < 2:
            confidence = "UNSTABLE"
        elif len(directional) > 1:
            confidence = "DISAGREES"
        elif not directional:
            confidence = "PARITY"
        elif "PARITY" in directions:
            confidence = "PARTIAL_AGREEMENT"
        else:
            confidence = "AGREES"
        intervals = [row["ratio_bootstrap_95pct"] for row in rows]
        result.append({"matrix_size": size, "session_count": len(rows), "confidence": confidence,
                       "directions": Counter(directions), "interval_overlap": max(lo for lo, _ in intervals) <= min(
                           hi for _, hi in intervals),
                       "observed_ratio_range": [min(row["median_stored_over_recomputed_ratio"] for row in rows),
                                                max(row["median_stored_over_recomputed_ratio"] for row in rows)],
                       "sessions": rows})
    return result
