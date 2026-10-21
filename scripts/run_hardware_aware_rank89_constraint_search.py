#!/usr/bin/env python3
"""Run one bounded CPU-only exact rank-8/9 finite-pool CP-SAT study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from dataclasses import asdict

from matmul_inspector.hardware_aware_rank89_search import (
    DEFAULT_CATALOG, DEFAULT_FROZEN, OBJECTIVES, SolveLimits, StructuredBounds,
    catalog_ids, control_pool, distinct_input_lower_bounds, frozen_existing_pool,
    pool_manifest, reuse_descriptors, solve_exact_decomposition, structured_pool,
    synthetic_rank9_control,
)
from matmul_inspector.hardware_aware_tensor_candidates import descriptor_vector
from matmul_inspector.tensor_game import standard_2x2_decomposition
from matmul_inspector.tensor_game_candidate_dataset import _record, canonical_actions


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, choices=(8, 9), default=8)
    parser.add_argument("--study-mode", choices=("control", "discovery"), default="control")
    parser.add_argument("--pool-mode", choices=("CONTROL", "FROZEN_EXISTING", "STRUCTURED_DOMAIN"), default="CONTROL")
    parser.add_argument("--solver-backend", choices=("cp-sat",), default="cp-sat")
    parser.add_argument("--objective", choices=OBJECTIVES, default="FEASIBILITY")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--max-pool-size", type=int, default=32)
    parser.add_argument("--max-variables", type=int, default=2048)
    parser.add_argument("--max-auxiliary-variables", type=int, default=512)
    parser.add_argument("--max-constraints", type=int, default=1024)
    parser.add_argument("--max-models", type=int, default=2)
    parser.add_argument("--max-accepted", type=int, default=2)
    parser.add_argument("--control-distractors", type=int, default=8)
    parser.add_argument("--frozen-artifact", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--existing-pools", default="residual_ranked,structural_diverse,algebraic_only")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--max-u-support", type=int, default=1)
    parser.add_argument("--max-v-support", type=int, default=1)
    parser.add_argument("--max-w-support", type=int, default=2)
    parser.add_argument("--max-abs-coefficient", type=int, default=1)
    parser.add_argument("--min-u-reuse", type=int, default=0)
    parser.add_argument("--min-v-reuse", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    if (args.study_mode == "discovery") == (args.pool_mode == "CONTROL"):
        parser.error("CONTROL is control-only; discovery must use a non-oracle finite pool")
    limits = SolveLimits(max_pool_actions=args.max_pool_size, max_variables=args.max_variables,
                         max_auxiliary_variables=args.max_auxiliary_variables,
                         max_constraints=args.max_constraints, timeout_seconds=args.timeout_seconds,
                         memory_limit_mb=args.memory_limit_mb, max_models=args.max_models,
                         max_accepted=args.max_accepted)
    bounds = StructuredBounds(args.max_u_support, args.max_v_support,
                              args.max_w_support, args.max_abs_coefficient)
    frozen_ids = tuple(value.strip() for value in args.existing_pools.split(",") if value.strip())
    if args.pool_mode == "CONTROL":
        actions, sources = control_pool(args.rank, args.max_pool_size, args.control_distractors)
    elif args.pool_mode == "FROZEN_EXISTING":
        actions, sources = frozen_existing_pool(args.frozen_artifact, frozen_ids, args.max_pool_size)
    else:
        actions, sources = structured_pool(bounds, args.max_pool_size)
    manifest = pool_manifest(actions, sources)
    catalog = catalog_ids(args.catalog)
    source = {"study_mode": args.study_mode, "pool_mode": args.pool_mode,
              "pool_hash": manifest["pool_hash"], "pool_artifact": str(args.frozen_artifact) if args.pool_mode == "FROZEN_EXISTING" else None,
              "existing_pool_identifiers": list(frozen_ids) if args.pool_mode == "FROZEN_EXISTING" else []}
    if args.pool_mode == "FROZEN_EXISTING":
        source["pool_artifact_sha256"] = hashlib.sha256(args.frozen_artifact.read_bytes()).hexdigest()
    result = solve_exact_decomposition(actions, args.rank, objective=args.objective,
                                       limits=limits, seed=args.seed, workers=args.workers,
                                       min_u_reuse=args.min_u_reuse, min_v_reuse=args.min_v_reuse,
                                       catalog_hashes=catalog, source=source)
    result["pool_hash"] = manifest["pool_hash"]
    result["pool_provenance"] = source
    baseline_actions = canonical_actions(standard_2x2_decomposition())
    baseline = _record("conventional-anchor", "conventional", baseline_actions, [])
    baseline_reuse = reuse_descriptors(baseline_actions)
    comparisons = []
    for row in result["solutions"]:
        comparisons.append({"canonical_decomposition_hash": row["canonical_decomposition_hash"],
                            "rank_delta": row["rank"] - 8,
                            "distinct_u_delta": row["reuse_descriptor_vector"]["distinct_u_forms"] - baseline_reuse["distinct_u_forms"],
                            "distinct_v_delta": row["reuse_descriptor_vector"]["distinct_v_forms"] - baseline_reuse["distinct_v_forms"],
                            "total_reuse_delta": row["reuse_descriptor_vector"]["total_reuse_count"] - baseline_reuse["total_reuse_count"],
                            "pre_addition_delta": row["descriptor_vector"]["pre_additions"] - baseline["static_cost"]["pre_additions"],
                            "post_addition_delta": row["descriptor_vector"]["post_additions"] - baseline["static_cost"]["post_additions"],
                            "support_delta": row["descriptor_vector"]["total_support"] - descriptor_vector(baseline)["total_support"],
                            "catalog_classification": row["project_catalog_classification"],
                            "known_rank8_split_parent_hash": row["known_rank8_split_parent_hash"]})
    synthetic_record = _record("synthetic-rank9", "synthetic-control", synthetic_rank9_control(), [])
    comparison = {"conventional_anchor": {"canonical_hash": baseline["canonical_id"],
                                            "rank": 8, "descriptor_vector": descriptor_vector(baseline),
                                            "reuse_descriptor_vector": baseline_reuse,
                                            "structural_signature_id": baseline["structural_signature_id"]},
                  "synthetic_rank9_control": {"canonical_hash": synthetic_record["canonical_id"],
                      "descriptor_vector": descriptor_vector(synthetic_record),
                      "reuse_descriptor_vector": reuse_descriptors(synthetic_rank9_control())},
                  "distinct_input_lower_bounds": distinct_input_lower_bounds(),
                  "candidate_comparisons": comparisons}
    summary = {"rank": args.rank, "study_mode": args.study_mode, "pool_mode": args.pool_mode,
               "pool_size": len(actions), "pool_hash": manifest["pool_hash"],
               "status": result["status"], "solver_status": result["solver_status"],
               "unique_exact_solutions": len(result["solutions"]),
               "catalog_classifications": {name: sum(row["project_catalog_classification"] == name
                                                    for row in result["solutions"])
                                           for name in sorted({row["project_catalog_classification"] for row in result["solutions"]})},
               "known_rank8_splits": sum(row["known_rank8_split_parent_hash"] is not None
                                         for row in result["solutions"]),
               "solve_wall_seconds": result["solve_wall_seconds"]}
    config = {"rank": args.rank, "study_mode": args.study_mode, "pool_mode": args.pool_mode,
              "objective": args.objective, "seed": args.seed, "workers": args.workers,
              "limits": asdict(limits), "structured_bounds": asdict(bounds),
              "existing_pool_identifiers": list(frozen_ids), "catalog": str(args.catalog),
              "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              "frozen_artifact": str(args.frozen_artifact), "control_distractors": args.control_distractors,
              "min_u_reuse": args.min_u_reuse, "min_v_reuse": args.min_v_reuse}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    provenance = {"schema": "hardware_aware_rank89_constraint_search.v1", "source_commit": revision,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "solver": result["solver"], "solver_version": result.get("solver_version"),
                  "seed": args.seed, "workers": args.workers,
                  "determinism": "single worker recommended; fixed seed; CP-SAT statuses preserved",
                  "measured_gpu_performance": False}
    notes = ("# HA-Tensor-2A run notes\n\nThis is an exact finite-pool CPU study. "
             "A feasible model is independently reconstructed. UNKNOWN or TIME_LIMIT is not "
             "infeasibility. NEW_TO_PROJECT_CATALOG is not mathematical novelty. "
             "Reuse descriptors do not prove physical cache or register reuse.\n")
    files = {"config.json": encoded(config), "provenance.json": encoded(provenance),
             "pool_manifest.json": encoded(manifest), "solver_model_summary.json": encoded({key: value for key, value in result.items() if key != "solutions"}),
             "solutions.jsonl": b"".join(encoded(row) for row in result["solutions"]),
             "baseline_comparison.json": encoded(comparison), "aggregate_summary.json": encoded(summary),
             "study_notes.md": notes.encode()}
    files["artifact_hashes.json"] = encoded({"algorithm": "sha256", "files": {
        key: hashlib.sha256(value).hexdigest() for key, value in sorted(files.items())}})
    args.output.mkdir(parents=True)
    for name, content in files.items():
        (args.output / name).write_bytes(content)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
