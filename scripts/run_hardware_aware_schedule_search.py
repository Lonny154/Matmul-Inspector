#!/usr/bin/env python3
"""Run a bounded CPU-only CP-SAT search over exact block execution schedules."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game_schedule_model import (
    BlockConfig, POLICIES, build_execution_dag, evaluate_schedule, make_schedule,
    reference_dag,
)
from matmul_inspector.tensor_game_schedule_search import (
    OBJECTIVES, SearchConfig, eligible_recompute_forms, fixed_operation_order,
    rebind_schedule, solve_schedule,
)
from run_hardware_aware_schedule_study import PROJECT_RANK8, project_actions


REPO = Path(__file__).resolve().parents[1]
FRONTIER_FIELDS = ("temporary_hbm_traffic_bytes", "recomputation_arithmetic_operations",
                   "peak_abstract_local_bytes", "logical_launch_groups")


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pareto_frontier(candidates: list[dict]) -> list[dict]:
    """Return one nondominated point per static metric vector, with tied schedules."""
    unique = {candidate["schedule_hash"]: candidate for candidate in candidates}
    vectors: dict[tuple[int, ...], list[dict]] = {}
    for candidate in unique.values():
        vector = tuple(candidate[field] for field in FRONTIER_FIELDS)
        vectors.setdefault(vector, []).append(candidate)
    ordered = [{**dict(zip(FRONTIER_FIELDS, vector)),
                "equivalent_schedules": sorted(({
                    "source": row["source"], "policy": row.get("policy"),
                    "run_id": row.get("run_id"), "schedule_hash": row["schedule_hash"]}
                    for row in rows), key=lambda row: (row["source"], row["schedule_hash"]))}
               for vector, rows in sorted(vectors.items())]
    def dominates(left: dict, right: dict) -> bool:
        return (all(left[field] <= right[field] for field in FRONTIER_FIELDS)
                and any(left[field] < right[field] for field in FRONTIER_FIELDS))
    return [candidate for candidate in ordered
            if not any(dominates(other, candidate) for other in ordered if other is not candidate)]


def _dag(algorithm: str, size: int, args) -> object:
    config = BlockConfig(size, args.input_dtype, args.accumulation_dtype, args.output_dtype)
    if algorithm == "project-rank8":
        return build_execution_dag(algorithm, project_actions(args.project_rank8_artifact), config)
    return reference_dag(algorithm, config)


def _budget(label: str, dag) -> int:
    if label == "unbounded":
        nodes = {node.node_id: node for node in dag.nodes}
        uses = {node_id: sum(node_id in node.dependencies for node in dag.nodes)
                for node_id in eligible_recompute_forms(dag)}
        return sum((uses[node_id] - 1) * nodes[node_id].arithmetic_operations for node_id in uses)
    return int(label) * dag.config.block_elements


def collect_search(args) -> dict:
    algorithms = ("conventional", "strassen", "project-rank8") if args.algorithm == "all" else (args.algorithm,)
    sizes = tuple(args.matrix_sizes or ((128, 256, 512, 1024) if args.full_study else (128,)))
    capacities = tuple(args.capacity_bytes if args.capacity_bytes is not None else
                       (args.capacity_block_units or ((1, 2, 3, 4, 6, 8) if args.full_study else (2, 4))))
    budget_labels = tuple(args.recompute_budget_ops if args.recompute_budget_ops is not None else
                          (args.recompute_budget_blocks or ("0", "1")))
    objectives = tuple(args.objectives or (("MIN_TEMP_HBM_TRAFFIC", "LEX_TEMP_RECOMPUTE_LAUNCH")
                                           if args.full_study else ("LEX_TEMP_RECOMPUTE_LAUNCH",)))
    normalized = args.capacity_bytes is None and args.recompute_budget_ops is None
    dags = {(algorithm, size): _dag(algorithm, size, args) for algorithm in algorithms for size in sizes}
    dag_manifest = [{"algorithm": algorithm, "matrix_size": size, "dag_hash": dag.dag_hash,
                     "decomposition_id": dag.decomposition_id, "block_elements": dag.config.block_elements,
                     "input_block_bytes": dag.config.block_bytes(dag.config.input_dtype),
                     "accumulation_block_bytes": dag.config.block_bytes(dag.config.accumulation_dtype),
                     "output_block_bytes": dag.config.block_bytes(dag.config.output_dtype),
                     "fixed_order": fixed_operation_order(dag),
                     "eligible_recompute_forms": eligible_recompute_forms(dag)}
                    for (algorithm, size), dag in dags.items()]
    manual = {(algorithm, size, policy): evaluate_schedule(dag, make_schedule(dag, policy))
              for (algorithm, size), dag in dags.items() for policy in POLICIES}
    runs: list[dict] = []
    schedule_rows: dict[tuple[str, int, str], dict] = {}
    metric_rows: list[dict] = []
    comparisons: list[dict] = []
    for algorithm in algorithms:
        solve_sizes = (sizes[0],) if normalized else sizes
        for solve_size in solve_sizes:
            source_dag = dags[(algorithm, solve_size)]
            target_sizes = sizes if normalized else (solve_size,)
            for capacity in capacities:
                capacity_bytes = (capacity * source_dag.config.block_bytes(source_dag.config.input_dtype)
                                  if args.capacity_bytes is None else capacity)
                for budget_label in budget_labels:
                    budget_ops = (_budget(str(budget_label), source_dag)
                                  if args.recompute_budget_ops is None else int(budget_label))
                    for objective in objectives:
                        run_id = f"R{len(runs):04d}"
                        search_config = SearchConfig(
                            on_chip_capacity_bytes=capacity_bytes, max_groups=args.max_groups,
                            max_matmuls_per_group=args.max_matmuls_per_group,
                            recomputation_budget_operations=budget_ops, objective=objective,
                            temporary_traffic_cap_bytes=args.traffic_cap_bytes,
                            seed=args.seed, workers=args.workers, timeout_seconds=args.timeout_seconds)
                        outcome = solve_schedule(source_dag, search_config)
                        run = {"run_id": run_id, "algorithm": algorithm, "dag_hash": source_dag.dag_hash,
                               "solved_matrix_size": solve_size, "evaluated_matrix_sizes": target_sizes,
                               "capacity_block_units": capacity if args.capacity_bytes is None else None,
                               "capacity_bytes": capacity_bytes,
                               "recomputation_budget_block_units": budget_label if args.recompute_budget_ops is None else None,
                               "recomputation_budget_operations": budget_ops,
                               "objective": objective, "stages": outcome.stages, "status": outcome.status,
                               "solver_summary": outcome.solver_summary,
                               "solver_predicted_metrics": outcome.predicted,
                               "evaluator_metrics": outcome.metrics,
                               "agreement": outcome.agreement, "diagnostics": outcome.diagnostics,
                               "schedule_hash": outcome.schedule.schedule_hash if outcome.schedule else None,
                               "validation_status": outcome.metrics["validation_status"] if outcome.metrics else "NOT_VALID"}
                        runs.append(run)
                        for size in target_sizes:
                            dag = dags[(algorithm, size)]
                            target_capacity = (capacity * dag.config.block_bytes(dag.config.input_dtype)
                                               if args.capacity_bytes is None else capacity)
                            target_budget = (_budget(str(budget_label), dag)
                                             if args.recompute_budget_ops is None else int(budget_label))
                            manual_controls = []
                            for policy in POLICIES:
                                row = manual[(algorithm, size, policy)]
                                feasible = (row["peak_abstract_local_bytes"] <= target_capacity
                                            and row["recomputation_arithmetic_operations"] <= target_budget
                                            and row["logical_launch_groups"] <= args.max_groups)
                                manual_controls.append({"policy": policy, "feasible_under_solver_limits": feasible,
                                                        "metrics": {field: row[field] for field in FRONTIER_FIELDS},
                                                        "schedule_hash": row["schedule_hash"]})
                            if outcome.schedule is None:
                                comparisons.append({"run_id": run_id, "algorithm": algorithm, "matrix_size": size,
                                                    "manual_controls": manual_controls, "solver_status": outcome.status})
                                continue
                            schedule = (outcome.schedule if size == solve_size else
                                        rebind_schedule(source_dag, dag, outcome.schedule))
                            metrics = evaluate_schedule(dag, schedule)
                            if metrics["peak_abstract_local_bytes"] > target_capacity or metrics["recomputation_arithmetic_operations"] > target_budget:
                                raise ValueError(f"normalized schedule exceeds target limits: {run_id}, size {size}")
                            key = (algorithm, size, schedule.schedule_hash)
                            if key not in schedule_rows and len(schedule_rows) < args.max_retained_schedules:
                                schedule_rows[key] = {"algorithm": algorithm, "matrix_size": size,
                                                      "schedule": asdict(schedule), "source_run_id": run_id}
                            if key in schedule_rows:
                                metric_rows.append({"run_id": run_id, "algorithm": algorithm, "matrix_size": size,
                                                    "capacity_bytes": target_capacity,
                                                    "recomputation_budget_operations": target_budget,
                                                    "schedule_hash": schedule.schedule_hash,
                                                    "normalized_reuse": size != solve_size,
                                                    "metrics": metrics})
                            best_manual = min(row["metrics"]["temporary_hbm_traffic_bytes"]
                                              for row in manual_controls if row["feasible_under_solver_limits"])
                            comparisons.append({"run_id": run_id, "algorithm": algorithm, "matrix_size": size,
                                                "capacity_bytes": target_capacity,
                                                "manual_controls": manual_controls,
                                                "best_feasible_manual_temp_hbm_traffic_bytes": best_manual,
                                                "solver_temp_hbm_traffic_bytes": metrics["temporary_hbm_traffic_bytes"],
                                                "solver_minus_manual_best_bytes": metrics["temporary_hbm_traffic_bytes"] - best_manual})
    frontiers = {}
    for algorithm in algorithms:
        size = sizes[0]
        candidates = []
        for policy in POLICIES:
            row = manual[(algorithm, size, policy)]
            candidates.append({"source": "manual", "policy": policy, "schedule_hash": row["schedule_hash"],
                               **{field: row[field] for field in FRONTIER_FIELDS}})
        for row in metric_rows:
            if row["algorithm"] == algorithm and row["matrix_size"] == size:
                metrics = row["metrics"]
                candidates.append({"source": "solver", "run_id": row["run_id"],
                                   "schedule_hash": row["schedule_hash"],
                                   **{field: metrics[field] for field in FRONTIER_FIELDS}})
        frontier = pareto_frontier(candidates)
        manual_candidates = [row for row in candidates if row["source"] == "manual"]
        solver_candidates = [row for row in candidates if row["source"] == "solver"]
        def dominates(left: dict, right: dict) -> bool:
            return (all(left[field] <= right[field] for field in FRONTIER_FIELDS)
                    and any(left[field] < right[field] for field in FRONTIER_FIELDS))
        frontiers[algorithm] = {"declared_dimensions": FRONTIER_FIELDS,
                                "candidate_count": len(candidates),
                                "unique_schedule_hashes": len({row["schedule_hash"] for row in candidates}),
                                "nondominated_manual_policies": sorted({row["policy"]
                                    for point in frontier for row in point["equivalent_schedules"]
                                    if row["source"] == "manual"}),
                                "solver_hashes_dominating_manual": sorted({row["schedule_hash"]
                                    for row in solver_candidates
                                    if any(dominates(row, manual_row) for manual_row in manual_candidates)}),
                                "frontier": frontier}
    aggregate = {"algorithm_count": len(algorithms), "matrix_sizes": sizes,
                 "normalized_reuse": normalized, "solver_run_count": len(runs),
                 "valid_agreement_pass_count": sum(row["agreement"] == "PASS" for row in runs),
                 "invalid_or_incomplete_count": sum(row["agreement"] != "PASS" for row in runs),
                 "unique_solver_schedule_hashes_at_anchor_size": {
                     algorithm: len({row["schedule_hash"] for row in metric_rows
                                     if row["algorithm"] == algorithm and row["matrix_size"] == sizes[0]})
                     for algorithm in algorithms},
                 "pareto_front_sizes": {algorithm: len(data["frontier"]) for algorithm, data in frontiers.items()},
                 "solver_beats_best_feasible_manual_count": sum(row.get("solver_minus_manual_best_bytes", 0) < 0
                                                                 for row in comparisons)}
    monotonicity = []
    for algorithm in algorithms:
        for budget_label in budget_labels:
            for objective in objectives:
                matching = sorted((row for row in runs
                                   if row["algorithm"] == algorithm
                                   and row["recomputation_budget_block_units"] == budget_label
                                   and row["objective"] == objective
                                   and row["solved_matrix_size"] == sizes[0]),
                                  key=lambda row: row["capacity_bytes"])
                if len(matching) < 2 or any(row["status"] != "OPTIMAL" for row in matching):
                    continue
                traffic = [row["evaluator_metrics"]["temporary_hbm_traffic_bytes"] for row in matching]
                monotonicity.append({"algorithm": algorithm, "budget": budget_label,
                                     "objective": objective, "traffic_nonincreasing":
                                     all(right <= left for left, right in zip(traffic, traffic[1:]))})
    aggregate["capacity_traffic_monotonicity_checks"] = monotonicity
    return {"algorithms": algorithms, "sizes": sizes, "capacities": capacities,
            "budget_labels": budget_labels, "objectives": objectives, "normalized": normalized,
            "dag_manifest": dag_manifest, "runs": runs, "schedules": list(schedule_rows.values()),
            "metrics": metric_rows, "frontiers": frontiers, "comparisons": comparisons,
            "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("all", "conventional", "strassen", "project-rank8"), default="all")
    parser.add_argument("--matrix-sizes", nargs="+", type=int)
    parser.add_argument("--input-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--accumulation-dtype", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--output-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--capacity-block-units", nargs="+", type=int)
    capacity.add_argument("--capacity-bytes", nargs="+", type=int)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--recompute-budget-blocks", nargs="+", choices=("0", "1", "2", "unbounded"))
    budget.add_argument("--recompute-budget-ops", nargs="+", type=int)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES)
    parser.add_argument("--traffic-cap-bytes", type=int)
    parser.add_argument("--max-groups", type=int, default=40)
    parser.add_argument("--max-matmuls-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-retained-schedules", type=int, default=512)
    parser.add_argument("--project-rank8-artifact", type=Path, default=PROJECT_RANK8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full-study", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    if args.max_retained_schedules < 1:
        parser.error("max-retained-schedules must be positive")
    if args.algorithm in ("all", "project-rank8") and not args.project_rank8_artifact.is_file():
        parser.error("frozen project-rank8 artifact is missing")
    study = collect_search(args)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    config = {"algorithms": study["algorithms"], "matrix_sizes": study["sizes"],
              "capacity_values": study["capacities"], "capacity_mode": "block_units" if args.capacity_bytes is None else "bytes",
              "recomputation_budgets": study["budget_labels"],
              "recomputation_budget_mode": "block_operations" if args.recompute_budget_ops is None else "operations",
              "objectives": study["objectives"], "input_dtype": args.input_dtype,
              "accumulation_dtype": args.accumulation_dtype, "output_dtype": args.output_dtype,
              "max_groups": args.max_groups, "max_matmuls_per_group": args.max_matmuls_per_group,
              "traffic_cap_bytes": args.traffic_cap_bytes, "seed": args.seed, "workers": args.workers,
              "timeout_seconds": args.timeout_seconds, "max_retained_schedules": args.max_retained_schedules,
              "normalized_reuse": study["normalized"]}
    provenance = {"source_commit": commit, "runtime_accelerator_used": False,
                  "project_rank8_artifact": str(args.project_rank8_artifact),
                  "project_rank8_artifact_sha256": hashlib.sha256(args.project_rank8_artifact.read_bytes()).hexdigest()
                  if args.algorithm in ("all", "project-rank8") else None}
    args.output.mkdir(parents=True)

    def write(name: str, value: object) -> None:
        (args.output / name).write_bytes(encoded(value))

    def write_lines(name: str, rows: list[dict]) -> None:
        (args.output / name).write_bytes(b"".join(encoded(row) for row in rows))

    write("config.json", config)
    write("provenance.json", provenance)
    write("dag_manifest.json", study["dag_manifest"])
    write("solver_model_summary.json", {"solver": "OR-Tools CP-SAT",
                                        "version": study["runs"][0]["solver_summary"].get("version") if study["runs"] else None,
                                        "fixed_order": "term-major topological",
                                        "capacity_liveness": "exact local birth-through-last-use, including EPHEMERAL",
                                        "run_count": len(study["runs"]),
                                        "variable_count_range": [min((row["solver_summary"].get("variables", 0) for row in study["runs"]), default=0),
                                                                 max((row["solver_summary"].get("variables", 0) for row in study["runs"]), default=0)],
                                        "constraint_count_range": [min((row["solver_summary"].get("constraints", 0) for row in study["runs"]), default=0),
                                                                   max((row["solver_summary"].get("constraints", 0) for row in study["runs"]), default=0)]})
    write_lines("solver_runs.jsonl", study["runs"])
    write_lines("schedules.jsonl", study["schedules"])
    write_lines("schedule_metrics.jsonl", study["metrics"])
    write("pareto_frontier.json", study["frontiers"])
    write("baseline_comparison.json", study["comparisons"])
    write("aggregate_summary.json", study["aggregate"])
    (args.output / "study_notes.md").write_text(
        "# HA-Schedule-2 CPU-only search\n\n"
        "All HBM and capacity numbers are abstract model counts, not GPU measurements. "
        "The independent HA-Schedule-1 evaluator gates retained schedules. "
        "See docs/hardware_aware_schedule_search.md for formulation, controls, results, and limits.\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.iterdir()) if path.is_file()}
    write("hash_manifest.json", hashes)
    print(f"{len(study['runs'])} solves, {study['aggregate']['valid_agreement_pass_count']} agreement passes, "
          f"{len(study['metrics'])} evaluated records -> {args.output}")
    return 0 if not study["aggregate"]["invalid_or_incomplete_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
