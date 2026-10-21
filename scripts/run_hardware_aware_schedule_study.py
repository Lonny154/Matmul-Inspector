#!/usr/bin/env python3
"""Record a CPU-only exact execution-DAG and abstract schedule study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_schedule_model import (
    BlockConfig, POLICIES, build_execution_dag, evaluate_schedule, make_schedule,
    reference_dag, validate_schedule,
)


REPO = Path(__file__).resolve().parents[1]
PROJECT_RANK8 = REPO / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"
DELTA_FIELDS = (
    "hbm_read_bytes", "hbm_write_bytes", "temporary_hbm_traffic_bytes",
    "peak_temp_hbm_bytes", "peak_on_chip_live_bytes", "logical_launch_groups",
    "number_of_recomputed_intermediates", "linear_form_arithmetic_operations",
    "reconstruction_arithmetic_operations",
)


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def project_actions(path: Path) -> tuple[RankOneAction, ...]:
    record = json.loads(path.read_text().splitlines()[0])
    if record["exact_reconstruction_status"] != "PASS" or record["rank"] != 8:
        raise ValueError("project candidate is not a validated rank-8 decomposition")
    return tuple(RankOneAction(**action) for action in record["actions"])


def collect_study(algorithms: tuple[str, ...], sizes: tuple[int, ...],
                  input_dtype: str, accumulation_dtype: str, output_dtype: str,
                  policies: tuple[str, ...], project_path: Path | None = None) -> dict:
    if "MATERIALIZE_ALL" not in policies:
        raise ValueError("MATERIALIZE_ALL must be selected for controlled deltas")
    if any(policy not in POLICIES for policy in policies):
        raise ValueError("unknown schedule policy")
    dags = []
    schedules = []
    results = []
    failures = []
    for algorithm in algorithms:
        for size in sizes:
            config = BlockConfig(size, input_dtype, accumulation_dtype, output_dtype)
            if algorithm == "project-rank8":
                if project_path is None:
                    raise ValueError("project-rank8 needs a frozen artifact")
                dag = build_execution_dag(algorithm, project_actions(project_path), config)
            else:
                dag = reference_dag(algorithm, config)
            dags.append({"algorithm": algorithm, "matrix_size": size, "dag_hash": dag.dag_hash,
                         "decomposition_id": dag.decomposition_id, "config": asdict(config),
                         "actions": [asdict(action) for action in dag.actions],
                         "nodes": [asdict(node) for node in dag.nodes],
                         "terms": [asdict(term) for term in dag.terms], "output_ids": dag.output_ids})
            for policy in policies:
                schedule = make_schedule(dag, policy)
                schedules.append({"algorithm": algorithm, "matrix_size": size,
                                  "schedule_id": schedule.schedule_id, "schedule_hash": schedule.schedule_hash,
                                  "dag_hash": dag.dag_hash, "policy": policy,
                                  "recompute_allowed": schedule.recompute_allowed,
                                  "groups": [asdict(group) for group in schedule.groups]})
                validation = validate_schedule(dag, schedule)
                if validation.valid:
                    results.append(evaluate_schedule(dag, schedule))
                else:
                    failures.append({"algorithm": algorithm, "matrix_size": size, "policy": policy,
                                     "issues": [asdict(issue) for issue in validation.issues]})
    baseline = {(r["algorithm"], r["matrix_size"]): r for r in results if r["policy"] == "MATERIALIZE_ALL"}
    comparisons = []
    for result in results:
        anchor = baseline[(result["algorithm"], result["matrix_size"])]
        comparisons.append({"algorithm": result["algorithm"], "matrix_size": result["matrix_size"],
                            "policy": result["policy"], "schedule_hash": result["schedule_hash"],
                            "deltas_vs_materialize_all": {field: result[field] - anchor[field] for field in DELTA_FIELDS},
                            "recomputation_arithmetic_operations": result["recomputation_arithmetic_operations"],
                            "extra_source_read_bytes": result["extra_source_read_bytes"]})
    return {"dags": dags, "schedules": schedules, "results": results,
            "aggregate": {"validation_failures": failures, "normal_result_count": len(results),
                          "comparisons": comparisons}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", choices=("all", "conventional", "strassen"), default="all")
    parser.add_argument("--matrix-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--input-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--accumulation-dtype", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--output-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--include-project-rank8", action="store_true")
    parser.add_argument("--project-rank8-artifact", type=Path, default=PROJECT_RANK8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    algorithms = ("conventional", "strassen") if args.algorithm == "all" else (args.algorithm,)
    if args.include_project_rank8:
        algorithms += ("project-rank8",)
    project_path = args.project_rank8_artifact if args.include_project_rank8 else None
    if project_path is not None and not project_path.is_file():
        parser.error(f"missing project-rank8 artifact: {project_path}")
    config = {"algorithms": algorithms, "matrix_sizes": args.matrix_sizes,
              "input_dtype": args.input_dtype, "accumulation_dtype": args.accumulation_dtype,
              "output_dtype": args.output_dtype, "policies": args.policies,
              "project_rank8_artifact": str(project_path.relative_to(REPO)) if project_path and project_path.is_relative_to(REPO) else str(project_path) if project_path else None}
    study = collect_study(algorithms, tuple(args.matrix_sizes), args.input_dtype,
                          args.accumulation_dtype, args.output_dtype,
                          tuple(dict.fromkeys(args.policies)), project_path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    provenance = {"source_commit": commit, "model": "CPU-only exact block DAG with abstract locations",
                  "source_artifact_sha256": hashlib.sha256(project_path.read_bytes()).hexdigest() if project_path else None,
                  "runtime_accelerator_used": False}
    args.output.mkdir(parents=True)

    def write(name: str, value: object) -> None:
        (args.output / name).write_bytes(encoded(value))

    write("config.json", config)
    write("provenance.json", provenance)
    write("dag.json", study["dags"])
    write("schedules.json", study["schedules"])
    (args.output / "schedule_results.jsonl").write_bytes(b"".join(encoded(row) for row in study["results"]))
    write("aggregate_summary.json", study["aggregate"])
    notes = ("# HA-Schedule-1 static study\n\n"
             "All counts are analytic block-model counts. Locations and launch groups are abstract. "
             "No GPU code ran and these records do not predict latency. Deltas compare with the same "
             "algorithm and matrix size under MATERIALIZE_ALL. See docs/hardware_aware_schedule_model.md.\n")
    (args.output / "study_notes.md").write_text(notes)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.iterdir()) if path.is_file()}
    write("hash_manifest.json", hashes)
    print(f"wrote {len(study['results'])} valid records to {args.output}; validation failures: {len(study['aggregate']['validation_failures'])}")
    return 0 if not study["aggregate"]["validation_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
