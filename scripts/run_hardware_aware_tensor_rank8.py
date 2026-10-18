#!/usr/bin/env python3
"""Retain a CPU-only exact rank-8 candidate study; never execute a GPU backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from matmul_inspector.hardware_aware_tensor_candidates import PARETO_DIMENSIONS, Rank8Config, run_study


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-accepted", type=int)
    parser.add_argument("--entry-bound", type=int, default=1)
    parser.add_argument("--maximum-coefficient", type=int, default=8)
    parser.add_argument("--maximum-total-support", type=int, default=96)
    parser.add_argument("--maximum-pre-additions", type=int, default=64)
    parser.add_argument("--maximum-post-additions", type=int, default=32)
    parser.add_argument("--representatives", type=int, default=16)
    parser.add_argument("--selection-mode", choices=("sample", "enumerate"), default="sample")
    parser.add_argument("--no-rank7-reference", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    config = Rank8Config(seed=args.seed,
                         max_attempts=args.max_attempts if args.max_attempts is not None else (32 if args.mode == "smoke" else 512),
                         max_accepted=args.max_accepted if args.max_accepted is not None else (32 if args.mode == "smoke" else 512),
                         entry_bound=args.entry_bound, maximum_coefficient=args.maximum_coefficient,
                         maximum_total_support=args.maximum_total_support,
                         maximum_pre_additions=args.maximum_pre_additions,
                         maximum_post_additions=args.maximum_post_additions,
                         representative_count=args.representatives, mode=args.selection_mode,
                         include_rank7_reference=not args.no_rank7_reference)
    result = run_study(config)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    provenance = {"schema": "hardware_aware_tensor_rank8.v1", "source_commit": revision,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "method": "seeded bounded GL(2,Z)^3 transformations of conventional rank-8",
                  "pareto_dimensions_minimized": list(PARETO_DIMENSIONS), "measured_gpu_performance": False}
    notes = ("# HA-Tensor-1 run notes\n\nAll candidates are exact representations of the known "
             "conventional rank-8 algorithm. No GPU work was run. Static descriptors and "
             "Pareto status do not prove latency. The known Strassen rank-7 reference is "
             "outside every rank-8 count.\n")
    files = {
        "config.json": encoded(result["config"]), "provenance.json": encoded(provenance),
        "baseline.json": encoded(result["baseline"]),
        "candidates.jsonl": b"".join(encoded(row) for row in result["candidates"]),
        "structural_signatures.json": encoded(result["structural_signatures"]),
        "pareto_frontier.json": encoded(result["pareto_frontier"]),
        "representatives.json": encoded(result["representatives"]),
        "aggregate_summary.json": encoded(result["aggregate_summary"]),
        "study_notes.md": notes.encode(),
    }
    if result["rank7_reference"] is not None:
        files["rank7_reference.json"] = encoded(result["rank7_reference"])
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    files["artifact_hashes.json"] = encoded({"algorithm": "sha256", "files": hashes})
    args.output.mkdir(parents=True)
    for name, content in files.items():
        (args.output / name).write_bytes(content)
    print(json.dumps({"output": str(args.output), "summary": result["aggregate_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
