#!/usr/bin/env python3
"""Plan or explicitly run controlled confirmation for hardware-search winners."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.controlled_benchmark import ControlledBenchmarkConfig
from matmul_inspector.hardware_search import (
    confirm_search_winners_many, read_hardware_search_manifest, search_winner_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_manifests", type=Path, nargs="+")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly enable randomized interleaved GPU confirmation")
    parser.add_argument("--output", type=Path,
                        help="new output directory; required with --execute")
    parser.add_argument("--scheduling-seed", type=int, default=8675309)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--samples-per-round", type=int, default=3)
    parser.add_argument("--launches-per-sample", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=100)
    args = parser.parse_args()

    manifests = [read_hardware_search_manifest(path) for path in args.search_manifests]
    winners, selection = search_winner_selection(manifests)
    print(f"Unique search winners selected for confirmation: {len(winners)}")
    for candidate, evidence in zip(winners, selection):
        print(f"  {candidate.candidate_id}: {candidate.parameters.to_dict()}; "
              f"discoveries={len(evidence['discoveries'])}")
    if not args.execute:
        print("DRY RUN: controlled confirmation was not executed.")
        return 0
    if args.output is None:
        raise SystemExit("--execute requires --output with a new directory")
    from matmul_inspector.controlled_benchmark import CudaControlledBenchmarkBackend

    config = ControlledBenchmarkConfig(
        scheduling_seed=args.scheduling_seed, rounds=args.rounds,
        samples_per_round=args.samples_per_round,
        launches_per_sample=args.launches_per_sample, warmups=args.warmups,
    )
    output = confirm_search_winners_many(
        manifests, CudaControlledBenchmarkBackend(), args.output, config)
    print(f"Controlled confirmation written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
