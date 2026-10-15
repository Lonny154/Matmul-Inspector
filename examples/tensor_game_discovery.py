"""Run and persist the matched TG-2B independent-search curriculum."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from matmul_inspector.tensor_game_experiments import (
    ExperimentConfig,
    run_trials,
    save_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new evidence directory")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="use an explicit larger CPU budget instead of the modest default",
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("artifacts/tensor-game") / f"tg2b-{timestamp}"
    config = ExperimentConfig(
        seeds=(3, 11, 29) if args.extended else (3, 11),
        simulations=2000 if args.extended else 200,
        max_actions=24 if args.extended else 12,
        node_limit=2000 if args.extended else 200,
        max_wall_time_seconds=20.0 if args.extended else 2.0,
    )
    def progress(record):
        print(
            f"completed {record['target_id']} {record['strategy']} seed={record['seed']}: "
            f"{record['termination_reason']} ({record['simulations_completed']} simulations)"
        )

    records = run_trials(config, progress=progress)
    save_experiment(output, config, records)
    print(f"Saved {len(records)} matched trials to {output}")
    for target in dict.fromkeys(record["target_id"] for record in records):
        for strategy in ("uniform_sampled", "residual_informed", "hybrid"):
            group = [r for r in records if r["target_id"] == target and r["strategy"] == strategy]
            successes = sum(record["success"] for record in group)
            print(f"  {target}: {strategy}: {successes}/{len(group)} exact successes")
    print("Guided reference policies were not used. Unsuccessful trials are budget-limited outcomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
