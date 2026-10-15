"""Plan or run TG-4B.2A diverse MCTS experience collection."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from matmul_inspector.tensor_game_diverse_experience import (
    DiverseCollectionConfig,
    DiverseTargetConfig,
    collect_diverse_experience,
    generate_diverse_targets,
    planned_episode_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--smoke", action="store_true", help="execute eight bounded episodes")
    stage.add_argument("--pilot", action="store_true", help="execute the balanced 40-episode pilot")
    stage.add_argument("--execute-full", action="store_true", help="explicitly execute the full schedule")
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11))
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=12)
    parser.add_argument("--wall-time-limit", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    targets = DiverseTargetConfig(target_count=args.target_count)
    collection = DiverseCollectionConfig(
        seeds=tuple(args.seeds), simulations=args.simulations,
        proposals=args.proposals, wall_time_limit=args.wall_time_limit,
    )
    curriculum = generate_diverse_targets(targets)
    selected_stage = "full" if args.execute_full else "pilot" if args.pilot else "smoke"
    schedule = planned_episode_schedule(curriculum, collection, selected_stage)
    output = args.output or Path("artifacts/tensor-game") / (
        f"tg4b2a-{selected_stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    print(f"curriculum={curriculum['curriculum_id']}")
    print(f"targets={args.target_count}; stage={selected_stage}; estimated_episodes={len(schedule)}")
    print(f"proposed_output={output}")
    if not (args.smoke or args.pilot or args.execute_full):
        print("dry_run=true; pass --smoke, --pilot, or --execute-full to collect experience")
        return 0
    audit = collect_diverse_experience(curriculum, collection, selected_stage, output)
    print(f"replay_valid={audit['replay_validation']['valid']}")
    print(f"outcomes={audit['outcomes']['counts']}")
    print(f"states={audit['state_diversity']}")
    print(f"cost={audit['cost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
