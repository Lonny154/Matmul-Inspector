"""Launch or resume the approval-gated TG-4B.2C held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matmul_inspector.tensor_game_heldout_runner import run_heldout_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_heldout_evaluation(
        plan_path=args.plan, addendum_path=args.addendum, approval_path=args.approval,
        experience_path=args.experience, output_path=args.output, resume=args.resume,
        progress_every=args.progress_every, device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
