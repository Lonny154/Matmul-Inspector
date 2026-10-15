"""Validate the TG-4B.2C plan and protocol addendum without opening test targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matmul_inspector.tensor_game_protocol_addendum import validate_protocol_addendum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_protocol_addendum(
        args.plan, args.addendum, output_path=args.output, approval_path=args.approval,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
