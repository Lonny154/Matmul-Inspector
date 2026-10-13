"""Evaluate a TG-4B.1 checkpoint on its held-out replay partition."""

import argparse
from pathlib import Path

from matmul_inspector.tensor_game_learning import resolve_device
from matmul_inspector.tensor_game_offline_learning import (
    ExperienceDataset, OfflineTrainingConfig, evaluate_offline, load_offline_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    model, config, manifest = load_offline_checkpoint(args.checkpoint, args.device)
    metrics = evaluate_offline(
        model, ExperienceDataset(args.experience, "test"),
        OfflineTrainingConfig(batch_size=args.batch_size, device=args.device),
        resolve_device(args.device), config.shape,
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"value_semantics={manifest['value_head_semantics']}")
    print(f"frozen_baseline={manifest['baseline']['test']}")
    print(f"candidate={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
