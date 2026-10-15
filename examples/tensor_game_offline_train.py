"""Train a TG-4B.1 candidate from a validated TG-4A replay buffer."""

import argparse
from pathlib import Path

from matmul_inspector.tensor_game_offline_learning import OfflineTrainingConfig, train_offline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("policy_only", "value_only", "joint"), default="joint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--policy-coefficient", type=float, default=1.0)
    parser.add_argument("--value-coefficient", type=float, default=1.0)
    parser.add_argument("--strict-torso-freeze", action="store_true")
    args = parser.parse_args()
    result = train_offline(
        args.experience,
        args.source_checkpoint,
        args.output,
        OfflineTrainingConfig(
            mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, policy_coefficient=args.policy_coefficient,
            value_coefficient=args.value_coefficient, seed=args.seed, device=args.device,
            strict_torso_freeze=args.strict_torso_freeze,
        ),
    )
    print(f"experiment={args.output}")
    print(f"samples={result['effective_samples']}")
    print(f"validation={result['final']['validation']}")
    print(f"test={result['final']['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
