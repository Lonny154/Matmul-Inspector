"""Train the TG-3B policy/value MLP on a TG-3A dataset."""

import argparse
from pathlib import Path

from matmul_inspector.tensor_game_dataset import SyntheticDatasetConfig, generate_dataset, load_dataset
from matmul_inspector.tensor_game_learning import PolicyValueConfig, TrainingConfig, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="persisted TG-3A dataset directory")
    source.add_argument("--smoke", action="store_true", help="generate a tiny in-memory CPU smoke dataset")
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--value-coefficient", type=float, default=1.0)
    args = parser.parse_args()
    dataset = (
        load_dataset(args.dataset)
        if args.dataset
        else generate_dataset(
            SyntheticDatasetConfig(
                shape=(4, 4, 4), construction_lengths=(1, 2, 3, 4),
                decompositions_per_length=3, seed=args.seed,
            )
        )
    )
    model_config = PolicyValueConfig(shape=tuple(dataset.manifest["config"]["shape"]), hidden_dim=args.hidden_dim)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_coefficient=args.value_coefficient,
        seed=args.seed,
        device=args.device,
    )
    result = train_model(dataset, args.output, model_config, training_config)
    print(f"experiment={args.output}")
    print(f"dataset={result['dataset_id']}")
    print(f"device={result['environment']['device']}")
    print(f"best_epoch={result['best_epoch']}")
    print(f"best_validation_loss={result['best_validation_loss']:.6f}")
    print(f"test_metrics={result['test_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
