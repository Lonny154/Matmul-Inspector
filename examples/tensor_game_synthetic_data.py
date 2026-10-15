"""Generate, persist, reload, and inspect a small TG-3A synthetic dataset."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from matmul_inspector.tensor_game_dataset import (
    SyntheticDatasetConfig,
    generate_dataset,
    load_dataset,
    save_dataset,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new dataset directory")
    parser.add_argument("--larger", action="store_true", help="generate 100 decompositions per length")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("artifacts/tensor-game") / f"tg3a-synthetic-{timestamp}"
    config = SyntheticDatasetConfig(
        shape=(4, 4, 4),
        construction_lengths=(1, 2, 3, 4),
        decompositions_per_length=100 if args.larger else 3,
        zero_probability=0.5,
        seed=37,
    )
    dataset = generate_dataset(config)
    validate_dataset(dataset)
    save_dataset(dataset, output)
    reloaded = load_dataset(output)
    if reloaded != dataset:
        raise RuntimeError("reloaded dataset differs from generated dataset")

    print(f"dataset={dataset.manifest['dataset_id']}")
    print(f"output={output}")
    print(f"decompositions={dataset.manifest['decomposition_count']}")
    print(f"residual_examples={dataset.manifest['sample_count']}")
    print(f"split_counts={dataset.manifest['split_decomposition_counts']}")
    print(f"generation_statistics={dataset.manifest['statistics']}")
    example = dataset.samples[0]
    print(
        f"example sample={example['sample_id']}, split={example['split']}, "
        f"remaining_upper_bound={example['known_completion_length']}"
    )
    print(f"example residual={example['residual']}")
    print(f"valid next actions={example['next_actions']}")
    print(f"model residual shape={config.shape}")
    print(
        f"padded action-target shape=({max(config.construction_lengths)}, "
        f"{sum(config.shape)}) with a target-validity mask"
    )
    print("All identities and the serialization round trip were verified exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
