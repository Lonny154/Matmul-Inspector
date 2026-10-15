"""Run the preregistered TG-3D neural value-guidance study."""

import argparse
import hashlib
import json
from pathlib import Path

from matmul_inspector.tensor_game_dataset import load_dataset
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_neural_mcts import NeuralValueEvaluator
from matmul_inspector.tensor_game_value_study import (
    UNTRAINED_SEED,
    ValueStudyConfig,
    create_untrained_model,
    run_value_study,
    save_value_study,
    synthetic_value_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/tensor-game/tg3b-dataset-2000"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tensor-game/tg3b-mlp-256-seed42"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tensor-game/tg3d-value-study-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    trained, model_config, manifest = load_checkpoint(args.checkpoint, args.device)
    untrained = create_untrained_model(model_config, UNTRAINED_SEED).to(args.device)
    config = ValueStudyConfig(
        seeds=(3, 11, 19, 27, 35, 43), simulations=5 if args.smoke else 200,
        trace_limit_per_trial=8 if args.smoke else 64,
    )
    checkpoint_file = args.checkpoint / manifest["checkpoint"]
    checkpoint_sha = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    if checkpoint_sha != "857cbf4aa5eec24033bb84180782e2675febec1edb8d4427d1fa9833db4cbbe5":
        raise ValueError("unexpected TG-3B checkpoint digest")

    def progress(row):
        print(f"moves={row['max_moves']} seed={row['seed']} {row['value_configuration']}: {row['termination_reason']}")

    records, diagnostics = run_value_study(trained, untrained, model_config, config, args.device, progress)
    synthetic = synthetic_value_diagnostic(dataset, {
        "trained_neural": NeuralValueEvaluator(trained, model_config, args.device),
        "untrained_neural": NeuralValueEvaluator(untrained, model_config, args.device),
    })
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text())
    save_value_study(args.output, config, records, diagnostics, synthetic, {
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_dataset_id": manifest["dataset_id"],
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_files": dataset_manifest["files"],
    })
    print(f"Saved {len(records)} trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
