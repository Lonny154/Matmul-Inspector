"""Run the preregistered TG-3E value-scale and trajectory study."""

import argparse
import hashlib
from pathlib import Path

from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_value_scale import ValueScaleConfig, run_value_scale_study, save_value_scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tensor-game/tg3b-mlp-256-seed42"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tensor-game/tg3e-value-scale-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    model, model_config, manifest = load_checkpoint(args.checkpoint, args.device)
    config = ValueScaleConfig(simulations=5 if args.smoke else 200, traced_simulations=5 if args.smoke else 20)
    checkpoint = args.checkpoint / manifest["checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != "857cbf4aa5eec24033bb84180782e2675febec1edb8d4427d1fa9833db4cbbe5":
        raise ValueError("unexpected checkpoint digest")
    def progress(row):
        print(f"moves={row['max_moves']} seed={row['seed']} {row['configuration']}: {row['termination_reason']}")
    records, trajectory, diagnostics = run_value_scale_study(model, model_config, config, args.device, progress)
    save_value_scale(args.output, config, records, trajectory, diagnostics, {
        "checkpoint_sha256": digest, "checkpoint_dataset_id": manifest["dataset_id"]
    })
    print(f"Saved {len(records)} trials to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
