"""Evaluate a TG-3B checkpoint and compare bounded proposal policies."""

import argparse
import json
from pathlib import Path

from matmul_inspector.tensor_game_dataset import load_dataset
from matmul_inspector.tensor_game_learning import compare_proposals, evaluate_model, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evaluation JSON path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--proposal-count", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = load_dataset(args.dataset)
    model, config, manifest = load_checkpoint(args.experiment, args.device)
    if dataset.manifest["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("evaluation dataset does not match the checkpoint dataset identifier")
    device = next(model.parameters()).device
    result = {
        "dataset_id": dataset.manifest["dataset_id"],
        "experiment_version": manifest["experiment_version"],
        "test_metrics": evaluate_model(model, dataset, "test", config, args.batch_size, device),
        "proposal_comparison": compare_proposals(
            model, dataset, "test", config, args.proposal_count, args.seeds, device
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
