"""Create a versioned paired TG-3B proposal-quality audit."""

import argparse
import hashlib
import json
from pathlib import Path

from matmul_inspector.tensor_game_dataset import load_dataset
from matmul_inspector.tensor_game_evaluation_audit import (
    checkpoint_sha256,
    run_evaluation_audit,
    save_audit,
)
from matmul_inspector.tensor_game_learning import create_policy_value_model, load_checkpoint, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--proposal-count", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 42))
    parser.add_argument("--untrained-seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    dataset = load_dataset(args.dataset)
    trained, config, experiment = load_checkpoint(args.experiment, args.device)
    if dataset.manifest["dataset_id"] != experiment["dataset_id"]:
        raise ValueError("dataset identifier does not match the checkpoint")
    dataset_digest = hashlib.sha256(
        "".join(record["content_digest"] for record in (*dataset.decompositions, *dataset.samples)).encode("ascii")
    ).hexdigest()
    if dataset_digest != experiment["dataset_content_digest"]:
        raise ValueError("dataset content digest does not match the checkpoint manifest")
    torch.manual_seed(args.untrained_seed)
    untrained = create_policy_value_model(config).to(resolve_device(args.device)).eval()
    device = next(trained.parameters()).device
    audit = run_evaluation_audit(
        trained,
        untrained,
        dataset,
        config,
        device=device,
        proposal_count=args.proposal_count,
        policy_seeds=args.seeds,
        untrained_initialization_seed=args.untrained_seed,
    )
    checkpoint = args.experiment / experiment["checkpoint"]
    audit["checkpoint"] = {
        "experiment_path": str(args.experiment),
        "filename": experiment["checkpoint"],
        "sha256": checkpoint_sha256(checkpoint),
        "model_config": experiment["model_config"],
        "training_seed": experiment["training_config"]["seed"],
        "best_epoch": experiment["best_epoch"],
        "selection_metric": "validation_loss",
        "best_validation_loss": experiment["best_validation_loss"],
        "test_not_used_for_selection": True,
    }
    audit["dataset"] = {
        "path": str(args.dataset),
        "content_digest": dataset_digest,
        "split_decomposition_counts": dataset.manifest["split_decomposition_counts"],
        "generator_version": dataset.manifest["generator_version"],
    }
    save_audit(args.output, audit)
    concise = {
        "output": str(args.output),
        "test_residual_rows": audit["test_residual_rows"],
        "test_decomposition_groups": len(audit["test_decomposition_groups"]),
        "summaries": audit["summaries"],
        "paired": audit["paired_trained_vs_residual_informed"],
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
