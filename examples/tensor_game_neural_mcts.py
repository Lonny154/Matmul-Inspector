"""Run the TG-3C neural-guided MCTS ablation on held-out targets."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from matmul_inspector.tensor_game_dataset import load_dataset
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_neural_experiments import (
    NeuralMCTSExperimentConfig,
    benchmark_inference,
    dataset_content_digest,
    heldout_curriculum,
    run_trials,
    save_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/tensor-game/tg3b-dataset-2000"))
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--experiment", type=Path, help="TG-3B experiment/checkpoint directory")
    checkpoint.add_argument("--checkpoint", type=Path, help="alias for --experiment")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11))
    parser.add_argument("--wall-time-limit", type=float, default=5.0)
    parser.add_argument("--smoke", action="store_true", help="run one held-out target with at most five simulations")
    args = parser.parse_args()

    experiment = args.experiment or args.checkpoint or Path("artifacts/tensor-game/tg3b-mlp-256-seed42")
    dataset = load_dataset(args.dataset)
    model, model_config, manifest = load_checkpoint(experiment, args.device)
    digest = dataset_content_digest(dataset)
    if manifest["dataset_id"] != dataset.manifest["dataset_id"] or manifest["dataset_content_digest"] != digest:
        raise ValueError("checkpoint and dataset identities do not match")
    if tuple(dataset.manifest["config"]["shape"]) != model_config.shape:
        raise ValueError("checkpoint and dataset shapes do not match")

    simulations = min(args.simulations, 5) if args.smoke else args.simulations
    per_length = 1
    problems = heldout_curriculum(dataset, per_length)
    if args.smoke or args.simulations == 20:
        problems = problems[:1]
    config = NeuralMCTSExperimentConfig(
        seeds=tuple(args.seeds), simulations=simulations, max_actions=args.proposals,
        node_limit=200, max_wall_time_seconds=args.wall_time_limit, heldout_per_length=per_length,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("artifacts/tensor-game") / f"tg3c-{timestamp}"
    diagnostic_residuals = [problem.target for problem in heldout_curriculum(dataset, per_length)[:4]]
    inference_diagnostic = benchmark_inference(
        model, model_config, diagnostic_residuals, device=args.device,
        max_actions=args.proposals,
    )

    def progress(record):
        print(
            f"{record['target_id']} {record['configuration']} seed={record['seed']}: "
            f"{record['termination_reason']} ({record['simulations_completed']} simulations)"
        )

    records = run_trials(model, model_config, config, problems, args.device, progress)
    checkpoint_file = experiment / manifest["checkpoint"]
    save_experiment(
        output, config, records, dataset=dataset, checkpoint_manifest=manifest,
        checkpoint_sha256=hashlib.sha256(checkpoint_file.read_bytes()).hexdigest(),
        inference_diagnostic=inference_diagnostic,
    )
    print(f"Saved {len(records)} trials to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
