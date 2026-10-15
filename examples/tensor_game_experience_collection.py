"""Collect and exactly replay a small CPU TensorGame experience buffer."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from matmul_inspector.tensor_game import matrix_multiplication_tensor
from matmul_inspector.tensor_game_dataset import load_dataset, tensor_from_list
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_neural_mcts import ConstantValueEvaluator, NeuralValueEvaluator
from matmul_inspector.tensor_game_search import HeuristicValueEvaluator, ResidualInformedPolicy
from matmul_inspector.tensor_game_experience import CollectionConfig, ReplayBuffer, load_buffer, play_episode, save_buffer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/tensor-game/tg3b-dataset-2000"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tensor-game/tg3b-mlp-256-seed42"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3,))
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    model, model_config, manifest = load_checkpoint(args.checkpoint, "cpu")
    checkpoint = args.checkpoint / manifest["checkpoint"]
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selected = {}
    for row in sorted(dataset.decompositions, key=lambda item: item["decomposition_id"]):
        length = row["construction_length"]
        if row["split"] == "test" and length in (1, 2, 3) and length not in selected:
            selected[length] = (row["decomposition_id"], tensor_from_list(row["target"]), length)
    problems = list(selected.values()) + [("ood_matmul_2x2_eight", matrix_multiplication_tensor(2, 2, 2), 8)]
    config = CollectionConfig(simulations=args.simulations)
    experiment_id = f"tg4a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output or Path("artifacts/tensor-game") / experiment_id
    buffer = ReplayBuffer(capacity_episodes=1000)
    configurations = {
        "residual_heuristic": lambda moves: HeuristicValueEvaluator(),
        "residual_optimistic": lambda moves: ConstantValueEvaluator(float(moves)),
        "residual_trained_value": lambda moves: NeuralValueEvaluator(model, model_config, "cpu"),
    }
    started = datetime.now(timezone.utc)
    for problem_id, target, max_moves in problems:
        for seed in args.seeds:
            for name, evaluator_factory in configurations.items():
                episode, decisions = play_episode(
                    target, max_moves,
                    policy_factory=lambda: ResidualInformedPolicy(model_config.shape, config.coefficients),
                    evaluator_factory=lambda factory=evaluator_factory, moves=max_moves: factory(moves),
                    config=config, seed=seed, experiment_id=experiment_id,
                    configuration_id=f"{name}:{problem_id}",
                    model_provenance={
                        "policy": "residual_informed", "value": name,
                        "checkpoint_sha256": checkpoint_sha if name == "residual_trained_value" else None,
                    },
                )
                buffer.add(episode, decisions)
                print(f"{problem_id} {name} seed={seed}: {episode['outcome']} ({len(decisions)} decisions)")
    save_buffer(output, buffer, {
        "experiment_id": experiment_id, "dataset_id": dataset.manifest["dataset_id"],
        "checkpoint_sha256": checkpoint_sha, "collection_config": config.__dict__,
        "started_at_utc": started.isoformat(),
    })
    reloaded = load_buffer(output)
    validation = reloaded.validate()
    outcomes = {name: sum(row["outcome"] == name for row in reloaded.episodes) for name in ("success", "truncated", "interrupted")}
    size = sum(path.stat().st_size for path in output.iterdir())
    print(f"Saved {len(reloaded.episodes)} episodes and {len(reloaded.decisions)} decisions to {output}")
    print(f"Replay valid: {validation['valid']}; outcomes={outcomes}; bytes={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
