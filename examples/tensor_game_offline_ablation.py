"""Run the controlled TG-4B.2B offline-learning ablation."""

import argparse
import json
from pathlib import Path

from matmul_inspector.tensor_game_offline_ablation import (
    integrity_audit,
    run_ablation,
    source_evidence_digests,
    train_candidate,
)
from matmul_inspector.tensor_game_offline_learning import OfflineTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print the frozen configuration")
    mode.add_argument("--smoke", action="store_true", help="run one joint-natural epoch")
    mode.add_argument("--train", action="store_true", help="explicitly run all five candidates")
    parser.add_argument("--experience", type=Path, default=Path("artifacts/tensor-game/tg4b2a-diverse-full-v1"))
    parser.add_argument("--source-checkpoint", type=Path, default=Path("artifacts/tensor-game/tg3b-mlp-256-seed42"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tensor-game/tg4b2b-offline-ablation-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    training = OfflineTrainingConfig(
        mode="joint", epochs=30, batch_size=128, learning_rate=0.001,
        policy_coefficient=1.0, value_coefficient=1.0, gradient_clip=1.0,
        patience=3, seed=42, device=args.device,
    )
    audit = integrity_audit(args.experience)
    digests = source_evidence_digests(args.experience, args.source_checkpoint)
    print(f"integrity_valid={audit['valid']}; held_out_metrics_accessed={audit['held_out_metrics_accessed']}")
    print(f"source_digests={json.dumps(digests, sort_keys=True)}")
    print(f"resolved_training={json.dumps(training.__dict__, sort_keys=True)}")
    print(f"proposed_output={args.output}")
    if not (args.smoke or args.train):
        print("dry_run=true; pass --smoke or --train explicitly")
        return 0
    if args.smoke:
        result = train_candidate(
            "joint_natural", args.experience, args.source_checkpoint, args.output,
            OfflineTrainingConfig(**{**training.__dict__, "epochs": 1, "patience": 1}),
            balanced=False,
        )
        print(f"smoke_checkpoint={args.output / result['checkpoint']}")
        print(f"validation={result['validation_metrics']['overall']}")
        return 0
    summary = run_ablation(args.experience, args.source_checkpoint, args.output, training)
    print(f"completed_models={summary['models']}")
    print(f"held_out_test_accessed={summary['held_out_test_accessed']}")
    print(f"report={args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
