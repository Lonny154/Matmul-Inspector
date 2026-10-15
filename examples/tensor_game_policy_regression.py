#!/usr/bin/env python3
"""Run the TG-4C paired policy-regression investigation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from matmul_inspector.tensor_game_policy_regression import (
    add_fixed_state_model_diagnostics,
    add_validation_metrics,
    analyze_trials,
    directory_content_digest,
    evidence_hashes,
    write_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze_trials(args.trials)
    analysis["provenance"] = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "trial_directory": str(args.trials),
        "trial_directory_content_sha256": directory_content_digest(args.trials),
        "evidence_sha256": evidence_hashes([
            args.validation,
            args.baseline_checkpoint / "best_model_state.pt",
            args.trained_checkpoint / "best_model_state.pt",
        ]),
    }
    add_validation_metrics(analysis, args.validation)
    add_fixed_state_model_diagnostics(analysis, args.baseline_checkpoint, args.trained_checkpoint)
    write_analysis(analysis, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
