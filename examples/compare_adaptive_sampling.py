#!/usr/bin/env python3
"""Run a CPU-only paired uniform-versus-adaptive simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.adaptive_experiment import (
    ARM_NAMES, PairedTrialRunner, SimulatedLandscape, TrialConfig, write_trial_result,
)
from matmul_inspector.sampling import SearchContext, TritonGemmSearchSpace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("adaptive_trial.json"))
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--landscape", choices=("structured", "interaction", "flat"),
                        default="structured")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")

    search_space = TritonGemmSearchSpace(
        {"block_m": [32, 64, 128], "block_n": [64, 128], "block_k": [16, 32],
         "num_warps": [4, 8], "num_stages": [2, 3]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="cpu-simulation"),
        identifier="milestone-3b-simulated-fc1",
    )
    result = PairedTrialRunner(
        search_space,
        SimulatedLandscape(kind=args.landscape, seed=2718, noise_us=0.5),
        TrialConfig(args.budget, args.batch_size, args.replicates, args.seed),
    ).run()
    write_trial_result(args.output, result)

    print("SIMULATED EVIDENCE — not a GPU benchmark or performance claim")
    print(f"trial={result['trial_id']} landscape={args.landscape} output={args.output}")
    for replicate in result["replicates"]:
        parts = []
        for arm in ARM_NAMES:
            final = replicate["arms"][arm]["metrics"][-1]
            parts.append(
                f"{arm}: best={final['best_observed_latency_us']:.3f} us "
                f"regret={final['simple_regret_us']:.3f} us rank={final['best_observed_rank']}"
            )
        print(f"replicate {replicate['replicate']}: " + " | ".join(parts))
    for arm in ARM_NAMES:
        summary = result["aggregate"]["arms"][arm]
        print(f"{arm} aggregate: mean best={summary['mean_best_observed_latency_us']:.3f} us "
              f"mean regret={summary['mean_simple_regret_us']:.3f} us")
    paired = result["aggregate"]["paired"]
    print("paired adaptive-minus-uniform mean best-latency delta="
          f"{paired['mean_adaptive_minus_uniform_best_latency_us']:.3f} us; "
          f"adaptive lower={paired['adaptive_lower_count']}, "
          f"uniform lower={paired['uniform_lower_count']}, ties={paired['tie_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
