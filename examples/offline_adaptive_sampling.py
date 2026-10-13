#!/usr/bin/env python3
"""Learn from a saved controlled experiment and sample future candidates."""

import argparse
from pathlib import Path

from matmul_inspector.adaptive_sampling import (
    AdaptiveSampler, load_controlled_feedback, write_adaptive_state,
)
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, make_sampling_records, write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=Path("controlled_triton_fc1"),
                        help="controlled benchmark directory")
    parser.add_argument("--output", type=Path, default=Path("offline_adaptive_search"))
    args = parser.parse_args()
    if not args.feedback.is_dir():
        raise SystemExit(
            f"feedback directory not found: {args.feedback}. Run the controlled benchmark first."
        )
    if args.output.exists():
        raise SystemExit(f"output path already exists: {args.output}")

    source_id, records = load_controlled_feedback(args.feedback)
    search_space = TritonGemmSearchSpace(
        {"block_m": [32, 64], "block_n": [64, 128], "block_k": [32, 64],
         "num_warps": [4, 8], "num_stages": [2, 3]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
        identifier="fc1-controlled-sm89-v1",
    )
    sampler = AdaptiveSampler(search_space, seed=2026, epsilon=0.25, prior_strength=2.0)
    before = sampler.parameter_scores()
    diagnostics = sampler.update(records, source_id=source_id)
    after = sampler.parameter_scores()
    future = sampler.sample(10, unique=True)

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "sampled_candidates.jsonl",
                make_sampling_records(future, sampler))
    write_adaptive_state(args.output / "adaptive_state.json", sampler)

    print(f"Feedback source: {source_id}")
    print(f"Eligible evaluations: {diagnostics['eligible_evaluation_count']}; "
          f"excluded: {diagnostics['excluded_observation_count']}")
    print(f"Exclusion reasons: {diagnostics['exclusion_reasons']}")
    print("Parameter score changes (0.5 is the uniform prior mean):")
    for name in before:
        changes = ", ".join(
            f"{value}: {before[name][value]:.4f}->{after[name][value]:.4f}"
            for value in before[name]
        )
        print(f"  {name}: {changes}")
    print("Sampled ten unique, previously unevaluated candidates:")
    for candidate in future:
        print(f"  {candidate.to_dict()}")
    print(f"Saved replayable state and candidate records to {args.output}")


if __name__ == "__main__":
    main()
