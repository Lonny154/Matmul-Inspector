#!/usr/bin/env python3
"""Sample and evaluate five candidates using synthetic CPU-only measurements."""

from pathlib import Path

from matmul_inspector.evaluation import (
    MockEvaluator, MockEvaluatorConfig, read_evaluation_jsonl, write_evaluation_jsonl,
)
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
)


space = TritonGemmSearchSpace(
    {
        "block_m": [32, 64], "block_n": [64, 128], "block_k": [16, 32],
        "num_warps": [4, 8], "num_stages": [2, 3],
    },
    SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
    identifier="mock-evaluation-example",
)
sampler = UniformSampler(space, seed=42)
sampled = make_sampling_records(sampler.sample(5, unique=True), sampler)
mock = MockEvaluator(MockEvaluatorConfig(synthetic_latency=1.25, unit="synthetic_ms"))
evaluations = [mock.evaluate(candidate) for candidate in sampled]

output = Path("mock_evaluations.jsonl")
write_evaluation_jsonl(output, evaluations)
loaded = read_evaluation_jsonl(output)
for result in loaded:
    measurement = result.benchmark
    print(f"{result.evaluation_id.value}: {result.status.value}; candidate="
          f"{result.candidate_id.value}; synthetic_latency="
          f"{measurement.latency} {measurement.unit}")
print(f"Read {len(loaded)} synthetic evaluation records from {output}")
