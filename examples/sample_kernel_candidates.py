#!/usr/bin/env python3
"""Sample ten Triton GEMM candidates without compiling or running kernels."""

from pathlib import Path

from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
    read_jsonl, write_jsonl,
)


search_space = TritonGemmSearchSpace(
    {
        "block_m": [32, 64, 128],
        "block_n": [64, 128, 256],
        "block_k": [16, 32, 64],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4],
    },
    SearchContext(
        m=512, n=3072, k=768, precision="fp16",
        backend="triton", hardware_target="sm_89",
    ),
    identifier="fc1-initial-space",
)
sampler = UniformSampler(search_space, seed=42)
candidates = sampler.sample(10, unique=True)
for candidate in candidates:
    print(candidate)

output = Path("sampled_candidates.jsonl")
write_jsonl(output, make_sampling_records(candidates, sampler))
loaded = read_jsonl(output)
print(f"Read {len(loaded)} records back from {output}")
