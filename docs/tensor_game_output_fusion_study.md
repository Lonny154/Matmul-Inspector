# Tensor Game output reconstruction fusion study

## Research question and hypothesis

Can one generic Triton reconstruction launch replace the eager output-block
operations left after input fusion, and does that reduce complete IEEE FP32
latency for conventional and Strassen decompositions?

## Baseline reconstruction audit

Each GEMM writes one contiguous half-size FP32 rank-product buffer. Baseline
reconstruction allocates four more half-size buffers. For each output block it
launches a zero fill and one in-place `add_` kernel per nonzero `w` coefficient,
then copies the four blocks into a full-size output. Conventional therefore uses
4 zero fills, 8 updates, and 4 copies. Strassen uses 4 zero fills, 12 updates,
and 4 copies. No operation combines several products in one GPU pass.

There are no atomics or explicit synchronizations. Default-stream ordering makes
the consumers depend on completed GEMM producers. These Python-level tensor
operations correspond to the static launch estimate, while Triton program
instances within a launch are not separate kernel launches. With fused input,
the previous totals reconcile to 8 GEMMs + 16 reconstruction/assembly launches
= 24 for conventional, and 7 GEMMs + 20 = 27 for Strassen.

## Implemented strategy

`reconstruction="baseline"` remains the default. The optional `"fused"` path
keeps rank-product storage unchanged and launches one generic Triton kernel. Its
three-dimensional grid selects one of four output blocks plus a 32x32 output
tile. Every program loads the needed product elements, selects the validated
`w` coefficients, accumulates FP32 in deterministic rank order, and writes each
final output element once. There are no intermediate output blocks or atomics.
The fused-input GEMM arithmetic and geometry are unchanged.

## Correctness

Zero, identity, negative, and seeded random inputs were tested at 128, 256, 512,
and 1024 for conventional and Strassen. All 64 benchmark rows passed the existing
`rtol=5e-3`, `atol=5e-3` rule against PyTorch.

| Algorithm | Reconstruction | Maximum absolute error | Largest per-case mean error | Maximum difference from baseline |
|---|---|---:|---:|---:|
| Conventional | Baseline | 1.373291e-4 | 1.039860e-5 | 0 |
| Conventional | Fused | 1.373291e-4 | 1.039860e-5 | 0 |
| Strassen | Baseline | 2.403259e-4 | 1.980958e-5 | 0 |
| Strassen | Fused | 2.403259e-4 | 1.980958e-5 | 0 |

The observed outputs were bitwise identical. That is not required by the API and
is not assumed for other compilers or coefficients. Relative error near zero is
retained in raw data but is not the acceptance metric.

## Benchmark methodology

The RTX 4060 Ti environment used Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8,
Triton 3.6.0, driver 616.92, and SM 8.9. Both paths use fused input, IEEE FP32,
32x32x16 GEMM tiles, four warps, three stages, identical inputs, and unchanged
rank products. Ten warmups preceded five runs of 30 interleaved CUDA-event
samples. Seed was `20261016`. Correctness and compilation preceded measurement.

CUDA events include all required device work. They exclude Python dispatch, JIT,
and host allocation time. Per-sample synchronization occurs after the end event.

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/benchmark_tensor_game_triton_lowering.py \
  --output artifacts/tensor-game/triton-output-fusion-v1/results.json \
  --sizes 128 256 512 1024 --warmups 10 --iterations 30 --runs 5 \
  --precisions ieee --strategies fused_input \
  --reconstructions baseline fused
```

## Complete-operation latency

Values are medians of five run medians. Parentheses show their range. Speedup is
baseline divided by fused.

| Size | PyTorch | Algorithm | Baseline | Fused output | Speedup | Fused / PyTorch |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 0.0389 ms | Conventional | 0.3425 (0.3088-0.3548) | 0.2185 (0.1807-0.2300) | 1.568x | 5.62x slower |
| 128 | 0.0389 ms | Strassen | 0.3482 (0.3072-0.3624) | 0.1843 (0.1748-0.2139) | 1.889x | 4.74x slower |
| 256 | 0.0266 ms | Conventional | 0.3158 (0.3140-0.7527) | 0.2053 (0.1945-0.4685) | 1.538x | 7.71x slower |
| 256 | 0.0266 ms | Strassen | 0.3409 (0.3053-0.7756) | 0.1894 (0.1793-0.4131) | 1.800x | 7.11x slower |
| 512 | 0.0963 ms | Conventional | 0.3789 (0.3232-1.1044) | 0.2396 (0.1906-0.6195) | 1.582x | 2.49x slower |
| 512 | 0.0963 ms | Strassen | 0.3569 (0.3248-1.1259) | 0.2396 (0.1853-0.5806) | 1.489x | 2.49x slower |
| 1024 | 0.2474 ms | Conventional | 0.8187 (0.4623-1.1561) | 0.4970 (0.3947-1.0747) | 1.647x | 2.01x slower |
| 1024 | 0.2474 ms | Strassen | 0.8665 (0.4892-1.1863) | 0.4821 (0.3994-1.0731) | 1.797x | 1.95x slower |

Fusion benefits both algorithms. Strassen has more baseline reconstruction
updates and usually gains more relatively. It is slightly faster than
conventional in several fused medians, including 1024, but their wide overlapping
ranges do not establish a stable crossover. Both remain slower than cuBLAS.

## Reconstruction-only timing

Rank products were precomputed, then baseline and fused reconstruction were
interleaved. These medians must not be added to complete-operation medians.

| Size | Conventional baseline / fused | Strassen baseline / fused |
|---:|---:|---:|
| 128 | 0.1402 / 0.0383 ms | 0.1623 / 0.0369 ms |
| 256 | 0.1393 / 0.0368 ms | 0.1807 / 0.0368 ms |
| 512 | 0.1519 / 0.0376 ms | 0.1848 / 0.0381 ms |
| 1024 | 0.1572 / 0.0568 ms | 0.1710 / 0.0490 ms |

Most reconstruction-only ranges are tight. Each 1024 strategy has one outlier
run, preserved in the raw JSON. The isolated evidence supports lower
reconstruction latency without assigning it to one hardware mechanism.

## Launches and intermediate memory

These are static source-level estimates, not profiler-measured traffic. At size
1024, one half-size FP32 block is 1 MiB.

| Algorithm | Reconstruction | GEMMs | Reconstruction launches | Total launches | Output-block intermediates | Rank products |
|---|---|---:|---:|---:|---:|---:|
| Conventional | Baseline | 8 | 16 | 24 | 4 MiB | 8 MiB |
| Conventional | Fused | 8 | 1 | 9 | 0 | 8 MiB |
| Strassen | Baseline | 7 | 20 | 27 | 4 MiB | 7 MiB |
| Strassen | Fused | 7 | 1 | 8 | 0 | 7 MiB |

Product reads are estimated unchanged at 8 MiB conventional and 12 MiB Strassen.
Baseline writes 4 MiB to intermediate blocks and reads/writes another 4 MiB in
assembly, so each output element is written twice. Fused writes the 4 MiB final
output once. Product buffers live through reconstruction in both paths. Actual
cache and DRAM traffic were not measured, and no profiler trace verified launch
counts in this milestone.

## Compiler findings

Coefficient specialization produced four size variants for each algorithm. The
rank-7 kernel has 14 static PTX global loads and 56 FP32 FMAs; rank-8 has 16
loads and 64 FMAs. Both contain two static global stores, use 40 registers per
thread, use no shared memory, and report zero local/stack bytes. SASS shows the
same load/store and FFMA categories. Counts are static, not dynamic.

Baseline reconstruction consists of PyTorch-generated fill, add, and copy
kernels, so this capture has no single baseline reconstruction artifact to
compare instruction-for-instruction. The meaningful source/compiler contrast is
many framework kernels and four intermediate buffers versus one coefficient-
specialized Triton kernel. Hashes for every captured specialization are in the
compiler summary; representative rank-7 and rank-8 text artifacts are preserved.

## Implications, limitations, and next experiment

The algebraic post-addition descriptor correctly predicts that Strassen has more
baseline reconstruction work and more to gain from fusion. It still does not
model launch count, intermediate writes, compiler specialization, cache behavior,
or GPU state. A hardware-aware cost should separate rank-product count, nonzero
`w` reads/arithmetic, reconstruction launches, and output materialization.

This is one GPU and software stack with fixed IEEE geometry. Complete-operation
timings show large state-dependent variation. CUDA events exclude host dispatch.
Launches and bytes are static estimates. There is no Nsight traffic, occupancy,
or timeline evidence. Output fusion is separate from GEMM and retains all rank
products, as required.

The next experiment should preallocate rank products and the final output, then
capture an Nsight Systems/Compute comparison at 1024. That would separate
allocation effects, verify launch counts and traffic, and test whether the
remaining roughly 2x cuBLAS gap is dominated by seven/eight small GEMMs or by
runtime overhead.
