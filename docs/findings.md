# Project Findings

This document summarizes key experimental findings from Matmul-Inspector across performance, numerical reproducibility, summation accuracy, and cross-hardware execution.

## 1. Benchmark and Performance Infrastructure

Matmul-Inspector now supports reproducible benchmarking with:

- CPU and CUDA timing
- kernel-only and end-to-end GPU measurements
- CPU↔GPU crossover analysis
- retained timing samples and trial identities
- median, percentiles, IQR, MAD, coefficient of variation, and bootstrap confidence intervals
- heuristic stability diagnostics
- roofline post-processing using explicit compute and memory-bandwidth ceilings

Roofline analysis uses a simple compulsory-traffic model rather than measured DRAM traffic. Results therefore describe the configured analytical model, not the complete memory behavior of the hardware.

## 2. Multi-Operation Support

The original matrix-multiplication workflow was generalized to support:

- matrix multiplication
- dot product
- FP32 sum reduction

Existing matmul workflows remain compatible while dot and reduction reuse the same benchmarking, diagnostics, replay, reporting, and provenance infrastructure.

CUDA reductions use deterministic multi-stage tree reductions with configurable block sizes of 64, 128, 256, and 512 threads.

## 3. Reduction Order Strongly Affects FP32 Results

Controlled reduction experiments showed that changing accumulation order can materially change the final FP32 result.

Across repeated runs with a fixed configuration, CUDA reductions were bitwise deterministic. However, changing the reduction structure, including CUDA block size, could produce different results for the same input.

Observed examples included:

- random-fixture CPU/GPU differences up to 14 ULP
- CUDA block-size differences up to 65,536 ULP with an absolute difference of 1.0
- cancellation-heavy comparisons producing dramatically larger ULP distances
- one cancellation case where different reduction orders produced values of 0 and 255

These results demonstrate that deterministic execution does not imply identical results across different floating-point evaluation orders.

## 4. Stable Summation Significantly Reduces Error

The project compares several summation strategies:

- forward FP32
- reverse FP32
- pairwise FP32
- Kahan compensated summation
- Neumaier compensated summation
- FP64 accumulation as a higher-precision analysis reference

In smoke studies across random, cancellation, and large-dynamic-range fixtures, exact matches to the FP64 result rounded back to FP32 were:

| Method | Exact matches |
| --- | ---: |
| Forward FP32 | 1/9 |
| Reverse FP32 | 2/9 |
| Pairwise FP32 | 1/9 |
| Kahan FP32 | 5/9 |
| Neumaier FP32 | 9/9 |
| FP64 cast to FP32 | 9/9 |
| CUDA tree reductions | 1–3/9 depending on block size |

For the largest tested cancellation case, naive forward accumulation had an absolute error of 99,224 relative to the FP64 analysis reference. Kahan reduced this error to 388, approximately a 255.7× reduction. Neumaier matched the FP64-rounded FP32 target exactly for that case.

Compensated summation did not improve every tested input, so these results should not be interpreted as a universal ranking.

## 5. Fresh-Process Determinism

Reproducibility was tested across independent process launches rather than only repeated execution within one process.

For the tested reduction, dot, and matmul configurations, repeated fresh-process execution produced identical output bits on the same machine.

This supports a distinction between:

- repeated-run determinism
- process-level reproducibility
- cross-hardware reproducibility

These are evaluated separately by the project.

## 6. RTX 4060 Ti vs Tesla T4

Cross-hardware captures were generated from the same source revision and matched configurations on:

- NVIDIA GeForce RTX 4060 Ti
- NVIDIA Tesla T4

The tested workloads included:

- cancellation reduction, length 257
- random reduction, length 100,003
- cancellation reduction, length 100,003
- random dot product, length 100,003
- tiled matrix multiplication, 257×257

Each configuration was executed in three fresh processes per machine.

For every tested cross-GPU comparison, all six captured outputs matched the RTX 4060 Ti baseline bitwise.

### Key result

For the controlled configurations tested here, moving from a Turing Tesla T4 to an Ada RTX 4060 Ti did not change the numerical output bits.

By contrast, changing the reduction structure or accumulation order on the same hardware did change numerical results.

This suggests that, for these kernels, **algorithmic evaluation order had a larger observed effect on FP32 reproducibility than the tested GPU architecture change**.

This conclusion is limited to the tested kernels, compiler/toolchain behavior, fixtures, block sizes, source revision, and NVIDIA hardware. It is not a general guarantee of cross-GPU CUDA reproducibility.

## 7. Current Interpretation

The project results support several recurring themes:

1. Floating-point reproducibility depends strongly on evaluation order.
2. Bitwise determinism and numerical accuracy are separate properties.
3. Stable summation can substantially reduce accumulation error for difficult inputs.
4. GPU architecture differences do not necessarily imply different numerical outputs when the execution structure remains fixed.
5. Performance and numerical behavior should be analyzed together, but not conflated.

## 8. Future Work

Useful next directions include:

- expanding cross-hardware testing to additional GPU architectures and toolchains
- independent-process experiments across more matmul kernels and reduction block sizes
- stronger reference summation methods
- CPU cache-blocking and SIMD microkernel studies
- hardware-counter-based memory analysis
- extending the operation set only where it helps answer specific performance or numerical-reproducibility questions