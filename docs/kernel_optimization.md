# CUDA Kernel Optimization Case Study

## Baseline
- 16×16 tiled kernel
- 256 threads/block
- one output per thread

## Profiling diagnosis
- LSU ~92%
- ~330M issued instructions
- ~98% occupancy
- ~1.27 ms at 1024³

## Hypothesis
Reduce load/store pressure by increasing per-thread reuse.

## Optimization
- 2×2 register blocking
- 8×8 threads/block
- four outputs/thread
- four accumulators/thread

## Correctness
- bitwise-identical outputs
- awkward square sizes pass
- rectangular case passes
- 29/29 tests pass

## Performance
- ~1.48× faster at 1024³
- ~1.47× faster at 2048³
- ~2.58 TFLOP/s at 2048³

## Profiler before/after
- registers/thread: 24 → 40
- occupancy: ~98% → ~94%
- LSU utilization: ~92% → ~57%
- issued instructions: ~330M → ~136M
- profiler duration: ~1.27 ms → ~0.86 ms

## Interpretation
- more registers
- slightly lower occupancy
- dramatically less instruction/load-store work
- faster overall execution

## 4x4 register blocking
- 4×4 register blocking reduced LSU pressure further, but crossed the optimal point for this kernel design. Using only 16 threads per block halved theoretical occupancy from 100% to 50%, reduced active warps per SM from ~45 to ~23, increased register usage to 72/thread, and increased issued instructions. The added per-thread reuse did not compensate for reduced parallelism and higher instruction overhead.

## Limitations
- custom kernel, not cuBLAS
- single GPU
- FP32 only
- no tensor cores
- profiler replay timing not used for benchmark claims
## Experimental size-aware dispatch

`register-blocked-auto` selects an existing kernel without adding a kernel body:

- When M=N=K and size is at least 3200, select `register-blocked-4x2`.
- Otherwise select `register-blocked-2x2`, including every non-square shape.

The threshold is centralized in `register_blocked_auto_square_threshold` in
`include/cuda_matmul.hpp`; `resolve_cuda_matmul_kernel` is a deterministic helper
that requires no GPU. Explicit kernel selections remain unchanged. Selection is
performed before CUDA-event timing and does not change arithmetic or K order.

This is an experimental heuristic derived from RTX 4060 Ti square FP32 measurements:
2x2 was stronger at smaller sizes, while 4x2 was more robust at large sizes, with a
measured crossover between approximately 3136 and 3200. That evidence does not
justify a rectangular-input heuristic or imply the same threshold on another GPU.
The policy applies as requested on any CUDA device; it does not detect the GPU or
autotune. Hardware-specific autotuning is a future improvement, not implemented here.

Both `--reference register-blocked-auto` and `--candidate register-blocked-auto`
are supported. Console output prints each shape's resolution. `summary.csv` retains
requested names in `kernel`/`reference_kernel` and adds `resolved_kernel` and
`resolved_reference_kernel`. Metadata records the policy identifier, threshold,
and per-shape requested/resolved pairs in `config.kernel_resolution`; saved output
sidecars also record `resolved_kernel`. Tile size remains 16 and accumulation
semantics remain increasing K. Replay preserves the requested policy; compare
recorded resolutions when replaying with a changed implementation.

```sh
./build/matmul-inspector compare --sizes 3136,3200,3328 \
  --reference tiled --candidate register-blocked-auto
```

A dispatch choice is not evidence of optimal performance. Compare matched runs of
both physical kernels, and inspect timing dispersion and confidence intervals;
sequential measurements of even the same kernel can differ with GPU state.

Validation on RTX 4060 Ti (CUDA compiler 13.3.73, existing sm_75 build) used five
warmups, 30 measured iterations per trial, five trials and 2000 bootstrap samples.
All compared outputs matched bitwise. Median-based auto speedups below are
**explicit-kernel median / auto median**, measured separately in each sweep:

| Square size | Selected kernel | Auto vs 2x2 | Auto vs 4x2 |
|---:|---|---:|---:|
| 1024 | 2x2 | 0.930x | 0.896x |
| 2048 | 2x2 | 1.003x | 1.078x |
| 3072 | 2x2 | 1.005x | 1.082x |
| 3136 | 2x2 | 0.982x | 1.018x |
| 3200 | 4x2 | 1.079x | 0.983x |
| 3264 | 4x2 | 1.217x | 1.013x |
| 3328 | 4x2 | 1.373x | 1.043x |
| 4096 | 4x2 | 1.816x | 0.952x |

The large-size switch helped versus 2x2, but the policy did not always select the
fastest measured kernel: explicit 4x2 was faster at 1024 in this run. The first
sweep flagged a first-iteration deviation at 1024. Ratios between auto and its
selected physical kernel also vary because measurements are sequential. These
observations support an experimental policy, not an optimality guarantee.
Full artifacts are in `results/kernel-optimization/auto-vs-2x2/` and
`results/kernel-optimization/auto-vs-4x2/` (generated, not committed).
