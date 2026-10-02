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

## cuBLAS baseline

`cublas` is a vendor-optimized FP32 SGEMM baseline for measuring the remaining
optimization gap. It is selectable as reference or candidate, without changing
any custom kernel. CUDA builds link `CUDA::cublas` from the detected toolkit;
CMake 3.17 and CUDA toolkit 11+ are required for this integration. CPU-only builds
do not search for or link cuBLAS.

Row-major A(M×K), B(K×N), C(M×N) are interpreted as column-major transposes.
`cublasSgemm(N,N, N,M,K, B,lda, A,ldb, C,ldc)` computes Cᵀ=BᵀAᵀ with alpha=1,
beta=0. Leading dimensions are **B.row_stride, A.row_stride, C.row_stride**.
Padded rows work directly; there is no layout conversion or temporary packing.
Dimensions/strides must fit the SGEMM int interface. Empty products follow the
existing zero-output behavior without launching cuBLAS.

The handle explicitly uses `CUBLAS_PEDANTIC_MATH`, host scalar pointers, the default
stream, and `CUBLAS_ATOMICS_NOT_ALLOWED`. Pedantic mode requests conventional FP32
arithmetic rather than TF32/Tensor Core acceleration; the math mode is queried and
recorded. This configuration is not a claim about peak reduced-precision cuBLAS
performance. See NVIDIA's [cuBLAS math-mode documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t).
cuBLAS can select different implementations and accumulation orders; neither
bitwise identity with serial/custom kernels nor reproducibility across toolkit
versions is promised.

One handle is reused for the warmups and all samples within each backend call.
CUDA events on the same default stream bracket SGEMM submission/execution. Host
and device allocations, transfers, handle creation/configuration/destruction,
output hashing and diagnostics are excluded from kernel-only timing. cuBLAS may
launch several internal kernels; their work is included. Submission/scheduling gaps
can affect tiny event intervals. End-to-end timing deliberately includes setup,
transfers and destruction, consistent with the existing end-to-end API.

Metadata records cuBLAS version, math/atomics mode, TF32 policy, mapping, conversion
status and timing scope. Per-shape `cublas_leading_dimensions` describes the packed
CLI matrices; padded API callers supply their actual row strides. Summary rows
identify `cublas`, tile size 0 (library internal tiling unspecified), pedantic FP32,
and unspecified accumulation order. Existing replay, hashes and native numerical
diagnostics are retained. Different output hashes are not an automatic failure.

Tests compare numerical values at explicit atol=rtol=1e-4, including partial tiles
and padded rectangular inputs. CLI defaults remain atol=1e-6, rtol=1e-5; they can
reject differences between correct FP32 algorithms near cancellation. For example,
the 1024 random-input comparison here had maximum absolute error 1.81e-5 and
1.23% default-tolerance failures. No tolerance is silently widened. Reports must
distinguish numerical acceptance under chosen tolerances from bitwise agreement.

### Representative measurement

RTX 4060 Ti, Release, NVCC 13.3.73, cuBLAS version code 130600, pedantic FP32. The requested counts were retained: auto used 10 warmups, 50 iterations, 10 trials, 5000 bootstrap samples; 2x2/4x2 used 5/30/5/2000. To limit runtime, sizes 512, 1536, 3072 and 3328 were skipped in all three sweeps.

GFLOP/s below is derived consistently from median latency (2N³ / median seconds), not the mean-derived GFLOP/s column in native summary.csv. Percentage is cuBLAS median / custom median × 100. Measurements are separate sequential sweeps.

| Size | Custom | cuBLAS ms | Custom ms | cuBLAS GFLOP/s | Custom GFLOP/s | % cuBLAS | Default tolerance |
|---:|---|---:|---:|---:|---:|---:|---|
| 256 | auto | 0.009888 | 0.019456 | 3393.4 | 1724.6 | 50.82% | FAIL |
| 1024 | auto | 0.198656 | 0.792576 | 10810.1 | 2709.5 | 25.06% | FAIL |
| 2048 | auto | 1.396640 | 6.734256 | 12300.9 | 2551.1 | 20.74% | FAIL |
| 3200 | auto | 5.743456 | 31.685536 | 11410.6 | 2068.3 | 18.13% | PASS |
| 4096 | auto | 12.356608 | 65.342449 | 11122.7 | 2103.4 | 18.91% | PASS |
| 256 | 2x2 | 0.009920 | 0.019456 | 3382.5 | 1724.6 | 50.99% | FAIL |
| 1024 | 2x2 | 0.199776 | 0.820544 | 10749.5 | 2617.1 | 24.35% | FAIL |
| 2048 | 2x2 | 1.384448 | 6.943232 | 12409.2 | 2474.3 | 19.94% | FAIL |
| 3200 | 2x2 | 5.892608 | 34.039297 | 11121.7 | 1925.3 | 17.31% | PASS |
| 4096 | 2x2 | 12.572672 | 116.736511 | 10931.6 | 1177.3 | 10.77% | PASS |
| 256 | 4x2 | 0.009216 | 0.028672 | 3640.9 | 1170.3 | 32.14% | FAIL |
| 1024 | 4x2 | 0.183088 | 0.712672 | 11729.2 | 3013.3 | 25.69% | FAIL |
| 2048 | 4x2 | 1.367040 | 7.184160 | 12567.2 | 2391.4 | 19.03% | FAIL |
| 3200 | 4x2 | 5.933008 | 32.245728 | 11046.0 | 2032.4 | 18.40% | PASS |
| 4096 | 4x2 | 12.331392 | 64.239105 | 11145.5 | 2139.5 | 19.20% | PASS |

The defaults rejected 17/65536 outputs at 256, 12948/1048576 at 1024, and 110190/4194304 at 2048 (maximum absolute differences 3.81e-6, 1.81e-5, and 5.05e-5 respectively). These captures correctly have status tolerance_failed. The rectangular check passed defaults; a separate 1024 check passed explicit atol=rtol=1e-4. Tests also use explicit 1e-4 tolerances, without changing CLI defaults.

At 3200 and 4096 the observed outputs were bitwise identical for all compared variants, including cuBLAS. That observation is size/toolchain-specific, not a cuBLAS guarantee. The different agreement pattern is consistent with library implementation selection, but no internal algorithm was profiled or identified.

Artifacts: `results/kernel-optimization/cublas-vs-auto/`, `cublas-vs-2x2/`, and `cublas-vs-4x2/`. Inspect timing_statistics.csv and the original summaries for reliability flags and confidence intervals. Custom kernels reaching only a fraction of vendor throughput quantify remaining optimization opportunities; lower throughput is not a correctness failure. No autotuning or profiler run was added, and dirty-source provenance is explicitly recorded.
