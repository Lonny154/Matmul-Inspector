# Tensor Game fused-input lowering study

## Research hypothesis

The existing Tensor Game runtime spends more time preparing and reconstructing
block operands than its rank reduction saves in GEMM work. This experiment asks
whether forming each rank term's input combinations inside the Triton GEMM load
path can remove enough launches and full-block materialization to improve the
complete operation. The unfused runtime remains the default and reference.

## Baseline and fused architectures

Both paths use the same validated generic `RankOneAction(u, v, w)` contract.
They use IEEE FP32, FP32 accumulation, 32x32x16 GEMM tiles, four warps, three
stages, rank-product storage, and eager output reconstruction.

The unfused path creates four block views per source matrix. For every rank term
it allocates two half-size matrices, initializes them, applies each nonzero `u`
or `v` coefficient with an eager update, and launches the existing Triton GEMM.
All rank products remain live while four output blocks are reconstructed and
copied into the final output.

The optional `fused_input` path passes the original A and B matrices plus the
rank term's compile-time coefficients to a separate Triton kernel. Each program
loads the required source-block tiles, applies the exact coefficients in the
load path, calls `tl.dot`, and writes the same rank-product layout. It does not
allocate complete input-combination matrices. It is generic across conventional
and Strassen decompositions. Output reconstruction is intentionally unchanged.

## Correctness

The benchmark used zero, identity, negative, and seeded random inputs at sizes
128, 256, 512, and 1024. It compared both strategies for conventional rank-8
and Strassen rank-7 against PyTorch with the existing `rtol=5e-3`, `atol=5e-3`
rule. All 64 rows passed.

| Algorithm | Strategy | Maximum absolute error | Largest per-case mean absolute error | Maximum difference from unfused |
|---|---|---:|---:|---:|
| Conventional | Unfused | 1.373291e-4 | 1.039860e-5 | 0 |
| Conventional | Fused input | 1.373291e-4 | 1.039860e-5 | 0 |
| Strassen | Unfused | 2.403259e-4 | 1.980958e-5 | 0 |
| Strassen | Fused input | 2.403259e-4 | 1.980958e-5 | 0 |

The tested outputs were bitwise identical across strategies. This is an
observation for these coefficients, compiler, and inputs. Correctness does not
depend on bitwise equality and future compiler operation ordering may differ.
Maximum elementwise relative error remains misleading near reference zeros.

## Benchmark methodology

The RTX 4060 Ti run used Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime
12.8, Triton 3.6.0, driver 616.92, and compute capability 8.9. PyTorch TF32 was
disabled. Compilation and correctness ran before timing. Each size used ten
warmups, five independent runs, 30 samples per implementation per run, seed
`20261016`, and randomized implementation order per run.

CUDA events measure the complete queued device operation. That includes input
preparation where required, GEMMs, reconstruction, and output copies. It excludes
Python dispatch time, JIT compilation, and host allocation time. The per-sample
`end.synchronize()` waits for completion but is outside the reported event
interval. Device allocator effects outside the event interval may be omitted.

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/benchmark_tensor_game_triton_lowering.py \
  --output artifacts/tensor-game/triton-fused-input-v1/results.json \
  --sizes 128 256 512 1024 --warmups 10 --iterations 30 --runs 5 \
  --seed 20261016 --precisions ieee --strategies unfused fused_input
```

## Complete-operation results

Values are medians of five run medians. Ranges are the minimum and maximum run
median. Speedup is unfused divided by fused.

| Size | PyTorch | Algorithm | Unfused | Fused input | Fused speedup | Fused / PyTorch |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 0.0399 ms | Conventional | 0.6364 ms (0.5921-1.3389) | 0.3405 ms (0.3058-0.7672) | 1.869x | 8.53x slower |
| 128 | 0.0399 ms | Strassen | 0.7212 ms (0.6454-1.4847) | 0.3446 ms (0.3184-0.7602) | 2.093x | 8.63x slower |
| 256 | 0.0440 ms | Conventional | 0.6279 ms (0.5965-0.6476) | 0.3502 ms (0.3141-0.3590) | 1.793x | 7.97x slower |
| 256 | 0.0440 ms | Strassen | 0.7223 ms (0.6531-0.7392) | 0.3674 ms (0.3325-0.3939) | 1.966x | 8.36x slower |
| 512 | 0.1075 ms | Conventional | 0.6917 ms (0.6497-0.7368) | 0.3973 ms (0.3405-0.4163) | 1.741x | 3.69x slower |
| 512 | 0.1075 ms | Strassen | 0.8074 ms (0.7726-0.8197) | 0.4152 ms (0.3610-0.4516) | 1.945x | 3.86x slower |
| 1024 | 0.2207 ms | Conventional | 0.6819 ms (0.6611-1.1263) | 0.4844 ms (0.4571-1.0056) | 1.408x | 2.19x slower |
| 1024 | 0.2207 ms | Strassen | 0.7450 ms (0.7409-1.1325) | 0.5105 ms (0.4724-1.0306) | 1.459x | 2.31x slower |

Fusion improves both algorithms, and its relative benefit is larger for
Strassen at every tested size. Strassen has more input-combination work to
remove. It still remains 1.2% to 5.3% slower than conventional in the fused
path. The experiment therefore reduces the overhead that hid rank reduction,
but it does not make Strassen or either Triton path faster than cuBLAS.

The first run median is an outlier for several 128 and 1024 implementations.
The raw values are preserved. The direction and large size of the fused benefit
remain visible in the other runs, but the wide ranges limit fine-grained claims.

## Phase-level evidence

The separate profiler labels the fused rank batch `fused_gemm` because input
combination is inseparable from its GEMMs. It does not invent a zero-duration
preparation phase. At 1024, median-of-run-median values were:

| Algorithm | Unfused preparation | Unfused GEMMs | Fused GEMMs plus input formation | Unfused reconstruction | Fused-run reconstruction |
|---|---:|---:|---:|---:|---:|
| Conventional | 0.7410 ms | 0.4283 ms | 0.5273 ms | 0.4219 ms | 0.5422 ms |
| Strassen | 1.0296 ms | 0.4695 ms | 0.4796 ms | 0.5995 ms | 0.6256 ms |

These independently timed phases have large and sometimes multimodal ranges.
The reconstruction code is identical across strategies, yet its medians differ.
That demonstrates environmental variation rather than a strategy-dependent
reconstruction mechanism. Phase medians must not be summed or used instead of
the interleaved complete-operation result.

## Launch counts and intermediate storage

The table uses a static source-level model of eager operations. It is not an
Nsight measurement. Counts include tensor initialization/update kernels and the
four output copies. Bytes use half-size FP32 blocks at full size 1024.

| Algorithm | Strategy | Preparation launches | GEMMs | Reconstruction plus copies | Total launches | Input-combination buffers / bytes | Rank-product storage |
|---|---|---:|---:|---:|---:|---:|---:|
| Conventional | Unfused | 32 | 8 | 16 | 56 | 16 / 16 MiB | 8 MiB |
| Conventional | Fused input | 0 | 8 | 16 | 24 | 0 / 0 | 8 MiB |
| Strassen | Unfused | 38 | 7 | 20 | 65 | 14 / 14 MiB | 7 MiB |
| Strassen | Fused input | 0 | 7 | 20 | 27 | 0 / 0 | 7 MiB |

Input combinations are short-lived call temporaries. Rank products live through
reconstruction. Fusion demonstrably removes the Python-visible preparation
operations and complete combination tensors. It does not establish lower device
traffic. A source-level 1024 estimate is 32 MiB versus 256 MiB of operand reads
for conventional and 38 MiB versus 384 MiB for Strassen. The fused estimate is
higher because each source block can be reloaded for every orthogonal output
tile. Cache behavior and actual DRAM transactions were not measured.

## Compiler observations

The capture contains four unfused size specializations and 60 fused
size/coefficient specializations. Coefficients are compile-time constants, so
zero terms disappear and distinct rank terms can produce distinct binaries.
All hashes are recorded. CUBINs were inspected locally but are not intended for
Git. The commit preserves the complete summary plus representative TTIR, TTGIR,
PTX, SASS, and metadata for unfused and the 6/9/12-load fused variants.

| Strategy | Registers/thread | Dynamic shared memory | Static async global-to-shared loads | PTX size | Local / stack |
|---|---:|---:|---:|---:|---:|
| Unfused | 48 | 8 KiB | 6 | 19,777-19,787 bytes | 0 / 0 |
| Fused input | 41-48 | 12-20 KiB | 6, 9, or 12 | 20,508-23,719 bytes | 0 / 0 |

SASS has the same 6/9/12 `LDGSTS` pattern. Both IEEE kernels contain 128 static
FP32 FMA operations and no Tensor Core MMA instructions. Fusion increases code
size, shared-memory use, compiler specialization count, and load instructions
for multi-term combinations. It did not cause spills or higher maximum register
allocation in this capture. Static instruction occurrences are not dynamic work.

## Static cost model and limitations

The original algebraic descriptor becomes more useful directionally: Strassen's
larger preparation count predicts a larger relative fusion benefit. It remains
insufficient for absolute timing. It does not model launches, shared memory,
coefficient-specialized code, repeated tile loads, caching, allocation, or eager
reconstruction. Rank alone also remains insufficient because fused Strassen is
still slower than fused conventional.

This is one GPU, one software stack, IEEE FP32, one fixed tile geometry, and a
nonrecursive 2x2 lowering. No Nsight trace or hardware memory-traffic counters
were captured. Launch counts and operand reads are static estimates. CUDA-event
latency excludes host dispatch. The benchmark allocates output and rank-product
tensors on each call. No output fusion, preallocation, tile tuning, search-policy
change, or TF32 tuning was included.

## Recommended next experiment

Capture one Nsight Systems and Nsight Compute run at size 1024 for both fused
algorithms. Verify actual kernel counts, cache/DRAM traffic, achieved occupancy,
and the effect of 12-20 KiB shared memory. That evidence can test whether repeated
source loads are served by cache and whether launch removal explains the measured
complete-operation gain. Keep geometry, IEEE precision, decomposition, and
reconstruction fixed.
