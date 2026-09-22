# Controlled FMA contraction and arithmetic association

## Two separate variables

FP32 `a*b+c` can round once with fused multiply-add, or twice with a separate
multiply and add. CUDA's explicit `__fmaf_rn(a,b,sum)` requests one round-to-nearest,
ties-to-even operation. `__fadd_rn(__fmul_rn(a,b),sum)` requests two such operations;
these explicit multiply/add intrinsics cannot be contracted. This is a CUDA-specific
control, avoiding translation-unit/global flag changes. See the
[NVIDIA intrinsic contract](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

`cuda-naive-fma` and `cuda-naive-no-fma` instantiate the same sequential kernel
body, thread assignment, strides, inputs and increasing-K loop. Only the arithmetic
primitive differs. The original `naive` and `tiled` implementations retain compiler
defaults. Machine instruction counts naturally differ when one FMA becomes two
instructions, but the multiplication terms and mathematical algorithm do not.

Association is separate: `(a+b)+c` need not equal `a+(b+c)`. The
`cuda-naive-reordered` kernel keeps explicit FMA and splits even and odd K terms
between two private FP32 accumulators, then performs one explicit FP32 add.
Both subsequences retain increasing K, including odd K dimensions. One thread
owns each output; there are no atomics, shared partials, races, mixed precision,
Tensor Cores or cross-thread reductions. Comparing it with `cuda-naive-fma`
changes association while keeping contraction fixed. Do not use no-FMA versus
reordered as the primary causal experiment: both controls would change.

## Instruction verification

The build's `verify-fp` target runs `scripts/verify_fp.py` with `cuobjdump --dump-ptx`
on the actual CUDA library. It checks every embedded controlled entry for expected
FP32 instruction classes, rejects ambiguous `mad`, out-of-line calls, malformed
bodies and missing entries, and embeds the outcome, method, counts and library
SHA-256 into metadata. Exact counts/registers are not required by the checker.
The script returns nonzero on failure; the build's `--allow-unverified` permits
ordinary kernels to remain usable but records false and blocks controlled CLI runs.

Python 3, cuobjdump and embedded PTX are required for automatic verification.
For CMake architecture selection, retain a virtual target: `89` or
`89-real;89-virtual`, not only `89-real`. Inspection needs no GPU and runs in the
CUDA CI job. To inspect manually:

```sh
python3 scripts/verify_fp.py --binary build-cuda/libcuda_matmul.a --report build-cuda/fp_verification.json
cuobjdump --dump-ptx build-cuda/libcuda_matmul.a
cuobjdump --dump-sass build-cuda/libcuda_matmul.a
```

Actual NVCC 13.3.73 Release inspection of the measured library:

| Kernel | PTX FMA / MUL / ADD | Embedded sm_75 SASS FFMA / FMUL / FADD |
| --- | ---: | ---: |
| explicit FMA | 5 / 0 / 0 | 5 / 0 / 0 |
| no-FMA | 0 / 5 / 5 | 0 / 5 / 5 |
| reordered | 2 / 0 / 1 | 16 / 0 / 1 |

These are static instruction counts after compiler transformations, not runtime
operation counts. The original naive SASS also had 5 FFMA, and tiled had 16 FFMA.
Automatic verification certifies the library's embedded PTX, not a driver JIT
cache or every possible machine-code target. The measured compute_75 PTX is JIT
compiled for the 8.9 GPU; the additional SASS inspection above is of embedded sm_75.
Keep toolchain/architecture/driver details with results. No fast-math flags were
added; explicit round-to-nearest operations are the experiment control.

## Reproduce

```sh
./build-cuda/matmul-inspector compare --reference cuda-naive-fma --candidate cuda-naive-no-fma --sizes 4,256,257,1024 --seed 42 --output results/fma-run
./build-cuda/matmul-inspector compare --reference cuda-naive-fma --candidate cuda-naive-reordered --sizes 4,256,257,1024 --seed 42 --output results/order-run
./build-cuda/matmul-inspector benchmark --reference naive --candidate cuda-naive-reordered --sizes 4,256,257,1024 --seed 42 --warmups 3 --iterations 50 --output results/order-timing
python3 scripts/reproduce.py results/order-run/metadata.json --executable build-cuda/matmul-inspector --output results/order-replay
```

Exit 1 records tolerance failure, which is an expected experimental result in
some configurations. Do not discard those artifacts. The replay script supports
schema 1 and 2, including kernel selections in benchmark mode, input generators
and mismatch caps. It does not reconstruct source/build/hardware from metadata.

## Measured random-input results

RTX 4060 Ti (8.9), NVIDIA CUDA compiler 13.3.73, runtime 13030, driver API 13040,
NVIDIA driver package 616.92, GCC 13.3.0, Release `-O3 -DNDEBUG`, compute_75/sm_75,
no fast-math. Host: AMD Ryzen 7 7700X, Linux 6.18.33.2-microsoft-standard-WSL2.
Measured on 2026-09-22 with dirty source explicitly recorded. Hardware/compiler
specific data and complete artifacts are under
[results/examples/rtx4060ti-controlled-fp](../results/examples/rtx4060ti-controlled-fp/README.md).
Seeds A=42/B=123, `lcg32-v1`, atol=1e-6, rtol=1e-5. Ordinary naive versus explicit
FMA was bitwise identical at all four sizes; the controlled comparisons below use
explicit FMA as reference. Neither result is an exact-arithmetic oracle.

| Candidate | Size | Divergent count (%) | Max ULP | Mean divergent ULP | Max abs error | Max rel error | Tolerance failures (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-FMA | 4 | 3 (18.75000%) | 1 | 1 | 7.45058e-09 | 8.89091e-08 | 0 (0.00000%) |
| no-FMA | 256 | 52183 (79.62494%) | 1740636161 | 99940.5 | 1.3113e-06 | 2.41494 | 0 (0.00000%) |
| no-FMA | 257 | 52508 (79.49855%) | 86332 | 13.7553 | 1.43051e-06 | 0.00628351 | 0 (0.00000%) |
| no-FMA | 1024 | 898868 (85.72273%) | 1770019358 | 8077 | 4.76837e-06 | 53.2345 | 1332 (0.12703%) |
| reordered | 4 | 6 (37.50000%) | 4 | 1.5 | 1.49012e-08 | 3.92136e-07 | 0 (0.00000%) |
| reordered | 256 | 59657 (91.02936%) | 1756159700 | 29912.6 | 4.76837e-06 | 1.53841 | 21 (0.03204%) |
| reordered | 257 | 60131 (91.03999%) | 51900 | 17.9114 | 4.76837e-06 | 0.00376811 | 20 (0.03028%) |
| reordered | 1024 | 999879 (95.35589%) | 1804999722 | 24967.6 | 1.71661e-05 | 62.4518 | 17839 (1.70126%) |

Large ULP distances occur near cancellation, sometimes crossing zero. They count
representable keys, not equal-width error increments or a proof of gross absolute
error. The mean includes all bitwise-divergent finite pairs, including signed-zero
pairs with zero ULP. Read absolute error and tolerance alongside ULP and relative
error. A bitwise mismatch alone never implies tolerance failure.

First bitwise divergence at 1024² was C[0,0] in both comparisons:

| Variant | Reference → candidate | Reference bits → candidate bits | ULP | Abs error | Rel error | Element tolerance |
| --- | --- | --- | ---: | ---: | ---: | --- |
| no-FMA | 2.441600084 → 2.441599607 | `0x401c432d` → `0x401c432b` | 2 | 4.768371582e-7 | 1.95296991e-7 | PASS |
| reordered | 2.441600084 → 2.441598654 | `0x401c432d` → `0x401c4327` | 6 | 1.430511475e-6 | 5.858909731e-7 | PASS |

The first *tolerance* failures were C[0,144] for no-FMA (229 ULP,
1.706182957e-6 absolute error) and C[0,61] for reordering (293 ULP,
2.183020115e-6 absolute error). The console captures retain full values/bits.
At 1024² the finite divergent ULP bins `[0,1,2,3–4,5–8,>8]` were:

```text
no-FMA:   [0, 194236, 199872, 197584, 146209, 160967]
reordered:[0,  95131,  91157, 162322, 230801, 420468]
```

## Sensitive deterministic inputs

These fixtures complement normal seeded random tests. They use exact hexadecimal
FP32 constants, ignore seeds by design (seeds remain recorded), repeat the same
row/column pattern, leave padding untouched, and are versioned in metadata:

- `--input cancellation` (`cancellation-v1`, K>=4): each A row repeats
  `[2^24, 1, -2^24, 1]`; every B element is 1. Products are exact so changing FMA
  cannot explain differences. For K=4 the sequential result is 1 and the even/odd
  result is 2. The latter happens to be the exact sum; this does not establish a
  general accuracy ranking.
- `--input fma-sensitive` (`fma-sensitive-v1`, K>=2): each A row starts
  `[-1, 1+2^-23]` then zeros; B rows start `[1, 1-2^-23]` for every column, then
  ones. Sequential FMA returns `-2^-46`; separate multiply/add returns zero.
  Input terms/order are identical between the kernels.

Both fixtures were run at all four sizes. FMA-sensitive outputs diverged 100%
with max=mean ULP 679477249, max absolute error 1.421085472e-14 and max relative
error 1; **all passed tolerance**. The cross-zero key convention explains the
large ULP count. Cancellation results:

| Size | Sequential → reordered | Divergent | Max = mean ULP | Max abs error | Max rel error | Tolerance failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 → 2 | 100% | 8388608 | 1 | 1 | 16 (100%) |
| 256 | 1 → 128 | 100% | 58720256 | 127 | 127 | 65536 (100%) |
| 257 | 16777216 → 16777344 | 100% | 64 | 128 | 7.629394531e-6 | 0 |
| 1024 | 1 → 512 | 100% | 75497472 | 511 | 511 | 1048576 (100%) |

Use the same compare commands with `--input cancellation` or `--input fma-sensitive`.
The partial K=257 fixture has a large uncancelled final term, so its relative
acceptance threshold is much larger; its tolerance pass is expected.

## Kernel timing (secondary observation)

Three warmups followed by 50 individual CUDA-event measurements, reference first,
then candidate. Allocation/transfers/initialization/comparison/serialization are
excluded. Standard deviation is population standard deviation. GFLOP/s retains
`2*M*N*K` for all kernels; the reordered final add is not charged separately.
Each row uses its own paired ordinary naive mean for the speedup denominator.

| Variant | Size | Naive mean ms | Variant mean ms | Median ms | Min ms | Stddev ms | GFLOP/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FMA | 4 | 0.021860 | 0.005231 | 0.003840 | 0.003072 | 0.006782 | 0.0244708 | 4.1792x |
| FMA | 256 | 0.042614 | 0.034950 | 0.034752 | 0.033344 | 0.002320 | 960.076 | 1.2193x |
| FMA | 257 | 0.036085 | 0.045192 | 0.046368 | 0.034816 | 0.008224 | 751.216 | 0.7985x |
| FMA | 1024 | 2.528952 | 2.501533 | 2.421760 | 2.221888 | 0.279881 | 858.467 | 1.0110x |
| no-FMA | 4 | 0.008312 | 0.004022 | 0.003072 | 0.002784 | 0.004402 | 0.0318269 | 2.0667x |
| no-FMA | 256 | 0.044603 | 0.031910 | 0.031744 | 0.030720 | 0.000479 | 1051.52 | 1.3978x |
| no-FMA | 257 | 0.040953 | 0.033727 | 0.033520 | 0.032640 | 0.001280 | 1006.58 | 1.2142x |
| no-FMA | 1024 | 2.365623 | 2.439210 | 2.328576 | 2.245632 | 0.267856 | 880.401 | 0.9698x |
| reordered | 4 | 0.008118 | 0.006201 | 0.003184 | 0.002944 | 0.009203 | 0.020642 | 1.3092x |
| reordered | 256 | 0.034580 | 0.040223 | 0.036192 | 0.033408 | 0.009078 | 834.216 | 0.8597x |
| reordered | 257 | 0.033551 | 0.034198 | 0.033792 | 0.032768 | 0.001890 | 992.713 | 0.9811x |
| reordered | 1024 | 2.394600 | 2.713601 | 2.579456 | 2.502656 | 0.328539 | 791.378 | 0.8824x |

At 1024² no-FMA took 3.11% longer and reordering 13.32% longer than their paired
naive means. Explicit FMA was about 1.08% faster; that does not establish a
performance improvement over the equivalent ordinary arithmetic. Unlocked clocks,
WSL scheduling and other system activity cause substantial variance. Small-size
ratios (including apparent no-FMA speedups) are especially unsuitable for causal
performance claims. These are raw observations, not optimization conclusions.

## Tests and limits

CPU tests cover aggregate means/maxima, bins, tolerance counts, signed zeros,
negative/cross-sign ULP mapping, NaN/infinity handling, configuration, fixtures,
serialization, schema compatibility and verifier rejection paths. GPU tests check
strided/partial products against explicit FP32 host arithmetic, repeat determinism,
known sensitive residuals, timing and actual compare/artifact replay. CTest skips
GPU execution with code 77 without a device. Validation passed all 8 Release
CUDA-build tests, all 6 CPU-only Release tests and all 6 CPU-only Debug tests.
A separate `89-real` build without PTX recorded `fp_verified: false`, rejected
controlled CLI execution, and successfully ran the ordinary naive/tiled comparison.
CI retains separate CPU and CUDA
compilation jobs; instruction inspection requires no NVIDIA GPU.

Finite tolerance uses the existing `max(atol, rtol*max(abs(ref),abs(candidate)))`
formula. Unequal infinities now fail explicitly, correcting the former infinite
threshold corner case; NaN always fails, including identical payloads. Aggregates
exclude nonfinite pairs and record their counts. Scalar errors remain FP32 and may
overflow for extreme values. For reference zero, relative error retains the legacy
zero result and a separate `zero_reference_nonzero` count. Samples are row-major
and capped, not a full enumeration or worst-error selection. No cross-hardware
reproducibility or exact numerical ground truth is claimed.

A useful next experiment is a controlled sweep of cancellation strength and K
using these same kernels and metrics, to study when a fixed tolerance starts to
fail. No additional accumulation algorithm is implemented here.
