# Matmul-Inspector

Matmul-Inspector is a C++17/CUDA laboratory for inspecting floating-point matrix
multiplication and recording reproducible experiments. It exposes values, physical
addresses, IEEE-754 bits, ULP distances and tolerances alongside kernel timing.
Its goal is transparent numerical analysis, not a replacement for a tuned BLAS.

Floating-point operations round. Changing accumulation order, FMA contraction or
compiler settings can change bits without producing an unacceptable numerical
error. A seed alone is insufficient provenance: source, build flags, hardware,
driver and timing method also matter.

The CPU implementation is a sequential FP32 reference. The naive CUDA kernel
assigns one thread per output; the 16x16 tiled kernel stages A and B in shared
memory. Both retain a single accumulator and increasing-K order, support strided
storage, and handle partial tiles. No Tensor Cores, mixed precision, tree reductions
or cuBLAS are used. Matrix owns layout, numeric/comparison code analyzes values,
Inspector reports them, CUDA executes kernels, and experiment code records runs.

## Build

Requires CMake 3.16+ and a C++17 compiler. Python 3 enables additional integration
checks and experiment replay; the executable itself needs no Python or third-party
JSON/statistics libraries. Linux is the exercised platform; unavailable optional
metadata on other platforms is recorded as `unknown`.

CPU-only (no NVIDIA software or GPU needed):

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DMATMUL_INSPECTOR_ENABLE_CUDA=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu -L cpu --output-on-failure
```

CUDA, when a supported compiler/toolkit is available:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DMATMUL_INSPECTOR_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

CUDA detection defaults to ON and falls back to CPU-only if unavailable. Check
configure output: requesting ON does not force CUDA installation. If necessary,
point `CUDACXX` or `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` at your compiler on the
first configuration. No local installation path is built into CMake. On CMake
3.18+, set `-DCMAKE_CUDA_ARCHITECTURES=...` for your intended target(s); otherwise
the compiler default is used. The selected architecture is captured in metadata.
The optional `cuda_matmul` target exports `MATMUL_INSPECTOR_HAS_CUDA=1`.

Use `-DCMAKE_BUILD_TYPE=Debug` for debugging, Release for timing. Multi-config
builds also need `--config Release` for build/CTest. Warnings default to ON
(`MATMUL_INSPECTOR_WARNINGS`); no fast-math flags are added. `BUILD_TESTING=OFF`
disables tests; `MATMUL_INSPECTOR_GPU_TESTS=OFF` omits GPU test registration while
still compiling CUDA. GPU tests have the `gpu` label and return skip code 77 when
hardware/runtime is unavailable. Normal CI never requires an NVIDIA GPU.

## Benchmark reliability

Benchmark and crossover modes retain raw warmup/measurement samples and support
`--trials`, `--bootstrap-samples`, `--confidence-level`, `--bootstrap-seed`, and
`--percentiles`. Medians are primary; legacy mean-based columns remain available.

```sh
./build-cuda/matmul-inspector benchmark --sizes 4,64,256 \
  --warmups 3 --iterations 50 --trials 5 --seed 42 \
  --bootstrap-samples 2000 --confidence-level 0.95 --output results/reliability
python3 scripts/report.py results/reliability
```

Reports include distributions, per-iteration traces, median bootstrap intervals,
trial variation and heuristic stability flags. `timing_samples.csv`,
`timing_statistics.csv`, and `pairwise.csv` extend existing artifacts. CI overlap
is **not** a significance test. See [benchmark methodology](docs/benchmarking.md)
for bootstrap assumptions, thresholds, timing boundaries and practical pitfalls.

## CPU/GPU crossover

```sh
./build-cuda/matmul-inspector crossover --candidate tiled \
  --sizes 1,2,4,8,16,32,64,128,256 --seed 42 \
  --warmups 3 --iterations 20 --output results/crossover-tiled
python3 scripts/report.py results/crossover-tiled

# Same sweep and optional report in one command (requires matplotlib):
python3 scripts/crossover.py --executable build-cuda/matmul-inspector \
  --candidate naive --sizes 2,4,8,16,32,64,128,256 --seed 42 \
  --warmups 3 --iterations 20 --output results/crossover-naive --report
```

The baseline is the existing **single-threaded FP32 CPU matmul**, not a tuned BLAS.
Each run tests one existing CUDA candidate; repeat with another candidate in a new
run directory. Square sizes are configurable, sorted, and must be unique. CPU,
GPU kernel-only, and GPU end-to-end use the same inputs and sample counts.

- `host_matmul`: steady-clock timing of CPU matmul, including host output allocation
  and zero initialization; input generation and comparison are excluded.
- `kernel_only`: existing CUDA-event timing, excluding allocation and transfers.
  This assumes inputs already on device and is not a host-to-host comparison.
- `end_to_end`: steady-clock timing of the synchronous CUDA wrapper, including its
  validation/device queries, host output allocation, device allocations, pageable
  H2D copies, output initialization, launch, synchronization, D2H copy and device
  frees. No pinned memory, transfer overlap or buffer reuse is introduced.

An extra untimed GPU priming call precedes each GPU timing mode in each trial, even with zero
warmups, to exclude initial context/module setup. Previous host-output destruction,
statistics, hashing, saved-output capture and numerical comparison are outside all
samples. Sizes run ascending; CPU then kernel-only then end-to-end, without order
randomization. Very small CPU timings can approach timer overhead.

`summary.csv` adds `backend` and `timing_mode`; its existing speedup remains
mean-based. **`crossover.csv` uses CPU median / GPU median** and marks the first
sampled size with strictly lower GPU latency separately for each mode. Ties are
not wins; missing/zero medians are unavailable. No interpolation, exact crossover,
or sustained win beyond that point is claimed. `metadata.json`, `console.txt`,
numerical diagnostics and optional `--save-output` retain the existing conventions.
Numerical tolerance failures remain failures even when GPU execution is faster.

Reports include logarithmic CPU/GPU latency and speedup plots, a 1× parity line,
first sampled wins and largest-size speedups. Crossover depends on CPU/GPU,
implementation, compiler, clocks/load, tested sizes and transfer assumptions.
Replay supports crossover metadata through `scripts/reproduce.py`; current
cross-hardware aggregation deliberately rejects crossover runs rather than mixing
timing modes. CUDA-unavailable runs record a skip and return 77. Optional plotting
dependencies and report output rules are described below.

## Reports from saved results

Generate plots without running benchmarks or requiring a GPU:

```sh
python3 -m pip install -r scripts/requirements-report.txt
python3 scripts/report.py results/examples/rtx4060ti-controlled-fp/benchmark-no-fma
```

Open `<run>/report/report.md` for latency, GFLOP/s, recorded kernel speedup,
and bitwise-divergence/tolerance-failure plots. Shapes use explicit M×N×K
labels, including rectangular matrices. Numerical fractions use full-summary
aggregates, never the bounded mismatch samples. Older/missing optional fields
produce warnings and skipped plots; compare-only runs need no timings.
Matplotlib is optional for the project build. `--output <new-directory>` selects
another report location; existing directories are never overwritten. Source
artifacts are read only. These plots summarize a capture, not a hardware ranking.

## Correctness comparison

```sh
# Naive CUDA reference vs tiled CUDA candidate, including a partial tile
./build-cuda/matmul-inspector compare --sizes 4,256,257 --seed 42

# Rectangular CPU vs GPU comparison
./build-cuda/matmul-inspector compare --m 17 --n 19 --k 33 --reference cpu --candidate tiled --atol 1e-6 --rtol 1e-5

# Exercise the same analysis/artifact workflow without a GPU
./build-cpu/matmul-inspector compare --reference cpu --candidate cpu --sizes 4,17 --seed 42 --output results/cpu-check
```

Compare mode performs no timing. It reports bitwise equality, divergent element
count, overall tolerance pass, and first bitwise/numerical divergence independently.
Each divergence includes row/column, reference and candidate values, 32-bit IEEE
patterns, ULP distance, absolute/relative errors, tolerance and element pass/fail.
The report uses `expected` for reference and `actual` for candidate. Kernels may
be `cpu`, `naive`, `tiled`, `cuda-naive-fma`, `cuda-naive-no-fma`, or
`cuda-naive-reordered`. Bitwise mismatch alone is not a correctness failure.
Aggregate reports include divergent percentage, max/mean divergent ULP, maximum
absolute/relative error, tolerance-failure count/percentage and ULP bins.

## Controlled floating-point experiments

FMA contraction rounds `a*b+c` once instead of rounding multiply and add separately.
Arithmetic reordering changes association: `(a+b)+c` can differ from `a+(b+c)`.
These are separate experiments, with the same FP32 inputs in each comparison:

```sh
./build-cuda/matmul-inspector compare --reference cuda-naive-fma --candidate cuda-naive-no-fma --sizes 4,256,257,1024 --seed 42 --output results/contraction
./build-cuda/matmul-inspector compare --reference cuda-naive-fma --candidate cuda-naive-reordered --sizes 4,256,257,1024 --seed 42 --output results/association
./build-cuda/matmul-inspector compare --reference cuda-naive-fma --candidate cuda-naive-reordered --input cancellation --sizes 4 --output results/cancellation
./build-cuda/matmul-inspector benchmark --reference naive --candidate cuda-naive-no-fma --sizes 4,256,257,1024 --seed 42 --warmups 3 --iterations 50 --output results/no-fma-timing
```

Use the explicit FMA reference for attribution: the original `naive` kernel keeps
compiler-default behavior. The two controlled sequential variants share a body;
CUDA round-to-nearest intrinsics force fused versus separate arithmetic. Reordering
keeps FMA but uses deterministic even/odd accumulators in one thread per output.
No global fast-math flags are enabled. Python 3 and `cuobjdump` inspect embedded PTX
at build time; the CLI refuses controlled runs without successful verification.
Keep a PTX target when setting CUDA architectures (for example `89`, not only
`89-real`). Ordinary CPU/naive/tiled builds remain usable without verification tools.

On the measured RTX 4060 Ti/NVCC 13.3 build, the original naive and explicit-FMA
kernels matched bitwise at all four sizes. At 1024², no-FMA diverged in 85.72% of
outputs (1,332 tolerance failures), and reordering in 95.36% (17,839 failures).
These are measurements against the selected reference and tolerance, not proof
that one algorithm is closer to exact arithmetic. See the
[full experiment report](docs/floating_point_experiments.md) for instruction
verification, fixtures, ULP/error tables, timing statistics and replay commands.

## Benchmark and capture results

```sh
./build-cuda/matmul-inspector benchmark --sizes 4,256,257,1024 --seed 42 --warmups 3 --iterations 50 --output results/my-run
```

Benchmark mode measures the selected reference then candidate using CUDA events
(defaults: naive then tiled; CPU kernels are unavailable in benchmark mode). It emphasizes mean,
median, minimum, population standard deviation, GFLOP/s and speedup. Input creation,
allocation, initialization, host/device transfers and warmups are excluded.
Comparison, statistics, console output and artifact writing happen outside timed
execution. An untimed post-run comparison records numerical agreement in the CSV.
Detailed divergence reports are available through compare mode and mismatch CSV.

`--sizes` accepts square sizes; alternatively use all of `--m`, `--n`, `--k` for
one rectangular configuration. A's default seed is 42; B's is `seed + 81` modulo
2^32, overridable with `--seed-b`. Both seeds and generator version are saved.
No global RNG state is used. Defaults: 3 warmups, 20 iterations, atol=1e-6,
rtol=1e-5. See `--help` for strict argument rules. No arguments preserves the
original 2x3 times 3x2 CPU/GPU demo; `--benchmark-cuda [iterations]` remains an alias.

A new result directory contains:

```text
metadata.json   # configuration, provenance, environment, method and completion status
summary.csv     # one row per configuration/kernel, timings and comparison results
mismatches.csv  # bounded row-major mismatch samples; default cap 100 per configuration
console.txt     # experiment stdout/stderr, including dirty-tree warnings
```

Existing result directories are never overwritten. Dirty sources are allowed, but
print **WARNING: DIRTY SOURCE TREE OR BUILD** and set `git_dirty: true`. The commit
hash alone cannot reconstruct such a run. Build-time and runtime Git state are
recorded separately. Optional metadata failures do not prevent execution.

Metadata includes UTC timestamp, commit/dirty state, build type, C++ and CUDA
compiler versions/flags, CUDA architecture/runtime, NVIDIA driver version, GPU and
compute capability, CPU/OS, dimensions, dtype, kernel/tile, seeds, counts,
tolerances and timing methodology. See [artifact/provenance details](docs/experiments.md)
for exact semantics, unknown values, exit codes and remaining limitations.

## Compare collected runs across machines

Every compare/benchmark output now has a SHA-256 fingerprint of its logical FP32
bits (little-endian, excluding padding). Use `--save-output` to retain optional
binary matrices for detailed comparison later; hashing/capture is outside timing.

```sh
./build-cuda/matmul-inspector benchmark --reference naive --candidate cuda-naive-fma --sizes 4,256,257,1024 --seed 42 --save-output --output results/cross-hardware/baseline
# After replaying and collecting a second machine's result directory:
python3 scripts/compare_hardware.py results/cross-hardware/baseline results/cross-hardware/second-machine --comparator build-cpu/matmul-compare-outputs --output results/cross-hardware/aggregate
```

The first run is the baseline unless `--baseline` selects another supplied run.
The tool reads hardware identity from metadata, validates configuration/source
compatibility, keeps within-run naive speedups separate from cross-run ratios,
and calls the existing C++ diagnostics for saved outputs. Missing hashes/matrices
are reported explicitly. Dirty or different source is flagged and suppresses
cross-run timing ratios. There is no remote execution or automatic hardware ranking.
See the [cross-hardware workflow and binary format](docs/cross_hardware.md) for
CPU-only validation, exact second-machine replay commands, compatibility rules,
aggregate artifacts and interpretation limits.

## Replay the example

The small [RTX 4060 Ti example](results/examples/rtx4060ti-naive-vs-tiled/README.md)
is a real capture from the new infrastructure. Its source was dirty; treat it as
an example of the format, not a fully reconstructible or universal baseline.
Build your binary first, then replay its configuration into a new directory:

```sh
python3 scripts/reproduce.py results/examples/rtx4060ti-naive-vs-tiled/metadata.json --executable build-cuda/matmul-inspector --output results/replay
```

Use `--dry-run` to print the command without execution. Replay reconstructs the
configuration and inputs, not the original source, compiler or hardware. Preserve
the source commit and build settings along with artifacts for publication.

## Original hardware-specific measurement

The preceding tiled-kernel milestone measured the following on an RTX 4060 Ti
with CUDA 13.3.73, Release `-O3`, compiler-default compute_75/sm_75 target. This is
the original historical table, not the fresh example capture linked above:

| Size | Naive ms | Tiled ms | Speedup | Naive GFLOP/s | Tiled GFLOP/s |
| ---- | -------: | -------: | ------: | ------------: | ------------: |
| 4 | 0.010227 | 0.007928 | 1.290x | 0.0125 | 0.0161 |
| 256 | 0.036686 | 0.027884 | 1.316x | 914.64 | 1203.35 |
| 257 | 0.046489 | 0.037900 | 1.227x | 730.26 | 895.75 |
| 1024 | 2.573123 | 1.213059 | 2.121x | 834.58 | 1770.30 |

All tested sizes were bitwise identical between naive and 16x16 tiled CUDA.
Both kernels' generated code used FP32 FMA; no fast-math flags were enabled.
CUDA events measured 50 iterations after three warmups, excluding memory allocation
and host/device transfers. Tiny matrix timings are dominated by launch/scheduling
effects. GPU clocks, contention, driver scheduling and caches affect larger cases
as well. These are example measurements, not universal speedup claims or valid
cross-hardware comparisons. See [NVIDIA's floating-point guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).

## Numerical conventions and storage

Absolute error is `abs(candidate-reference)`. Relative error divides by
`abs(reference)`; the original reporting convention returns zero when reference
is zero, so use absolute error/tolerance near zero. Numerical acceptance uses:

```text
abs(candidate-reference) <= max(atol, rtol * max(abs(candidate), abs(reference)))
```

Exact equality is accepted first; NaN operands never pass. Signed zeros compare
numerically equal with ULP distance zero, though their bits differ. ULP ordering
is sign-aware; adjacent same-sign floats are one ULP apart. NaN distance is
UINT32_MAX. The original bit mapping gives signed zeros adjacent keys, so distances
crossing zero include that extra key except for equal zeros. Equal infinities pass;
unequal infinities now explicitly fail tolerance. Aggregate ULP/errors exclude
nonfinite pairs and report their counts separately. The original zero-reference
relative-error convention is retained and flagged by a separate counter. Generated
benchmark inputs are bounded finite floats. These conventions are covered by tests.

`Matrix(rows, cols)` is packed row-major; `Matrix(rows, cols, row_stride)` permits
padding. Strides are in floats and must be at least cols. `data()` includes padding;
logical indexing, Inspector, CPU matmul and both CUDA kernels respect the stride.
`matmul(a,b,stride)` and `cuda_matmul(a,b,stride,kernel)` select padded outputs.
The experiment CLI uses packed storage; API tests exercise padded layouts.

## Repository and tests

Existing kernel/storage files stay in `include/` and `src/`. New comparison,
benchmark statistics and experiment modules sit alongside them, with build
provenance scripts in `cmake/`, replay in `scripts/`, documentation in `docs/`,
and only small curated artifacts under `results/examples/`. Other generated
results and build directories are ignored by Git.

CPU tests cover numeric helpers, negative ULP ordering, tolerances, bitwise equality,
strides, diagnostics, deterministic inputs, parsing, statistics, JSON/CSV escaping,
output protection and Git metadata. GPU tests separately cover partial tiles,
padded/non-square matrices, numerical agreement, timings and validation, plus
explicit arithmetic host models, repeated reordered results, sensitive inputs and
artifact replay. CPU tests also check aggregate/special-value statistics and
instruction-verifier rejection paths.
GitHub Actions tests GCC/Clang Debug/Release CPU builds and has a separate CUDA
container job that compiles kernels and runs CPU tests without assuming a GPU.
