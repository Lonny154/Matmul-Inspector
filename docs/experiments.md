# Experiment format and reproducibility

An experiment is a configuration plus a particular executable, source state,
machine and timing method. Matching input seeds does not promise matching elapsed
times or floating-point bits across toolchains. The C++17 code and artifacts have
no third-party dependencies; Python 3 is used for replay, integration tests and controlled-kernel PTX verification.

## Artifacts (schema version 2; replay also accepts version 1)

`--output PATH` reserves a **new** directory. Existing paths are refused, even if
empty. Each file is written through a temporary file; `metadata.json` is written
last as the completion marker. Files left without metadata after interruption are
incomplete. Disk errors are reported, never silently ignored.

* `metadata.json`: UTC timestamp, build/run provenance, hardware/software metadata,
  timing methodology, status and the complete configuration. Unknown optional
  values are the string `"unknown"`. Known `git_dirty`, `build_git_dirty`,
  `runtime_git_dirty`, and fast-math fields are JSON booleans. `config` records
  M/N/K, float32/packed row-major layout, kernel names, tile size, both seeds,
  generator version, actual warmups/iterations (zero in compare mode), atol/rtol.
  Schema 2 also records input mode, sample cap, reference/candidate contraction and
  accumulation modes. `fp_verified` is a boolean for the controlled kernels in the
  inspected library (not a claim about ordinary kernels). Verification method,
  instruction counts and library SHA-256 accompany it. `mismatches_truncated` is
  true if any row's sample was capped, even with cap zero.
* `summary.csv`: one row per configuration/kernel. Compare mode emits the candidate
  row with empty timing cells. Benchmark emits selected reference and candidate rows. Columns are
  M/N/K, kernel, tile size (zero for naive/CPU), mean/median/minimum/population
  standard deviation in milliseconds, GFLOP/s, reference kernel, speedup, bitwise
  equality, divergent count and tolerance pass. Schema 2 appends contraction and
  accumulation modes for each kernel, divergent percentage, max/mean divergent ULP,
  max absolute/relative error, tolerance-failure count/percentage, ULP bins, finite
  and nonfinite pair counts, zero-reference count, and sample count/truncation.
  Undefined timing rates remain empty.
* `mismatches.csv`: present when at least one sample is saved. Records up to
  `--max-mismatches` row-major pairs per configuration (default 100; range 0–1000)
  that differ bitwise or fail tolerance. `kind` marks first bitwise/numeric points
  when they occur within the cap; all other rows are `sample`. First diagnostics
  remain in console output even beyond the sample cap. Records include logical
  coordinates, both values and 32-character IEEE-754 bits, ULP distance, errors,
  effective tolerance and pass/fail. Bit strings are authoritative for special
  values; CSV may contain `nan`/`inf` in value/error columns.
* `console.txt`: captured stdout and stderr for the experiment, including warnings
  and failures. No capture or output occurs inside kernels.

Statuses are `complete`, `tolerance_failed`, `skipped`, or `failed`. `complete`
means all requested work finished and tolerance passed, not that bits necessarily
matched. Failed runs may contain a partial summary. Do not publish a result merely
because its directory exists. Exit codes: 0 success, 1 execution/artifact/tolerance
failure, 2 invalid CLI arguments, 77 CUDA unavailable. The legacy benchmark alias
preserves its original exit-0 skip behavior.

## Deterministic inputs

`lcg32-v1` preserves the original benchmark generator. For each logical element in
row-major order, using uint32 wraparound:

```
state = 1664525 * state + 1013904223
value = float(int((state >> 16) % 101) - 50) / 100.0f
```

A starts at `--seed` (default 42); B starts at `--seed-b`, defaulting to
`seed + 81` modulo 2^32 (123 for the default seed). Each configuration restarts
these independent generators. Padding is untouched. Reproduce the generator
version, shape, seeds and floating-point compilation settings together.

## Provenance and build settings

The build refreshes Git commit and dirty status whenever `cmake --build` runs.
`git_commit` refers to the source at build time. Runtime probing uses the original
source directory recorded in the binary, not the caller's working directory, and
records `runtime_git_commit`/`runtime_git_dirty` separately. Untracked nonignored
files count as dirty. `git_dirty` is true if either build or runtime was dirty;
unknown provenance stays unknown. The CLI warns prominently for dirty/unknown
sources or differing commits. There is no strict-publication mode yet.

Dirty source cannot be recovered from its commit hash. Commit or separately
archive source before publishing measurements; this repository does not archive
patches or binaries automatically. Copying an old executable to a new source tree
does not update its embedded provenance. Missing Git/source checkout is allowed.

Metadata records build type, CMake version, C++ compiler/version/path and CMake
flags, CUDA compiler/version/path/flags and architecture selection. C++ fast-math
uses GCC/Clang's `__FAST_MATH__` indicator; NVCC fast-math detects the standard
`use_fast_math` flag in recorded CUDA flags. Other compiler families report unknown.
Raw flags are authoritative: arbitrary
toolchain wrappers, response files, per-function pragmas, independently overridden
FTZ/FMA controls, or MSVC `/fp` modes cannot all be inferred from these booleans.
Keep `compile_commands.json` with published experiments when custom toolchains
are involved. Warning switches do not change floating-point semantics.

Hardware metadata includes CPU model and OS when available, GPU name/compute
capability, CUDA runtime and driver API versions, and NVIDIA driver package version
from `nvidia-smi`. CUDA API versions use NVIDIA's integer format; the driver API
version is **not** the driver package version. Probes are best-effort and run
outside measured intervals. CPU-only runs leave GPU fields unknown.

## Timing and interpretation

CUDA events bracket individual kernel launches on the default stream. Each kernel
gets configured warmups, then iterations; the reference is measured before the candidate. Device
allocation, zeroing, transfers, warmups, host comparisons, statistics and file I/O
are excluded. Statistics use individual event elapsed times and population standard
deviation; raw samples are not currently saved. GFLOP/s = `2*M*N*K / (mean_ms*1e6)`;
speedup = reference mean / candidate mean (the reference row reports 1). Both kernels overwrite C each iteration.

Measurements are affected by clock state, contention, temperature, driver scheduling
and warm caches. No clock locking or cache flushing is performed. Tiny kernels are
dominated by launch/event scheduling. Debug builds are useful for testing, not for
performance claims. Do not compare hardware as if these measurements normalized its
capabilities. See [NVIDIA's event timing guidance](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).

The README's original milestone table predates this schema and has incomplete
provenance and no per-sample statistics. The checked-in example is a fresh run
captured by the experiment infrastructure, not metadata retrofitted to that table.
`scripts/reproduce.py` replays its input/configuration into a fresh result directory;
it does not recreate a dirty source tree, compiler, driver or hardware.

## Controlled arithmetic schema semantics

`compiler_default` labels the original CPU, naive and tiled kernels. Controlled
contraction modes are `explicit_fma_rn` and `separate_rn_mul_add`; accumulation is
`increasing_k` or `even_odd_partials`. The tiled label is
`increasing_k_zero_padded_tiles`. No contraction flag is silently applied globally.
The selected settings and raw compiler flags must be interpreted together.

ULP bins count only bitwise-divergent finite pairs: 0 (signed zeros), 1, 2, 3–4,
5–8, >8. Their sum is `finite_divergent_count`, the denominator for mean ULP.
Max ULP/error values consider finite pairs only; empty populations report zero.
NaN-containing and other infinity-containing pairs are counted separately.
Divergent/tolerance percentages use all logical output elements as denominator.
NaN fails tolerance even if its payload bits match; equal infinities pass and
unequal infinities fail. This corrects the previous unequal-infinity tolerance
corner case without changing finite tolerance arithmetic.

Errors retain the scalar helpers' FP32 arithmetic; extreme finite differences
can overflow to `inf` in CSV. Relative error is zero for an exactly zero reference;
`zero_reference_nonzero` makes that reporting convention explicit. Very small
nonzero references can yield large relative errors and ULP distances even when
absolute error passes. See [the controlled experiment report](floating_point_experiments.md)
for the versioned cancellation and FMA-sensitive generators and actual measurements.
