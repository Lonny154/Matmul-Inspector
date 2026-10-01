# Collecting and comparing results across machines

This workflow runs an existing experiment elsewhere, collects its result directory,
and compares it against an explicitly selected baseline. It does not provision
machines, run SSH commands, rank hardware or add kernels. Cross-kernel comparisons
ask how two algorithms behave on one machine; cross-hardware comparisons here match
**the same kernel and configuration** between runs. The two ratios are separate.

## Build and validate locally

A C++17 compiler, CMake and Python 3 are sufficient for aggregation and detailed
numerical analysis. CUDA is not needed to read captured GPU outputs.

```sh
cmake -S . -B build-cross-cpu -DCMAKE_BUILD_TYPE=Release -DMATMUL_INSPECTOR_ENABLE_CUDA=OFF
cmake --build build-cross-cpu --parallel 4
ctest --test-dir build-cross-cpu -L cpu --output-on-failure

# Small CPU-only capture and replay smoke test; no performance experiment.
./build-cross-cpu/matmul-inspector compare --reference cpu --candidate cpu --sizes 4 --seed 42 --save-output --output results/cross-smoke-a
python3 scripts/reproduce.py results/cross-smoke-a/metadata.json --executable build-cross-cpu/matmul-inspector --output results/cross-smoke-b
python3 scripts/compare_hardware.py results/cross-smoke-a results/cross-smoke-b --comparator build-cross-cpu/matmul-compare-outputs --output results/cross-smoke-report
```

Use new directory names for subsequent runs: result and aggregate directories are
never overwritten. Tests use temporary directories, synthetic hardware metadata
and synthetic timings; they require no GPU and publish no performance claims.

## Baseline experiment

First commit the source changes you intend to measure. Use a **clean commit that
contains this milestone**, and build from that checkout. `git status --short`
should show no uncommitted source. A dirty build is allowed and recorded, but its
commit cannot reconstruct the measured source; the aggregator flags it as
uncontrolled and suppresses cross-run performance ratios.

```sh
git status --short
git rev-parse HEAD
cmake -S . -B build-cross-cuda -DCMAKE_BUILD_TYPE=Release -DMATMUL_INSPECTOR_ENABLE_CUDA=ON
cmake --build build-cross-cuda --parallel 4

# Untimed numerical experiment, with outputs for later element-wise comparison.
./build-cross-cuda/matmul-inspector compare --reference naive --candidate cuda-naive-fma --sizes 4,256,257,1024 --seed 42 --save-output --output results/cross-hardware/baseline-compare

# Optional performance experiment, run when ready to collect actual measurements.
./build-cross-cuda/matmul-inspector benchmark --reference naive --candidate cuda-naive-fma --sizes 4,256,257,1024 --seed 42 --warmups 3 --iterations 50 --save-output --output results/cross-hardware/baseline
```

The compare summary retains its existing candidate-row convention. Benchmark
summaries contain both reference and candidate rows, allowing same-run naive
speedups. Both outputs are hashed and optionally saved in either mode. To study
no-FMA or reordering, select those existing candidates instead. No output storage
is enabled automatically; use `--save-output` only for runs needing later detailed
cross-machine inspection. Hashing and output writing occur after kernel timing.

## Replay on a second machine

Clone the repository and copy the complete baseline result directory into
`baseline/` in that checkout. No directory name is treated as hardware identity.
Run the following from the repository root (the recorded commit must be available
in the clone):

```sh
git checkout "$(python3 -c 'import json; print(json.load(open("baseline/metadata.json"))["git_commit"])')"
cmake -S . -B build-cross-cuda -DCMAKE_BUILD_TYPE=Release -DMATMUL_INSPECTOR_ENABLE_CUDA=ON
cmake --build build-cross-cuda --parallel 4
python3 scripts/reproduce.py baseline/metadata.json --executable build-cross-cuda/matmul-inspector --output results/cross-hardware/second-machine
```

Replay preserves dimensions, both seeds, generator/fixture, kernels, tolerances,
warmups, iterations, mismatch cap and `--save-output`. It does not recreate build
flags, the compiler, CUDA architecture targets or driver. Inspect baseline metadata
and reproduce build settings where intended. Use `CMAKE_CUDA_ARCHITECTURES` if
needed; retain embedded PTX for the existing controlled-kernel verification.
Expected differences include GPU/compute capability, CPU, OS, compiler, runtime,
driver, timestamps and paths. Source, input configuration and kernel semantics
should remain the same for a controlled comparison.

Copy the **whole** second-machine result directory back to the collection machine.
Preserve binary sidecars and relative filenames. Then aggregate:

```sh
python3 scripts/compare_hardware.py --baseline results/cross-hardware/baseline results/cross-hardware/baseline results/cross-hardware/second-machine --comparator build-cross-cpu/matmul-compare-outputs --output results/cross-hardware/aggregate
```

Omit `--baseline` to use the first input directory; the tool prints and records
that selection. A baseline must be among the supplied runs. Omit `--comparator`
for a hash/performance report, or put `matmul-compare-outputs` on PATH for automatic
discovery. Differing hashes without both saved outputs and a comparator are
explicitly reported as `hash_differs_details_unavailable`, with empty metrics.
The helper always runs locally on the CPU and reuses `comparison::compare`.

## Output fingerprints and binary format

Schema 3 adds `output_sha256`, `reference_sha256`, `output_file` and
`reference_output_file` to each summary row. Metadata config declares
`output_hash: "sha256"`, `output_encoding: "fp32-le-row-major-v1"` and the boolean
`save_output`. Schemas 1 and 2 remain readable/replayable; without hashes they
support performance analysis only, rather than invented numerical agreement.

SHA-256 consumes each logical matrix element in row-major order, encoded as its
exact IEEE-754 binary32 **little-endian** four bytes. Row padding, dimensions and
file headers are not hashed. Storage is copied as bits without floating-point
arithmetic, preserving signed zeros and quiet/signaling NaN payloads. SHA-256 is
implemented locally with no new dependency and tested against Python's standard
library across block/padding boundaries. Hash equality is strong evidence of
bitwise equality for a matched shape/encoding, subject to the usual theoretical
hash-collision caveat; it does not replace numerical diagnostics.

`--save-output` requires `--output`. Each configuration writes two uniquely named
files such as `output-0-reference.bin` and `output-0-candidate.bin` with matching
`.bin.json` sidecars. The binary is:

| Offset | Size | Meaning |
| ---: | ---: | --- |
| 0 | 8 bytes | ASCII `MIFP32LE` (format v1, binary32, little-endian) |
| 8 | 8 bytes | unsigned 64-bit little-endian row count |
| 16 | 8 bytes | unsigned 64-bit little-endian column count |
| 24 | rows × cols × 4 bytes | logical FP32 elements; no padding |

No trailing bytes or compression are allowed. The sidecar records format, dtype,
byte order, rows/columns, M/N/K, kernel, both seeds, generator, contraction and
accumulation modes, and payload SHA-256. The aggregator checks the sidecar against
the source metadata/summary, validates header dimensions and exact length, and
rehashes saved payloads. Corrupt, truncated or contradictory files are rejected;
paths escaping the run directory are rejected. The C++ helper independently checks
format/length/dimensions before comparing. It does not read sidecars by itself;
context validation is the aggregator's responsibility.

Storage is approximately `4*M*N` bytes per saved output, plus small headers and
sidecars. Both reference and candidate are saved, even when identical. Saving is
optional; fingerprints are always produced by experiment workflows. The original
no-argument demo is unchanged.

## Compatibility policy

Rows are matched by exact kernel name and M/N/K. Shapes or kernels absent from the
baseline are **separate configurations**, not numerical mismatches. Duplicate
kernel/shape rows within a run are rejected as ambiguous (use distinct result
runs instead of repeated `--sizes` entries or benchmarking a kernel against itself).

| Difference | Treatment |
| --- | --- |
| Seed/seed_b, dtype, input mode/generator, tile size, contraction or accumulation | Incompatible for cross-run output comparison; no cross-run ratio |
| Commit differs, dirty source, build/runtime commit differs, or source provenance unknown | Flag `uncontrolled_source`; hashes/details remain observational for matching inputs, but cross-run timing ratio is suppressed |
| Compiler, flags, runtime, driver or compiled architecture differs | Allowed as a flagged `cross_toolchain` experiment; no claim of isolating hardware alone |
| Build type, timing methodology, reference/candidate pair, mode, warmups or iterations differs | Flag configuration difference; suppress cross-run timing ratio |
| atol/rtol differs | Hashes remain comparable; detailed comparison explicitly uses baseline tolerances; suppress cross-run timing ratio |
| Failed, skipped, incomplete run, invalid binary metadata or ambiguous rows | Reject rather than publish a misleading report |

The seed checks remain conservative even for fixtures that ignore seeds. Ordinary
kernels retain compiler-default contraction semantics, so compiler/flag changes are
independent variables that may affect bits. Matching names do not certify identical
machine instructions. Controlled FMA verification fields remain available in each
source's embedded metadata.

Hardware labels use GPU name, compute capability and CPU model from metadata.
Runtime, driver, compiler and architecture are separate environment columns.
Hostnames, usernames and directory names are not canonical hardware identifiers;
source directory paths are retained only as provenance and to distinguish runs.

## Aggregate artifacts and numerical definitions

The aggregate output directory contains:

- `cross_hardware_summary.csv`: source/baseline directories, hardware/environment,
  source commit/dirty state, kernel/shape/seed, output hash, timing statistics and
  GFLOP/s, compatibility and warnings. `within_run_naive_speedup` is that run's
  ordinary naive mean divided by its kernel mean, if both exist.
  `baseline_performance_ratio` is baseline mean divided by this run's mean for the
  same eligible configuration. A value above one means shorter observed time in
  this run; it is not a ranking or explanation. Undefined ratios are empty.
- `cross_hardware_numerics.csv`: hash comparison status, baseline tolerances,
  divergent count/percentage, max and mean divergent ULP, maximum absolute/relative
  error, tolerance-failure count/percentage, nonfinite counts, ULP bins and first
  bitwise/tolerance divergence. Structured cells use JSON. Unavailable diagnostics
  are empty, never assumed zero.
- `cross_hardware_metadata.json`: aggregate schema version, timestamp, selected
  baseline and selection rule, original metadata for every source directory,
  SHA-256 of source metadata/summary files, aggregator/comparator hashes and ratio
  and tolerance conventions. It is written last as the completion marker.

When binaries and the helper are available, full diagnostics run even for matching
hashes: identical NaN payloads match bitwise but still fail numerical tolerance.
Without binaries, matching hashes establish only bitwise agreement, not tolerance
success. Full diagnostics use the existing C++ definitions, including signed-zero
ULP=0, nonfinite exclusions from ULP/error aggregates, the sign-boundary key gap,
and the legacy relative-error-zero convention for an exactly zero reference.
See [numerical conventions](experiments.md#controlled-arithmetic-schema-semantics).
The helper emits strict JSON; nonfinite error/value fields are strings `nan`,
`inf` or `-inf`. A divergence is data: successful aggregation exits 0 even for
tolerance failures; malformed/incompatible binary artifacts or CLI errors exit 2.
Configuration incompatibility is reported in rows rather than treated as a crash.

## Interpretation and limits

Clock state, thermals, power limits, driver/runtime versions, compiler flags,
background load, GPU architecture and memory subsystems all affect elapsed time.
A single performance difference does not identify its cause. Kernel order and warm
caches also matter; no clock control, thermal normalization or statistical causal
analysis is performed. Per-run statistics are preserved, not pooled into a
hardware score. No new performance results were generated for this milestone.

All binaries are optional and uncompressed. Detailed comparison loads two matrices
into host memory; hashing/capture adds host work outside timed execution. A missing
helper or missing saved outputs limits the report to hashes/performance. Hashes
and metadata are integrity checks, not signatures proving where an artifact came
from. Source snapshots, executables and remote environments are not automatically
archived or recreated. Keep original run directories and use clean commits.

## Fresh-process reproducibility captures

`scripts/process_reproducibility.py` captures **one configuration** in at least two
fresh native executable launches (default three). It supports `reduction_sum`,
`dot`, square or rectangular `matmul`, all existing compatible methods, fixtures,
seeds, tolerances and CUDA block sizes. Repeat the command for other configurations.
No new kernels or timing methodology are introduced.

These are different questions:

- Same-process determinism: existing native vector `--repeats` and
  `scripts/reproducibility.py` repeat execution within one process.
- Process-level reproducibility: this runner starts a new native process for each
  observation, preserving inputs/configuration with the existing replay builder.
- Cross-hardware reproducibility: collect complete captures independently, then
  compare compatible captures offline. Two captures on one machine validate the
  workflow but provide no evidence about another GPU.
- Numerical agreement: the existing native absolute/relative tolerance can pass
  even when bits differ. FP64 remains an approximate analysis reference, not proof
  of an exact mathematical result.

### Machine A

Build from the intended commit, then create a portable capture:

```sh
python3 scripts/process_reproducibility.py \
  --executable build-cross-cuda/matmul-inspector \
  --operation reduction_sum --size 257 --fixture cancellation \
  --method cuda-tree --block-size 256 --seed 42 --process-repeats 3 \
  --output results/process-machine-a
```

CPU methods need no GPU. For example, use `--method neumaier_fp32`,
`kahan_fp32`, `fp32_pairwise`, or `fp32_forward`. Dot accepts these same methods.
For rectangular matrices use `--operation matmul --m 5 --n 7 --k 17 --fixture random
--method tiled` instead of vector flags. `--size 17` selects a square matrix for
matmul. Choose a fresh output directory for every capture; overwriting is refused.

The runner uses native compare mode with a CPU reference, one selected candidate,
and one observation per launch. Only the selected candidate's saved output is used
for process-level comparisons. A candidate disagreeing with the CPU reference is
valid data, not a failed capture. Exit 77 records an unavailable CUDA runtime/device
as skipped; failed/incomplete/skipped captures are not eligible for comparison.
The script returns 0 on successful collection even when output varies; inspect the
classification and tolerance fields. Invalid inputs/captures return 2.

### Machine B and offline comparison

Copy the entire Machine A directory, including `runs/`, to the collection machine.
On Machine B, build the same commit/configuration where possible, and run:

```sh
python3 scripts/process_reproducibility.py \
  --executable build-cross-cuda/matmul-inspector \
  --operation reduction_sum --size 257 --fixture cancellation \
  --method cuda-tree --block-size 256 --seed 42 --process-repeats 3 \
  --output results/process-machine-b
```

Copy the complete Machine B capture back. No network or simultaneous access is
needed by these tools. Run the comparison locally (the comparator is CPU-only):

```sh
python3 scripts/compare_hardware.py --process-captures \
  results/process-machine-a results/process-machine-b \
  --baseline results/process-machine-a \
  --comparator build-cross-cpu/matmul-compare-outputs \
  --output results/process-comparison
```

The first input is the baseline if omitted. Every captured launch is compared to
baseline launch zero, including baseline's other launches. Process classification
for each capture is reported separately. Configurations must agree on operation,
shape/length, exact kernel name, fixture/generator, both seeds, dtype/layout,
encoding, contraction/accumulation, relevant tree block/stages, and tolerances.
Incompatibilities **reject the comparison before writing a report**; unlike legacy
run aggregation, they are not simply marked as separate rows. Process repeat
counts may differ. Hardware, compiler, runtime, driver and executable hashes may
differ; they are retained and flagged. Dirty/different/unknown source is explicitly
uncontrolled. A difference alone does not show whether hardware or compiler caused
it. Matching method names do not verify identical machine instructions.

### Capture contents and interpretation

- `process_reproducibility.csv`: repeat index, UUID, observed PID, timestamp,
  operation/configuration, exact output SHA-256, scalar bits/value when the output
  is 1×1, bitwise/tolerance agreement and all native metrics versus launch zero.
- `process_pairs.csv`: every earlier/later process pair, with native diagnostics.
  Tolerance is directional, using the earlier output as reference and recorded
  tolerances. All pairs are checked; tolerance is not assumed transitive.
- `process_reproducibility_metadata.json`: completion status, capture UUID,
  executable/comparator/script/helper hashes, full native software/hardware/Git
  provenance, per-launch UUID/PID/timestamps, commands, relative run paths and
  artifact digests. Commands may contain original paths for audit; loading a moved
  capture resolves only paths relative to the capture. PIDs can be reused by the OS;
  UUIDs distinguish launches. These records are integrity checks, not signatures.
- `runs/process-0000/`, etc.: unchanged native metadata, summary, console,
  mismatches/scalar diagnostics where present, binaries and sidecars. Each child
  can be replayed with `scripts/reproduce.py runs/process-0000/metadata.json ...`.
- `report.md`: concise process classification and largest observed differences.

Classification is `bitwise_stable` if every pair matches bits,
`numerically_stable_but_not_bitwise` if all pair tolerances pass but some bits
vary, otherwise `varying`. Identical NaN payloads are bitwise stable while failing
numerical tolerance; both fields remain visible. Signed zeros differ bitwise but
have zero ULP distance and pass tolerance. All diagnostics come from the existing
C++ comparator, even for matching hashes. Nonfinite aggregate exclusions and the
existing zero-reference relative-error convention remain unchanged.

Offline comparison writes `cross_hardware_comparison.csv`,
`cross_hardware_metadata.json`, and `report.md`. They retain hashes, hardware/build
identities, compatibility warnings, first divergence, ULP/absolute/relative metrics,
and tolerance results. Native binary validation checks headers, sidecars, payload
hashes, dimensions and path containment before comparison. No alternative binary
format or Python floating-point comparison is introduced.

This is untimed numerical analysis, not a performance study. Independent processes
still share the same driver, caches, clocks and machine environment: this does not
reset hardware or guarantee future reproducibility. Native comparison can require
memory proportional to saved output size, and all-pair process checks cost O(R²)
comparisons for R repeats. Use small repeat counts and intentional matrix sizes.
The capture archives outputs/provenance, not executables, source snapshots or full
environments. FP64 references and compensated-method limitations are described in
[summation.md](summation.md).
