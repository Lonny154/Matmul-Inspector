# Compiler-to-performance reporting

`scripts/compiler_performance_report.py` combines existing NVIDIA PTX comparison,
SASS comparison, and Matmul Inspector benchmark artifacts. It is a Stage 1
integration tool: it does not compile code, execute GPU benchmarks, or run a
profiler. It uses only the Python standard library and treats every input as
read-only.

## Usage

```bash
python3 scripts/compiler_performance_report.py \
  --ptx-comparison results/ptx-fma-comparison/comparison.json \
  --sass-comparison results/sass-fma-comparison/comparison.json \
  --benchmark results/fma-vs-no-fma-sass \
  --reference-map ptx=mi_naive_fma,sass=mi_naive_fma,benchmark=cuda-naive-fma \
  --candidate-map ptx=mi_naive_no_fma,sass=mi_naive_no_fma,benchmark=cuda-naive-no-fma \
  --output results/unified-fma-study
```

The output directory must not exist. Successful execution creates exactly:

```text
OUTPUT/
  experiment.json
  report.md
```

PTX-only, SASS-only, and benchmark-only reports are supported. An omitted layer is
recorded as `absent`; missing values inside supplied evidence remain `unavailable`.
Missing required benchmark files make benchmark data `incomplete`. They are not
silently treated as an empty successful run.

## Identity mapping versus compilation provenance

These are deliberately separate concepts.

`--reference-map` and `--candidate-map` state the user's intended correspondence
among the selected PTX symbol, SASS function, and benchmark implementation. When
multiple layers are supplied, every present layer must be mapped explicitly. The
tool verifies that mapped names equal the identities recorded in each input.

An explicit mapping does **not** prove that the artifacts came from the same build.
Compilation provenance is evaluated separately. A relationship is:

- `verified` only when artifacts share a matching strong build or binary identifier;
- `unverified` when no such identifier links them;
- `incompatible` when recorded strong identifiers or architecture properties conflict;
- `absent` when only one evidence layer is present.

Source commits, architecture agreement, and similar kernel names are useful
context but are not sufficient build identity. PTX/SASS text hashes are hashes of
those text inputs, not automatically hashes of the benchmark executable. The
current inspector schemas do not carry a shared executable identifier, so ordinary
separately captured PTX, SASS, and benchmark evidence is expected to remain
unverified.

## Evidence scopes

The unified JSON preserves the scopes recorded by the source tools:

- PTX raw and normalized comparisons cover full modules, while instruction and
  virtual-register statistics cover selected entries.
- SASS raw and normalized comparisons cover full files, while instruction,
  physical-register, and helper-call statistics cover selected functions.
- Benchmark rows retain their recorded timing mode. Kernel-only time is not
  relabeled as end-to-end time.

PTX virtual-register declarations and SASS physical register allocation are kept
distinct. PTX instructions are not matched one-to-one with SASS instructions.
Static counts are not dynamic execution counts.

## Benchmark states and metrics

The report independently records:

- `completion_status`, copied from benchmark metadata when available;
- `data_completeness`, based on required files (`metadata.json`, `summary.csv`);
- `correctness_status`, derived from selected candidate tolerance results;
- metric availability, with missing values retained as unavailable.

Recorded median latencies, confidence intervals, sample/trial counts, reliability
flags, bitwise divergence, tolerance failures, and error metrics are preserved.
Pairwise speedup is the recorded reference median divided by candidate median; it
is not replaced by the legacy mean-based speedup. Reported GFLOP/s is retained as
the source artifact defines it rather than relabeled as median-derived throughput.

A `tolerance_failed` run or a candidate with any failed tolerance row is prominently
reported as numerically failed and is never described as a successful optimization.
Confidence-interval overlap is descriptive and is not treated as a hypothesis test.

## JSON schema

`experiment.json` currently uses schema version 1. Top-level fields are:

- `identity_mapping`: explicit correspondence only;
- `compilation_provenance`: strong-identifier relationships and classifications;
- `evidence`: PTX, SASS, and benchmark summaries plus source hashes;
- `findings`: deterministic informational, warning, and incompatible conditions;
- `interpretation_limits`: machine-readable caveats.

JSON keys and findings are deterministically ordered, no report-generation
timestamp is inserted, and source timestamps are preserved as evidence. Identical
inputs and options produce identical JSON bytes.

## Limitations

The tool cannot establish scheduling, instruction throughput, occupancy, memory
bandwidth, stalls, dynamic execution counts, or performance causality. It does not
decode CUBINs or inspect executables itself. Verification is only as strong as the
identifiers recorded by input artifacts. A user mapping can be wrong in intent even
when all mapped names exist, so it remains visibly user-declared.

Tests are CPU-only:

```bash
python3 tests/compiler_performance_report_test.py
ctest --test-dir build -R compiler-performance-report-tests --output-on-failure
```
