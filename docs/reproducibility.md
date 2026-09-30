# Reduction reproducibility experiments

This suite studies **numerical behavior**, independently of performance/roofline
claims. `reduction_sum` is the primary operation; `dot` uses the same fixtures
and reduction machinery. Existing matmul, benchmark and crossover workflows
remain available.

## Orders and references

- `cpu`: FP32 serial, left to right, starting from positive zero.
- `cpu-reverse`: FP32 serial, right to left, starting from positive zero.
- `cuda-tree`: one term per thread; shared-memory adds at offsets B/2, B/4, …, 1.
  Each block writes one partial; later launches repeat the same tree on partials.
  Missing lanes load positive zero. Every lane reaches every barrier. There are
  no atomics or scheduling-dependent updates. B is 64, 128, 256 (default), or 512;
  both the allowed set and the device's thread/shared-memory limits are checked.

A fixed tree is expected to be repeatable, but **repeat determinism is measured**.
Different block sizes partition the same terms differently and can change the
answer even when every configuration repeats identical bits. Floating-point
addition is not associative: intermediate rounding in `(a+b)+c` can differ from
`a+(b+c)`. For example, FP32 `2^24 + 1` rounds back to `2^24`; later cancellation
cannot recover the lost one.

The chosen FP32 baseline is explicit (`--reference`; sweep default `cpu`). ULPs
measure representable FP32 steps to that baseline; they do not measure distance
to FP64. Large ULP counts near cancellation can coexist with small absolute error.
Bitwise equality, numerical tolerance, and repeat determinism answer different
questions. Tolerances keep the existing symmetric formula
`abs(candidate-reference) <= max(atol, rtol*max(abs(candidate),abs(reference)))`.

The additional analysis reference accumulates serially in **FP64**, and dot
promotes both FP32 operands before multiplying. It is **not mathematically exact**
or an error guarantee. Absolute error vs FP64 is computed in double. Relative
error vs FP64 is omitted for nonfinite pairs and `abs(reference) <= 1e-12`;
that fixed threshold is recorded. Existing FP32 comparison conventions remain
unchanged, including their legacy zero-reference relative-error behavior.

CPU dot contraction follows build flags; CUDA dot uses explicit rounded FP32
multiply, then explicit rounded FP32 tree additions. Therefore a CPU/GPU dot
comparison does not isolate association from contraction. Use the existing
controlled matmul variants for a contraction-only experiment. This milestone
adds no compensated, exact, or globally reproducible summation algorithm.

## Deterministic fixtures

The fixture version is recorded in `config.generator`. Sizes may be arbitrary
positive lengths. Inputs remain finite; no fixture intentionally overflows.

| Fixture | Construction |
|---|---|
| `random` / `random_uniform` | Existing `lcg32-v1`: LCG state `1664525*state+1013904223` modulo 2^32; `((state>>16)%101-50)/100`. A uses seed; dot B uses seed_b. This is the existing discrete uniform-style fixture, not a continuous distribution. |
| `ascending_magnitude` | Positive powers of two, exponents `-20 + floor(40*i/(L-1))`; length one is 1. Endpoints and exponent bands are deterministic. |
| `descending_magnitude` | Exact reverse of ascending fixture; same multiset and length. |
| `alternating_sign` | `(-1)^i * (1+(i mod 7)/8)`, exactly representable terms. |
| `cancellation` | Repeat a 257-element segment: `2^24`, 255 ones, `-2^24`. A partial final segment is retained, so arbitrary lengths work; lengths not divisible by 257 can have a large nonzero reference. |
| `large_dynamic_range` | Advance the same LCG each element; sign from low bit, exponent `(state>>8)%41-20`; value is signed `2^exponent`. Correlated signs are deliberate, not a claim of independent random sampling. |
| `repeated_small_plus_large` | First value `2^24`, then ones. Forward/reverse/tree orders expose loss or preservation of the small terms. |

All structured dot fixtures set **B=1** and A to the listed terms. This explicitly
controls cancellation and signs in the products without expanding the fixture
system. Seeds affect random inputs and `large_dynamic_range`; other fixtures
are size-dependent only. `seed_b` is recorded but unused for structured fixtures.
Vector fixture IDs use `vector-<fixture>-v1`, distinct from the older matmul
`cancellation-v1`; random and its alias retain `lcg32-v1`.

## Commands

Use a CUDA-enabled build for GPU configurations. CPU-only runs need no GPU:

```bash
build/matmul-inspector compare --operation reduction_sum --size 257 \
  --input cancellation --reference cpu --candidate cpu-reverse \
  --repeats 3 --save-output --output results/reverse-order

build/matmul-inspector compare --operation reduction_sum --size 100003 \
  --input random_uniform --reference cuda-tree --reference-block-size 64 \
  --candidate cuda-tree --block-size 256 --repeats 10 \
  --save-output --output results/block-order

python3 scripts/reproducibility.py --executable build/matmul-inspector \
  --operation reduction_sum --sizes 257,1024,100003 \
  --fixtures random_uniform,cancellation --block-sizes 64,128,256,512 \
  --repeats 3 --seed 42 --output results/reduction-study
```

Default implementations are `cpu,cpu-reverse,cuda-tree`; select
`--implementations cpu,cpu-reverse` for a CPU-only suite. `--operation dot` reuses
the workflow. For a larger follow-up sweep choose `--sizes 1024,1000003,4194304`
explicitly. Plotting uses the existing `scripts/requirements-report.txt`; use
`--no-plots` for CSV/JSON/Markdown without matplotlib. Existing output directories
are refused. CPU serial remains the sweep baseline even if omitted from the
candidate implementation list.

Tolerances remain `atol=1e-6, rtol=1e-5` unless explicitly supplied. Native compare
returns 1 on a tolerance failure **but still captures observations**. The sweep
accepts these completed diagnostic runs, records failures, and returns success
when collection succeeds. Execution errors remain errors; unavailable CUDA is
recorded as skipped. No tolerance is widened to make an experiment pass.

## Repeats and artifacts

`--repeats` belongs to vector `compare`, not timed benchmark mode. Each candidate
is executed repeatedly in one process against the first chosen FP32 baseline.
CUDA calls allocate fresh buffers and have the existing untimed priming launch;
only the returned scalar is an observation. Repeat classifications compare each
returned result with repeat zero using native FP32 helpers:

- `bitwise_deterministic`: all bit patterns identical, including NaN payloads.
- `numerically_equal_bitwise_varying`: bits differ, every value passes tolerance
  against repeat zero.
- `numerically_varying`: some value fails that repeat-zero tolerance.
- `insufficient_repeats`: one observation; no determinism claim.

This checks same-process executions, not independent process restarts or universal
cross-hardware reproducibility. Benchmark trial fingerprint checks remain unchanged.

Native vector runs add `scalar_results.csv` with each returned FP32 scalar,
32-bit binary/hex representations, sign, finite/NaN/infinity classification,
FP32 baseline, FP64 analysis reference, errors, ULPs, tolerance status, and repeat
classification. Nonfinite FP32/FP64 values use CSV strings `nan`, `inf`, `-inf`;
unavailable numerical metrics are blank, not misleading zeros. Summary and
saved binary artifacts describe the first compare execution; later observations
are in the scalar CSV. Metadata records `operation_extension: "2"`, repeats,
reference/candidate block sizes, fixture version, analysis method and relative
threshold. Schema 3 remains additive. Saved scalar binaries retain `MIFP32LE`.

The sweep creates:

- `reproducibility_results.csv`: all native scalar observations plus source-run
  path and bitwise agreement with the FP32 baseline.
- `block_sensitivity.csv`: every pair of observed CUDA block sizes for each
  fixture/length, using **first-execution** binaries and the native comparator;
  includes whether all tested block sizes agree bitwise. The first requested
  block is the selected plot reference, never a performance-selected baseline.
- `reproducibility_metadata.json`: configuration, executable/comparator/script
  hashes, native source metadata/provenance and aggregate numerical summaries.
- `ulp_drift.png`, `absolute_error.png`, `block_sensitivity.png` (when at least
  two CUDA block configurations execute), plus `report.md`.
- `runs/…`: complete native captures including replayable metadata and scalar
  output binaries. No enormous input/output vectors are saved.

Plots show repeat zero; every repeat remains in CSV. Connected lengths do not
imply monotonic error growth. Reports separate observed repeat determinism from
agreement across orders. Timing/roofline metrics are not used to rank variants.

Replay any child with the existing workflow:

```bash
python3 scripts/reproduce.py results/reduction-study/runs/cancellation-257-cuda-tree-64/metadata.json \
  --executable build/matmul-inspector --output results/replayed-reduction
```

Replay preserves fixture, seeds, both block sizes, implementation and repeats.
Cross-hardware scalar comparison retains operation/input/accumulation checks;
different block sizes are different semantics, not controlled same-kernel runs.
Legacy vector artifacts still replay at their recorded 256-thread block size.

## Special values and boundaries

Signed zeros have distinct bits but zero ULP distance and pass tolerance. NaNs
never pass numerical tolerance, even if identical bits repeat. Equal same-sign
infinities pass tolerance; opposite infinities do not. Scalar analysis leaves
ULP and FP64 error fields unavailable for nonfinite pairs. Tree zero padding and
serial positive-zero initialization may affect signed-zero outputs; bits expose
that behavior. Main sweep fixtures target finite arithmetic; special values are
covered by unit tests.

FP64 is only an analysis reference; no exact-error bounds are provided. The tree
block size is selected from a small validated set, not tuned. CPU pairwise
reduction is not implemented; reverse serial provides a second simple CPU order.
The current same-kernel/different-block workflow uses compare mode; timed
benchmark artifacts are not redesigned for duplicate kernel names. Future
research could add compensated/exact references, independent-process repeats,
and dedicated product fixtures that separate dot contraction from association.

## Observed smoke experiment

A local Release build (GNU 13.3.0, CUDA compiler 13.3.73, no fast-math) on an
RTX 4060 Ti, compute capability 8.9, tested seed 42; lengths 257, 1024, 100003;
`random_uniform` and `cancellation`; serial, reverse, and blocks 64/128/256/512.
All **36 configurations were bitwise deterministic across three repeats**.
This is a dirty-tree, hardware/compiler-specific smoke capture, not a universal
reproducibility claim or performance result.

- Random inputs: largest difference from CPU serial was **14 ULP** (reverse,
  length 100003). Its absolute error vs FP64 was about **2.0791e-4**.
- Cancellation, length 257: CPU serial returned **0**, reverse returned **255**,
  and the FP64 analysis reference was **255**. The FP32 distance is
  **1,132,396,544 ULP**; that large count spans the exponent range from zero and
  must be interpreted alongside the values and absolute errors.
- Largest CUDA block-pair distance: **65,536 ULP**, blocks 64 versus 512 at
  cancellation length 257; the absolute difference was **1.0**.
- Largest absolute error vs FP64 across the suite was **99,224**. Default
  FP32-baseline tolerances failed for **45 of 108 observations**. Errors are
  preserved as experiment data; tolerances were not relaxed.
- A separate small dot sweep (lengths 257/1003, the same two fixtures, blocks
  64/256, two repeats) was bitwise deterministic for all **16 configurations**.

Identical repeated bits coexist here with disagreements across orders. The
cancellation cases also show that passing against a serial FP32 baseline is
not, by itself, evidence of accuracy against a higher-precision reference.
