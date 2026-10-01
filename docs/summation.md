# Stable summation and error attribution

This workflow compares accumulation methods using the existing vector fixtures,
FP32 diagnostics, native result captures, and provenance. It studies accuracy,
not performance or a new CUDA implementation. Existing matmul, reproducibility,
benchmark, crossover, reporting and roofline commands remain supported.

## Methods and arithmetic

| Method | Implementation |
|---|---|
| `fp32_forward` | Existing `cpu` left-to-right FP32 loop, starting at +0. |
| `fp32_reverse` | Existing `cpu-reverse` loop in decreasing index order. |
| `fp32_pairwise` | Adjacent FP32 pairs combined level by level; an odd tail is carried unchanged. Works for arbitrary positive lengths and uses O(N) scratch storage. |
| `kahan_fp32` | FP32 sum and compensation: `y=x-c; t=sum+y; c=(t-sum)-y; sum=t`. |
| `neumaier_fp32` | FP32 sum and compensation, with correction chosen according to the larger magnitude operand; returns `sum+c`. |
| `fp64_accumulation` | Existing serial FP64 analysis accumulation, cast once to FP32 at the output boundary. The uncast accumulator is also recorded. |
| `cuda-tree` | Existing configurable 64/128/256/512-thread tree; unchanged by this milestone. |

Pairwise limits the depth of addition chains. Compensation attempts to retain
rounding information that ordinary accumulation discards; neither method is a
universal cure for cancellation, overflow or finite precision. For example,
Kahan can return 0 for `[2^27, 1, -2^27]`, while Neumaier returns 1. On the existing
257-term cancellation fixture, Kahan returns 256 while Neumaier and FP64 return
255. These are tested behaviors, not reasons to silently change the fixture.

The new pairwise/compensated implementation is isolated in `src/summation.cpp`.
GCC/Clang compile that file with `-fno-fast-math -ffp-contract=off`; MSVC uses
`/fp:strict`. The override is recorded in metadata and prevents algebraic
reassociation from erasing compensation. Existing forward/reverse, FP64 reference,
and CUDA compilation policies are preserved. On an unsupported compiler the
old workflows still build, but new stable methods refuse execution rather than
silently claiming a verified compilation policy. No process rounding/flush mode
is changed; experiments assume the usual default floating-point environment.

For dot products, pairwise/Kahan/Neumaier form **separately rounded FP32 products**
before accumulating them. Compensation cannot recover product-rounding error.
Forward/reverse retain their original compiler-dependent FMA contraction. The
FP64 reference promotes both FP32 operands before multiplying. Thus dot errors
can include multiplication precision and contraction differences as well as
summation error; this is not a contraction-only experiment. The controlled
matmul FMA workflow remains available for that separate question.

## Two reference targets

Every native vector capture now records:

1. `fp64_reference`: increasing-index FP64 accumulation of the inputs (promoted
   products for dot). **This is an analysis reference, not a mathematically exact
   sum.** FP64 can lose information too.
2. `fp64_rounded_to_fp32`: that reference cast to FP32, with its exact bits.

Every method returns an FP32 scalar through the existing output contract. For
`fp64_accumulation`, `fp64_accumulator_result` additionally records the uncast
FP64 value. Therefore its FP32 result matches the rounded target by construction,
but its absolute error versus FP64 may be nonzero. Its match rate is a validation
check, not proof that it is universally best or exact.

## Metrics and improvement rules

Let `R` be the FP64 reference, `x` the method's FP32 output, and `f` the existing
forward FP32 output. Calculations below use double precision:

- Signed error: `double(x)-R`.
- Absolute error: `abs(double(x)-R)`.
- Relative error: absolute error / `abs(R)`, omitted when `abs(R)<=1e-12` or values
  are nonfinite. This is separate from the legacy FP32 zero-reference convention.
- `ulp_to_rounded_target`: native ULP distance between x and rounded FP64 target,
  available for finite pairs. `ulp_vs_baseline` still refers to the chosen FP32
  compare reference; these columns answer different questions.
- `rounded_target_bitwise_match`: exact FP32 bit-pattern equality, including the
  sign of zero. It is not a tolerance check or a mathematical exactness claim.
- Absolute-error reduction: `abs(double(f)-R) - abs(double(x)-R)`. Positive is an
  improvement; negative is worse. This always uses **fp32_forward**, even when
  the native CLI's selected comparison reference is another implementation.
- Error-reduction factor: baseline absolute error / method absolute error.

| Baseline error | Method error | Factor | `factor_status` |
|---:|---:|---:|---|
| positive | positive | ratio | `finite` |
| positive | zero | infinity (`inf` in CSV) | `eliminated_error` |
| zero | zero | 1 by convention | `both_zero` |
| zero | positive | 0 | `zero_baseline` |
| unavailable | any | blank | `unavailable` |

Classification prioritizes `exact_match` when the finite result has the rounded
target's bits. Otherwise smaller error is `improved`, larger error `worsened`,
and equal error `no_improvement`. Nonfinite/undefined error comparisons use
`unavailable`. A target match can have no improvement over an already optimal
baseline; reports separately count cases without positive error reduction.
Signed-zero results can have zero ULP/error while differing in bits.

FP64 errors and improvement metrics are left blank for nonfinite pairs. Target
bit equality is still recorded, but nonfinite results/targets are excluded from
match-rate denominators. The compensated implementations reset compensation
when the tentative sum becomes nonfinite to avoid manufacturing an `inf-inf`
correction for a single infinite input; opposite infinities still produce NaN.
This is not overflow recovery, and the normal study fixtures remain finite.

## Running a study

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
python3 -m pip install -r scripts/requirements-report.txt

python3 scripts/summation_analysis.py --executable build/matmul-inspector \
  --operation reduction_sum --sizes 257,1024,100003 \
  --fixtures random_uniform,cancellation,large_dynamic_range \
  --cuda --block-sizes 64,128,256,512 --seed 42 \
  --output results/summation-study
```

Omit `--cuda` for CPU-only analysis. Use `--operation dot` for the same machinery
with existing dot inputs. `--methods kahan_fp32,neumaier_fp32` selects methods;
the forward baseline is always included. When CUDA is requested, pairwise,
Kahan, Neumaier and FP64 target captures are automatically included as required
comparison references and recorded in `config.methods` separately from the
requested list. Fixtures are reused unchanged, including ascending/descending
magnitudes, alternating signs, cancellation and repeated-small-plus-large.
Structured dot fixtures still use B=1; random dot uses both recorded seeds.

`--no-plots` needs only Python's standard library. Output directories must be
new. Missing CUDA hardware/runtime produces recorded skips; execution errors
are not treated as numerical disagreement. A native tolerance failure is valid
study data and does not stop collection. The sweep does not widen tolerances.

Individual methods also work with ordinary compare/benchmark and replay:

```bash
build/matmul-inspector compare --operation reduction_sum --size 257 \
  --input cancellation --reference fp32_forward --candidate neumaier_fp32 \
  --save-output --output results/neumaier-case
python3 scripts/reproduce.py results/neumaier-case/metadata.json \
  --executable build/matmul-inspector --output results/neumaier-replay
```

Existing `cpu`/`cpu-reverse` names remain valid aliases. All numerical analysis,
including recomputation of the forward baseline, happens outside benchmark
sample timers. Nominal FLOP models describe mathematical work; they do not count
compensation operations or scratch storage. GPU roofline analysis continues to
exclude CPU rows.

## Artifacts and report

- Native `scalar_results.csv` receives additive columns for forward result,
  rounded target and bits, signed error, target ULP, target match, baseline error,
  error reduction/factor/status/classification, and optional FP64 accumulator.
  Native schema remains 3; `operation_extension` is now `"3"`.
- `summation_results.csv` retains native scalar fields plus method, size, source
  directory, commit/dirty state, compiler, build and hardware identity.
- `summation_summary.csv` contains overall and per-fixture method statistics:
  finite target-match rates, largest errors/reductions, finite factors, eliminated
  error counts, zero-baseline cases and cases without positive improvement.
- `summation_cuda_comparisons.csv` compares each CUDA block against forward,
  pairwise, Kahan, Neumaier and `fp64_rounded_to_fp32`, using the unchanged native
  binary comparator and existing diagnostic semantics.
- `summation_metadata.json` records configuration, full source-run provenance,
  executable/helper/artifact digests and summary statistics. Child `runs/`
  directories preserve replayable native captures and compact scalar binaries.

The report plots absolute error versus length, finite error-reduction factors
(infinite factors are explicitly counted rather than plotted at an invented
finite height), and bitwise match rates. It summarizes worst naive/CUDA errors,
compensation improvements and failures to help, pairwise/CUDA agreement, and
fixture dependence. These are observations for the recorded hardware, compiler,
inputs and sizes—not universal rankings, monotonic-error claims or performance
measurements. Dirty-source warnings remain prominent.
