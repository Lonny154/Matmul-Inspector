# MI-Research-3C — Explicit reference and Split-K graph crossover

## Plan recorded before new GPU execution

MI-Research-3B found that the default FP16 cuBLAS reference diverged from
the same mathematical GEMM at M=256/512 under the enabled reduced-precision
reduction setting. Conventional and Split-2 Triton outputs agreed under the
established tolerance with an FP32 matmul of the exact FP16 operand values,
rounded once to FP16. Selected FP64 dot products supported that interpretation.
The older MI-Research-3 gate and measurements remain unchanged.

For this study, the **numerical target** is the FP32 matmul of FP16 inputs
with TF32 disabled, rounded once to FP16. It approximates the mathematical
sum more closely than the default FP16 cuBLAS result in the MI-Research-3B
counterexamples. Selected FP64 dots remain diagnostics for individual
coordinates, not a whole-matrix gate. The established
`atol=rtol=0.01` remains unchanged. Validate A and B independently on
seeds 0, 1, and 2 at M=32,128,256,512 before timing either new shape.

Keep three other outputs distinct. Default PyTorch/cuBLAS FP16 output is a
historical production-library observation and is **not** the new gate.
PyTorch/cuBLAS FP16 with `allow_fp16_reduced_precision_reduction=False` is
the matched graph performance anchor C. It must also pass the explicit
numerical target. The switch does not specify cuBLAS's exact algorithm.
Record default and active precision flags. Scope all flag changes and
restore them after each process. Do not change historical evidence or the
production experiment CLI's default reference.

The new graph timing budget is two fresh processes each for M=256 and
M=512, with K=4096 and N=2048. Reuse the MI-Research-3 complete-operation
graph runner with an explicit opt-in reference policy. Keep 100 complete
operations per replay, 30 warmups, eight balanced A/B rounds, eight A/A
rounds, the three clocks, PTX, profiler graph topology, and separate A/C
and B/C phases. A uses the same 64×128×32 conventional tile. B uses the
same Split-2 source, fixed FP32 scratch, and deterministic reduction.
No tile, kernel, tolerance, or decision-rule change is planned.

M=32 and M=128 retain their two historical A/B sessions as reference
observations because A and B executable behavior and the timing method
are unchanged. Their historical cuBLAS C timing used a different precision
setting and cannot be pooled with the new C anchor. Do not repeat those
shapes merely to make the result look cleaner. If the new correctness
preflight fails, stop before timing. A reproducible crossover requires
supported A/B decisions on both sides of a boundary in independent
sessions; otherwise report the narrower evidence. Configured CTA count
is not measured occupancy.

## Numerical contract and implementation

The intended operation is the GEMM sum of the exact FP16 input values,
accumulated in FP32 and rounded once to FP16. The executable gate uses
`torch.matmul(a.float(), b.float()).half()` with TF32 disabled and
float32 matmul precision set to `highest`. It is an operational FP32
approximation to that sum, not an assertion that PyTorch's FP32 reduction
has a specified bitwise order. Selected FP64 dot products at high-error
coordinates provide a separate diagnostic. They are retained in the
correctness preflight raw records.

The gate still uses `atol=rtol=0.01`. A and B are checked independently.
The graph C anchor is `torch.mm(a, b, out=c)` with
`allow_fp16_reduced_precision_reduction=False`; C is checked against the
same mathematical target. Disabling the flag does **not** select or prove
any particular cuBLAS algorithm. The ambient/default FP16 cuBLAS output
is recorded only as a diagnostic. Its earlier failures are preserved in
[MI-Research-3](mi_research_3.md) and
[MI-Research-3B](mi_research_3b.md).

The new `--reference-policy fp32_rounded_fp16` option applies only to
the shape-study graph runner. The old command keeps the original cuBLAS
reference and schema 1. The opt-in output uses schema 2 and records the
original and active precision flags. The runner restores the flags on
exit, including error exits. The correctness preflight records that the
settings were restored. Source snapshots and hashes accompany every run.
No kernel, tile, tolerance, benchmark count, or decision rule changed.

## Correctness results

The preflight ran all 12 shape/seed cases before any new timing. A, B,
and C passed every case. The table gives failing elements for the ambient
FP16 cuBLAS diagnostic across seeds 0/1/2. A/B/C each had zero failures.
The raw file also includes maximum and mean absolute error, relative
error for reference magnitudes at least one, representative coordinates,
NaN/inf checks, and selected FP64 dots.

| M | A/B/C gate | Ambient FP16 cuBLAS failures, seeds 0/1/2 |
| ---: | --- | --- |
| 32 | 3/3 each | 0 / 0 / 0 |
| 128 | 3/3 each | 0 / 0 / 0 |
| 256 | 3/3 each | 1225 / 1213 / 1259 |
| 512 | 3/3 each | 809 / 813 / 802 |

Mean absolute error against the FP32-rounded target was about
0.00032–0.00034 for A/C and 0.00016–0.00017 for B across these cases.
The ambient FP16 cuBLAS mean error rose to about 0.0115 at M=256 and
0.0064 at M=512. A few larger absolute errors in A/B/C occur at large
output magnitudes and still pass the unchanged mixed tolerance.
The selected FP64 checks are diagnostics, not proof of whole-matrix
correct rounding.

## Graph experiment and observations

Each new session was a fresh Python process. The order was M256 session 1,
M512 session 1, M512 session 2, M256 session 2. Each shape had two
sessions with 30 warmups, 100 complete operations per graph replay,
eight paired rounds per pair, balanced A/B order, and eight A/A rounds.
A graph has one conventional kernel per operation. B has one partial GEMM
and one deterministic reduction per operation. Profiler traces show 100
A nodes, 200 B nodes, and 100 C nodes in each replay. B reuses fixed FP32
scratch: 4 MiB at M256 and 8 MiB at M512. Capture setup times are retained
separately from replay timings. These are fixed-address graph results,
not eager or changing-address results.

CUDA-event values below are microseconds per **complete operation**. The
paired difference is A minus B, so a negative value points toward A.
The A/A column is the largest absolute same-kernel paired gap in that
session. It describes observed variability; it is not a statistical
confidence interval.

| M | Session | A median | B median | Paired A−B median | Paired A−B range | A/A max | A-faster rounds | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 1 | 109.29 | 112.31 | -3.24 | -14.05 to 9.43 | 14.36 | 5/8 | Inconclusive |
| 256 | 2 | 110.78 | 114.02 | -1.19 | -8.29 to 3.11 | 10.38 | 6/8 | Inconclusive |
| 512 | 1 | 210.59 | 219.86 | -12.94 | -14.74 to -3.73 | 8.88 | 8/8 | Inconclusive |
| 512 | 2 | 209.94 | 217.61 | -7.77 | -14.80 to -0.11 | 7.94 | 8/8 | Inconclusive |

The existing controller rule requires **every** paired gap to exceed the
observed A/A maximum before ranking a pair. That condition fails in all
four sessions, including the directional M512 results. We did not adjust
that rule after seeing the data. `perf_counter` and MONOTONIC_RAW results
are retained with every sample. The host-clock preflights reported no
resolved rate difference. The clock-aware decision reports no event/wall
or RAW boundary contradiction for these comparisons. This does not turn
the small paired differences into supported performance wins.

The separately retained historical M32 sessions favored B: A/B event
medians were 61.76/30.27 and 60.13/30.57 µs with 8/8 B-faster rounds
in both sessions. The historical M128 sessions were inconclusive:
A/B medians were 59.48/60.06 and 57.59/57.89 µs. A/B source and tile
settings are unchanged, and this study's new preflight checks those
shapes against the explicit target. Still, the older sessions had a
different ambient cuBLAS precision setting and were not rerun in the
same sequence. They are reference observations, not pooled samples.

Matched C comparisons were captured under the no-reduced-precision
configuration. All A/C and B/C decisions at M256 and M512 are also
inconclusive under the existing A/A rule. C medians ranged from
104.82–106.60 µs at M256 and 211.33–213.22 µs at M512 across the
paired phases. This is a separate production-library anchor, not an
optimization claim about changing precision. The default cuBLAS mode
at M256/M512 is numerically ineligible under this study's target.

Configured conventional output CTAs are 16, 32, 64, and 128 for
M32/128/256/512. Split-2 doubles the partial-GEMM CTA count, then pays
for scratch traffic and reduction. The new compiler records contain
PTX and launch metadata for all three Triton kernels. At M256/M512,
conventional and partial GEMM each use 24,576 bytes of shared memory;
their register counts are 121 and 124 per thread. The reduction uses
22 registers and zero shared bytes. The PTX hashes vary by shape and
are stored in each raw record. PTX and configured CTA counts do not
establish occupancy or machine-code behavior.

Environment: RTX 4060 Ti, 34 SMs, compute capability 8.9, driver 616.92,
CUDA runtime 12.8, PyTorch 2.11.0+cu128, Triton 3.6.0, Python 3.13.11.
The evidence records full Git HEAD `a7f605383449ba5de9f6cddb35e720c34408e472`,
exact source/script/spec hashes, raw timing samples, graph traces, PTX,
and per-directory manifests.

## Decision and limits

**Decision: correctness established; performance crossover unproven.**
Split-2 retains a supported graph-mode advantage at M32 from the prior
study. The new M256/M512 data have lower conventional medians, but the
established paired variability gate remains inconclusive. M128 is also
inconclusive. Thus the tested shapes show a directional pattern but do
not establish a supported crossover interval or justify an automatic
strategy selector. A bounded hardware-specific recommendation is to
consider Split-2 only for the tested M32 fixed-address graph case. For
larger M, choose by the requirements of a new workload and new matched
measurements. No eager-mode, other precision, other GPU, or changing-
address performance conclusion follows from this evidence.

## Reproduce and verify

From the repository root, using its `.venv`:

```bash
.venv/bin/python scripts/run_mi_research_3c_correctness.py \
  --output artifacts/mi-research-3c/evidence-v1/correctness-preflight
.venv/bin/python scripts/run_mi_research_3_graph.py \
  --m 256 --session 1 --reference-policy fp32_rounded_fp16 \
  --output artifacts/mi-research-3c/evidence-v1/m256-s1
```

Use a **fresh** output path for reruns; the tools reject overwrites.
Repeat the graph command for M512/session 1, M512/session 2, and
M256/session 2. The existing `--verify` commands are read-only:

```bash
.venv/bin/python scripts/run_mi_research_3c_correctness.py \
  --verify --output artifacts/mi-research-3c/evidence-v1/correctness-preflight
.venv/bin/python scripts/run_mi_research_3_graph.py \
  --verify --output artifacts/mi-research-3c/evidence-v1/m256-s1
cd artifacts/mi-research-3c/evidence-v1 && sha256sum -c SHA256SUMS
```

The other three graph directories use the same `--verify` command.
Offline verification needs no CUDA. The historical MI-Research-3 and
MI-Research-3B bundles remain unchanged and verify separately.
