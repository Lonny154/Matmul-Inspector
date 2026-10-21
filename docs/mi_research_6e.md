# MI-Research-6E — staged CUDA Graph timing controls

## Question and frozen protocol

**Hypothesis.** The large staged-Laderman A/A gaps in MI-Research-6D might
reflect variable graph execution, timing-window effects, or a limitation of
using the largest A/A gap as a performance gate. This study tests those
possibilities without changing the MI-Research-6D decision or Matmul
Inspector's schema-v3 rule.

The control is the committed [MI-Research-6D](mi_research_6d.md) result at
`d569e063859b09096b709eb9a6438c9d7bdb7b00`. Its two session decisions
remain `INCONCLUSIVE`. Before new GPU execution, the
[6E protocol](../artifacts/mi-research-6e/protocol.json) fixed N=96,
20 complete operations per graph, ten eager and ten graph warmups per
instance, eight paired rounds, and two fresh processes for each of three
conditions:

| Condition | X | Y | Purpose |
| --- | --- | --- | --- |
| `staged_null` | Staged Laderman | Independently prepared staged Laderman | Many-kernel identical-algorithm control |
| `fused_null` | Fused Perminov-55 | Independently prepared fused Perminov-55 | Single-kernel identical-algorithm control |
| `cross` | Staged Laderman | Fused Perminov-55 | Repeat the complete-operation comparison with matched buffer separation |

Each instance owns distinct fixed-address FP16 left/right inputs and output.
Staged instances also own separate coefficient and FP32 product-scratch
buffers. Corresponding input buffers contain the same values. Both instances
have their own graph objects and the same 20-operation capture design.
Session 1 captures X then Y; session 2 captures Y then X. The eight paired
rounds alternate X/Y and Y/X. The cross condition also measures eight
same-instance staged A/A blocks before its paired blocks, following the
original 6D convention. The source kernels, mathematical coefficients,
precision policy, tile mapping, warps, stages, numerical tolerances, and
production decision rule are unchanged.

The predeclared offline checks are signs, paired distributions, absolute
null-gap medians/90th percentiles/maxima, first-versus-second-half drift,
block position, host/device timing intervals, and the unchanged schema-v3
decision. The quantiles are descriptive sensitivity checks. They are not a
replacement performance threshold. No observed cross result was used to
choose a new rule.

## Measurement and correctness

The RTX 4060 Ti has 34 SMs and compute capability 8.9. These sessions used
driver 616.92, PyTorch 2.11.0+cu128, and Triton 3.6.0. Every graph was
compiled and warmed before capture. Graph construction, first calls,
correctness checks, and profiler traces ran outside primary timed rounds.
The retained raw records include session setup costs, clock preflight,
source hashes, environment, and each timing block's round, phase, order,
position, and outer `perf_counter_ns` timestamps.

One timed block measures **one graph replay containing 20 complete
operations**. Matmul Inspector starts `perf_counter`, then RAW if available,
records a CUDA start event, submits the replay, records a CUDA end event,
records host submission end, synchronizes the end event, reads RAW and
`perf_counter`, then queries CUDA-event elapsed time. CUDA events bracket a
stream interval that may include queue or stream idle time. Host submission
ends before device completion. The synchronized host intervals include the
wait. These are different windows. The script retains all three clock
readings, with no outlier removal or clock rescaling. All six host-clock
preflights flagged a resolved RAW versus `perf_counter` rate difference.

Seeds 0, 1, and 2 and the established zero, identity, and cancellation
cases used the common FP32-rounded reference
`torch.matmul(left.float(), right.float()).half()`. TF32 and reduced FP16
reduction were disabled during the checks and restored afterward. The
existing `atol=rtol=0.01` gate was unchanged. Before each of two graph
replays per case, all outputs and staged product scratch were filled with
NaNs. All 144 graph checks passed with zero failing and nonfinite output
elements. Each graph output matched its own eager call bitwise; X and Y
also matched bitwise in both null controls. Changed contents at the same
addresses passed. These checks establish correctness for the tested inputs
and sequence, not every input.

Every staged profiler trace has 480 kernel events: 20 repetitions of 23
product kernels followed by reconstruction. Every fused trace has 20
single-kernel events with the expected 2×2 grid and 128 threads per CTA.
Each trace has one `cudaGraphLaunch`. The profiler ran outside the timed
comparison. It is not a measurement of the slow primary timing blocks.

## Raw result summary

Values below are untrimmed CUDA-event microseconds **per complete
operation**. The X−Y sign has no winner meaning in the null controls.
The paired absolute median, 90th percentile, and maximum use eight
whole-block paired differences. The empirical percentile linearly
interpolates sorted observations.

| Condition / session | X median | Y median | Absolute paired median / q90 / max | Positive X−Y rounds |
| --- | ---: | ---: | ---: | ---: |
| Staged null 1 | 59.061 | 58.375 | 0.445 / 7.536 / 22.350 | 5/8 |
| Staged null 2 | 58.722 | 58.058 | 0.706 / 0.965 / 1.238 | 5/8 |
| Fused null 1 | 16.057 | 15.837 | 0.232 / 0.755 / 0.949 | 4/8 |
| Fused null 2 | 16.107 | 16.174 | 0.305 / 2.026 / 5.091 | 4/8 |
| Cross 1 | 59.401 | 16.706 | 42.695 / 47.818 / 57.322 | 8/8 |
| Cross 2 | 58.164 | 15.863 | 42.372 / 43.232 / 43.600 | 8/8 |

The paired X−Y differences in round order are retained here as well as in
each raw bundle:

| Condition / session | Eight differences, µs/op |
| --- | --- |
| Staged null 1 | −0.058, +0.957, −0.011, +22.350, +0.288, +0.602, −0.238, +1.187 |
| Staged null 2 | −0.646, −1.238, −0.205, +0.848, +0.555, +0.814, +0.766, +0.646 |
| Fused null 1 | −0.037, −0.949, −0.035, −0.382, +0.008, +0.672, +0.294, +0.170 |
| Fused null 2 | +0.712, −0.208, +0.400, −0.600, −0.210, −0.038, +5.091, +0.006 |
| Cross 1 | +43.053, +57.322, +43.746, +43.530, +42.338, +40.742, +41.798, +41.654 |
| Cross 2 | +43.600, +42.573, +41.808, +43.074, +42.382, +42.362, +42.112, +41.200 |

The largest staged-null session-1 gap is one 22.350 µs block pair. Its
second-largest absolute gap is 1.187 µs. The largest fused-null session-2
gap is one 5.091 µs pair; its second-largest is 0.712 µs. Thus the
maximum is much more sensitive to isolated slow blocks than the median
in these two null sessions. Relative to the two instance medians, those
maxima are about 38% for staged and 32% for fused. Fused relative
variability can therefore also be large even though its absolute gaps are
smaller. The other staged/fused null maxima were 1.238/0.949 µs. Eight
rounds per session are too few to estimate tail probabilities.

Between the two sessions of each condition, the X/Y median changes were
−0.339/−0.318 µs for staged null, +0.050/+0.337 µs for fused null, and
−1.237/−0.842 µs for cross. First-half versus second-half medians and
first-versus-second block-position medians are saved per session in
[`analysis.json`](../artifacts/mi-research-6e/analysis.json). They show
small shifts for most sessions and no repeatable balanced-order effect.
They do not explain the older 6D A/A pattern.

## Slow blocks, host intervals, and profiler evidence

The staged-null session-1 X block in round 4 was 80.717 µs/op by CUDA
event, against its 59.061 µs/op session median. Its host submission
interval was 5.638 µs/op; the synchronized `perf_counter` and RAW
intervals were 83.102 and 83.284 µs/op. This is a larger device-event
window without a comparably large host submission interval. The data do
not separate longer kernel execution from queue or stream idle time inside
that event window.

The fused-null session-2 X block in round 7 was 21.554 µs/op by event,
against its 16.107 µs/op median. Host submission was 8.143 µs/op, and
synchronized `perf_counter`/RAW were 24.781/24.779 µs/op. Host submission
was also elevated in this block. That correlation does not prove the host
caused the event-window increase.

The older 6D A/A data had a broader pattern than an isolated maximum. In
session 1, the first three Y blocks were 120.960, 119.394, and 97.200
µs/op while their X counterparts were 68.696, 62.994, and 65.426. In
session 2, the first four Y blocks were 113.507–128.237 µs/op while X
was 60.390–70.776. These slow Y blocks occurred under alternating X/Y
and Y/X order. Host submission intervals did not rise in a matching,
consistent way. A later X block in 6D session 2 was also slow. The 6E
controls did not reproduce this sustained early Y pattern. The old
56.400 and 66.546 µs maximum gaps were therefore not just single
isolated outliers.

Normal, instrumented staged traces in 6E show median product-kernel
durations near 2.39 µs, reconstruction near 1.66–1.70 µs, and median
inter-kernel gaps near 0.032 µs. Their per-operation spans were about
59 µs. No unexpected long gap appears in those *normal profiled replays*.
The trace cannot explain a slow timed block that it did not capture.
We did not run a broader Nsight sweep or treat profiler time as benchmark
time. A targeted slow-block trace would need its own bounded protocol.

## Decision-rule sensitivity and limits

The unchanged schema-v3 rule returns `B_FASTER` in both **new** cross
sessions. All eight cross gaps in each session are positive. Their minimum
gaps, 40.742 and 41.200 µs/op, exceed both same-instance staged A/A
maxima (12.498 and 9.106 µs/op) and the independently prepared staged-null
maxima (22.350 and 1.238 µs/op). The fused-null maxima are 0.949 and
5.091 µs/op. Thus the new cross result remains directional under each
predeclared descriptive control comparison. It is a new measurement under
separate fixed input buffers and different process conditions; it does
**not** retroactively change either 6D decision.

The maximum-gap rule can be dominated by one slow null block, as the
staged-null and fused-null examples show. That is a conservative feature
of the current gate, not proof of a defect. The large 6D A/A gaps were
several slow blocks, not one stray maximum. Independent null conditions
were measured in separate processes from the cross condition, so their
distributions are not simultaneous estimates of cross-session noise.
Eight blocks and two processes cannot establish tail coverage or whether
same-instance versus independent-instance A/A is the better production
control. No alternative rule was adopted or used to relabel 6D. Any
replacement would need prospective testing on independent nulls and known
contrasting implementations, with block and session structure respected.

**C — variability remains unexplained.** All controls and the new
comparison passed correctness and physical-execution gates. We observed
isolated slow null blocks and a repeatable directional new cross result,
but did not establish the physical or timing-window cause of 6D's
sustained staged A/A gaps. The controls do not demonstrate that the
production rule is inappropriate. This classification concerns the
measurement investigation; it is not a new mathematical or kernel
optimization claim.

**One evidence-backed next step:** design one prospective, bounded
slow-block trace study that can capture a staged graph replay when its
event interval grows while recording kernel durations and queue gaps.
Keep that profiler run separate from ordinary timing. Do not change the
decision rule before null and known-contrast validation.

## Evidence and reproduction

The evidence root is [`artifacts/mi-research-6e`](../artifacts/mi-research-6e/).
Each of its six session bundles retains source, coefficient, protocol,
and diagnostic-script snapshots; repeated correctness checks; one
out-of-band profiler trace per graph; raw timing blocks; a replayed
summary; and a SHA-256 manifest. The protocol SHA-256 is
`36749ef82bff1a9f278df0dc488dad00c0b2f0c7cc64a2e402cd34bc11dc337e`.
Source hashes are the frozen 6D values: staged
`8b328950567868b2d119e021a0dd3bcc312d6e1c8138a1577b0e0491949fbca1`,
fused
`c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db`,
and staged coefficients
`4abdcb3db4094b17dba24e8c23b17206fae2e82d769b61fa35952831480a68ef`.
The [analysis manifest](../artifacts/mi-research-6e/analysis_manifest.json)
binds the cross-session analysis to all six bundles and the protocol.

Cross session 1 completed GPU collection, then its first offline summary
attempt failed because the adapter omitted the required
`iterations_per_measurement` field. The original `failure.json` and
diagnostic-script snapshot remain. A separate, hashed repair-script
snapshot replayed the complete retained raw data, wrote `summary.json`,
and recorded `recovery.json`. No GPU timing was repeated or discarded.
Cross session 2 used the corrected adapter. The normal verifier checks
both sessions and the recovery provenance.

Offline verification needs no GPU:

```sh
for condition in staged_null fused_null cross; do
  for session in 1 2; do
    .venv/bin/python scripts/run_mi_research_6e_controls.py --verify --output "artifacts/mi-research-6e/${condition}-s${session}"
  done
done
.venv/bin/python scripts/analyze_mi_research_6e.py --verify
.venv/bin/python -m pytest -q tests/mi_research_6e_test.py tests/mi_research_6d_test.py tests/mi_research_6c_test.py tests/mi_research_6b_test.py tests/mi_research_6_test.py tests/mi_research_2c_test.py
```

The collection command for each fresh process was:

```sh
.venv/bin/python scripts/run_mi_research_6e_controls.py --condition staged_null --session 1 --output /tmp/mi-research-6e-staged-null-s1
```

Substitute `staged_null`, `fused_null`, or `cross` and session `1` or `2`
for the six fixed combinations. Use new output directories. The runner
refuses to overwrite retained evidence. Run
`.venv/bin/python scripts/analyze_mi_research_6e.py --root /tmp/your-evidence-root`
after placing the protocol and all six bundles under that root. No kernel
binary was retained.
