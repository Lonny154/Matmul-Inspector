# MI-Timing-2 — clock-aware paired benchmark diagnostics

## Result

**Workflow decision A: the paired report now explains the clock discrepancy and
retains a local A/A variability control. Performance decision: INCONCLUSIVE.**
The one bounded RTX 4060 Ti run compared the established FP16 FC1 GEMM with
three pipeline stages (A) against the same configuration with two stages (B).
Both passed the independent PyTorch/cuBLAS reference on seeds 0, 1, and 2.
The five-window host preflight found a RAW/`perf_counter` duration ratio near
1.0545. CUDA-event time exceeded `perf_counter` wall time in all 34 measured
blocks, but stayed below the enclosing RAW interval in all 34. This is
consistent with [MI-Timing-1](mi_timing_1.md). It does not establish the
absolute accuracy of either host clock.

The paired CUDA-event A−B signs split four to four. The A/A control also
produced a 0.006728 ms/operation absolute paired gap. There is no supported
ranking or performance improvement. B's lower median event time and reduced
compiler shared memory are observations, not a performance win.

## Shortest workflow

From the repository root, use a fresh output path and the project `.venv`:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_timing_2_fc1_pair.json \
  --output results/my-clock-aware-pair
.venv/bin/matmul-inspector-python experiment pair \
  --verify results/my-clock-aware-pair
```

Read `report.md` for correctness, three-clock medians, paired per-round
differences, A/A variability, warnings, compiler resources, and PTX identity.
Read `comparison.json` for the decision, warning codes, preflight samples,
and A/A/A/B statistics. `raw_result.json` retains the original total block
durations and every A/A and A/B measurement. `timings.jsonl` retains paired
A/B rows and their operation counts. The manifest hashes all files. The
[v3 example](../examples/mi_timing_2_fc1_pair.json) opts into diagnostics
through `schema_version: 3`; the [v3 schema](../schemas/mi_timing_2_pair_v3.schema.json)
keeps the v2 preallocated kernel and resource contract. Versions 1 and 2
retain their original measurement and decision paths.

## Hypothesis and control

The hypothesis was that adding matched RAW host timing, a CPU host-clock
preflight, and an interleaved identical-kernel control would explain the
recurring event-versus-`perf_counter` reversal and make small A/B gaps easier
to judge. The control is kernel A measured twice in each round. Its `X` and
`Y` slots use the same function, inputs, output, warm state, and 100-call
block count. Slot order alternates XY/YX. One X/Y pair precedes each A/B
pair. A/A is a local null control, not a second candidate or a confidence
interval. There are eight A/A and eight A/B rounds after separate compilation,
correctness checks, warmups, and two unranked timer controls.

Each timed block uses the existing eager preallocated wrapper. The exact
sequence is:

```text
perf start → RAW start (when available) → CUDA start event
→ exactly 100 wrapper calls on the current stream → CUDA end event
→ perf submission end → end-event synchronization → RAW end → perf end
```

The CUDA events bracket GPU stream work. Both host wall windows enclose
event submission and completion. The perf window is slightly wider because
it also includes both RAW timestamp reads. Host submission time ends before
synchronization and is not used as an enclosing interval. We record total
milliseconds per block before dividing by the exact call count. The same
call count and bracket apply to X, Y, A, and B. These windows are not
identical in host read overhead, so numerical cross-clock comparisons are
diagnostics, not an automatic declaration of which clock is correct.

The preflight measures five separate roughly 20 ms CPU sleep windows. It
retains each clock duration, ratio, difference, and timestamp-read span.
`RATE_DIFFERENCE_OBSERVED` means all five differences share a sign and each
exceeds the largest measured paired read span. This is a detection rule for
a visible host clock rate gap. It is not a correction factor, a speedup
threshold, or proof that RAW gives absolute seconds. If RAW is unavailable,
the report says so and withholds a clock-aware ranking.

For v3, a supported A/B direction requires the established numerical gate,
complete controls, nonoverlapping CUDA-event ranges, agreement of every
paired `perf_counter` and RAW host direction, and every absolute A/B paired
event gap larger than the largest observed A/A gap. A positive event-minus-
RAW enclosing interval makes the decision inconclusive. If the host preflight
does not resolve a clock-rate difference, a positive event-minus-perf
interval also makes it inconclusive. This rule uses observed null variability
and range separation. It assigns no probability or arbitrary percentage
confidence. A preflight rate difference explains why the perf boundary can
reverse; it does not silently rescale any measurement. Historical v1/v2
decisions are never recomputed with this v3 rule.

## Retained RTX 4060 Ti run

The [versioned evidence bundle](../artifacts/mi-timing-2/fc1-pair-v1/)
was produced with the example command above, changing only the output path.
The workload was M=512, K=768, N=3072, FP16, preallocated eager mode,
seed 0, 30 warmups per kernel, 100 calls per measured block, and eight
paired rounds. A and B use `64×128×32` tiles, four warps, and
`group_size_m=1`. A has three stages and B has two. Both use the packaged
FC1 source. Python was 3.13.11, PyTorch 2.11.0+cu128, Triton 3.6.0,
CUDA runtime 12.8, and driver 616.92. The run recorded the RTX 4060 Ti
with compute capability 8.9.

Both kernels passed shape, FP16 dtype, finiteness, and `allclose` at the
unchanged `atol=rtol=0.01`. Maximum absolute error was 0.0 for both on
all three retained seeds. Those checks cover the saved inputs only.

| Metric | A: three stages | B: two stages |
|---|---:|---:|
| CUDA-event median, ms/operation | 0.063466 | 0.061648 |
| CUDA-event range, ms/operation | 0.059556–0.066998 | 0.059474–0.070583 |
| CUDA-event CV | 4.45% | 6.12% |
| `perf_counter` wall median, ms/operation | 0.060586 | 0.058878 |
| MONOTONIC_RAW wall median, ms/operation | 0.063888 | 0.062085 |

The paired CUDA-event A−B median was **+0.0000667 ms/operation**, with range
**−0.007108 to +0.003799**, MAD **0.000737**, and four signs in each
direction. The A/A X−Y median was **+0.0000194 ms/operation**, with range
**−0.001294 to +0.006728**. The largest absolute A/A gap was **0.006728**;
several A/B gaps were smaller. The A/A outlier is retained. It could reflect
warm-state or background variation; this run does not isolate its cause.

The preflight median RAW/perf ratio was **1.054522** across five windows.
Across the two unranked controls, 16 A/A blocks, and 16 A/B blocks,
**34/34** had event time above the perf wall and **0/34** had event time
above RAW wall. Median event-minus-perf was **+0.283602 ms/block**; median
event-minus-RAW was **−0.044196 ms/block**. The latter relation is compatible
with RAW's enclosing placement. Neither relation proves absolute clock
accuracy. The report emits `HOST_CLOCK_RATE_DIFFERENCE`,
`CUDA_EVENT_EXCEEDS_PERF_COUNTER_WALL`, and `AB_WITHIN_AA_VARIABILITY`.
Its machine-readable decision is `INCONCLUSIVE` with no speedup value.

Triton reported A at **24,576 shared-memory bytes/program** and **121
registers/thread**; B at **12,288 bytes** and **122 registers/thread**.
Both had zero spills. Both PTX files were captured, with 1,037 normalized
changed lines in the bounded comparator. Decoded SASS was unavailable.
These compiler differences do not establish a machine-code latency gain.

The bundle has 16 hashed files, including both PTX files, their normalized
comparison, source snapshot, raw result, report, correctness, provenance,
and timing rows. Its manifest SHA-256 is
`aae53bcb0239ffb5bbc8f1b7bd4f008f192ed24ea5c38137a0fff23f891338c3`.
After the GPU run, only `report.md` was regenerated offline to add the
per-round three-clock table; the raw measurements, correctness, and compiler
captures did not change. The manifest was regenerated and the complete
bundle replayed successfully. The source tree was dirty because unrelated
user files and this implementation were present; the source snapshot and
manifest supply artifact identity.

## Validation and limits

The relevant CPU suite passed **49 tests**, including v3 clock preflight,
matched-call windows, A/A ordering, correctness gating, missing RAW,
inconclusive controls, report generation, and offline replay. The historical
[MI-Workflow-2](../artifacts/mi-workflow-2/fc1-pair-v1/) and
[MI-Workflow-3](../artifacts/mi-workflow-3/local-fc1-v1/) bundles still
verify as `INCONCLUSIVE` with 16 and 17 hashed files respectively. The new
bundle verifies offline as `PASS`, `INCONCLUSIVE`, eight A/B rounds, and
16 hashed files. No historical measurement was changed or reinterpreted.

One GPU run cannot establish long-session clock stability. The CPU
preflight precedes the warmups, so a later rate change might escape it.
The A/A control samples local variation but cannot prove that the A/B
interval had the same noise distribution. It also launches A twice before
each A/B pair, so it may alter A's immediate warm state. The observed A/B
ranges overlap
and the paired directions disagree even without the A/A guard. Future
optimization studies should use this report to diagnose timing first,
then repeat a promising comparison across independent sessions before
claiming a small gain. Investigating WSL2 clock behavior remains a separate
system task; this milestone does not alter system clocks or benchmark history.
