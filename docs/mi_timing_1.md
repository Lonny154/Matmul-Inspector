# MI-Timing-1 — CUDA timing reliability investigation

## Decision

**Clock-discrepancy explanation: supported. Small-kernel performance rankings:
not established.** On this WSL2 RTX 4060 Ti session, Python's
`perf_counter_ns` (`CLOCK_MONOTONIC`) advanced about 5.3–5.5% less per measured
interval than `CLOCK_MONOTONIC_RAW`. CUDA-event time closely tracked RAW on
matched long FC1 blocks. A CPU-only run reproduced the host-clock rate gap
without launching GPU work. This explains why an enclosing `perf_counter`
interval can be numerically shorter than a CUDA-event interval. It does not
establish which clock gives absolutely correct seconds or identify the WSL2
component responsible for the rate difference.

The existing paired runner **does execute the same configured count of calls
for A and B**. In the retained FC1 studies, each call launches one GEMM. It
keeps their CUDA-event measurements separate from host time. The
host-clock mismatch does not by itself invalidate paired *event* differences.
It does make the current event-versus-`perf_counter` boundary gate unsuitable
as a physical ordering check in this environment. The gate has correctly
withheld performance claims, but it cannot diagnose the cause. A same-kernel
A/A control here also showed paired variation comparable to the few-microsecond
per-operation gaps in [MI-Dogfood-2](mi_dogfood_2.md). No prior result is
reclassified and no kernel winner is declared.

## Prior evidence and hypothesis

[MI-Workflow-1](mi_workflow_1.md), [MI-Workflow-2](mi_workflow_2.md),
[MI-Workflow-3](mi_workflow_3.md), [MI-Dogfood-1](mi_dogfood_1.md), and
[MI-Dogfood-2](mi_dogfood_2.md) all reported positive CUDA-event-minus-host-wall
blocks and inconclusive performance decisions. The earlier
[Tensor Game timing study](tensor_game_timing_diagnostics.md) also found
warm-state and order sensitivity. The repository's prior
[host-clock examples](../results/examples/host-clock-timing-diagnostics/README.md)
found that CUDA events and `CLOCK_MONOTONIC_RAW` could agree while
`perf_counter` ran at a different rate. The new hypothesis was that a current
host-clock rate difference, rather than a missing GEMM operation or mismatched
stream, accounts for most of the recurring positive event-minus-wall values.

The control was the same operation and same stream measured within one block
by three clocks. A separate CPU-only clock run tested whether GPU work was
necessary for the difference. A/A blocks used the same function in alternating
slots to estimate paired variability without a kernel change.
The retained environment was Python 3.13.11, PyTorch 2.11.0+cu128, Triton
3.6.0, CUDA runtime 12.8, NVIDIA driver 616.92, and WSL2 Linux
6.18.54.1-microsoft-standard-WSL2.

## Exact measurement windows

The current shared helper in `experiment.py` takes `wall_start =
perf_counter_ns()` immediately before `start.record()`, submits exactly
`iterations` calls, records the end event, takes `submission_end`, synchronizes
the end event, then takes `wall_end`. It queries
`start.elapsed_time(end)` after synchronization. Its event interval spans the
two events on the current CUDA stream. Its host submission interval ends before
synchronization; its synchronized host-wall interval encloses event submission
and completion. The helper divides the event block duration by `iterations`
only for per-operation reporting and retains total block durations. The paired
runner first compiles reference, A, and B with synchronized invocations,
checks A and B against cuBLAS, performs 30 warmup calls per kernel with a
synchronization, records one unranked 100-operation control per kernel, then
eight AB/BA rounds of one 100-operation block per kernel. Its eager
preallocated FC1 wrappers each issue one Triton GEMM launch per call. Thus A
and B have identical operation counts and workload settings, with separate
outputs, and the previous block's end-event synchronization prevents overlap.
Python submission gaps can appear inside the event interval but cannot alone
explain the independent CPU clock-rate finding.

The existing scaling and cross-clock scripts use one explicit stream for both
events and the FC1 kernel. They synchronize before each block, and the end
event is synchronized before reading host end time. The new probe similarly
uses one explicit stream and the following exact sequence:

```text
device synchronize → perf start → RAW start → start event on stream
→ submit N operations on that stream → end event on stream
→ perf submission end → end-event synchronize → RAW end → perf end
```

Both host intervals enclose completion; the `perf_counter` interval is wider
by the two RAW timestamp calls. All raw results are retained as **total
milliseconds per block**. Division by the documented operation count is only
used when interpreting per-operation latency. For `eager_1`, N=1 and one
submission; for `eager_100`, N=100 and 100 submissions; for `graph_100`, a
capture of exactly 100 calls is replayed once, so N=100 operations and one
graph submission. Eager and graph are separate execution modes. The pointwise
control is existing `torch.add(..., out=...)` on 4,096 FP32 elements. FC1 is
the packaged 3-stage `64×128×32` FP16 kernel at M=512, K=768, N=3072.
Both passed their numerical checks; FC1 retained `atol=rtol=0.01` and zero
maximum absolute error on this input. No new kernel was written.

## Raw observations

The [versioned bundle](../artifacts/mi-timing-1/evidence-v1/) contains raw
rows, configurations, software and hardware provenance, summaries, telemetry,
and SHA-256 hashes. The existing scaling diagnostic recorded 12 blocks at
each size, with 100 workload warmups and 10 timing warmup blocks:

| FC1 launches/block | Median CUDA event, ms/block | Median `perf_counter`, ms/block | Median event − perf, ms/block | Median event/perf ratio |
|---:|---:|---:|---:|---:|
| 50 | 3.251 | 3.140 | +0.106 | 1.0346 |
| 500 | 31.207 | 29.683 | +1.568 | 1.0527 |
| 5,000 | 310.736 | 294.561 | +16.173 | 1.0550 |

The descriptive event-versus-perf fit over all 36 blocks had slope 1.0553
and intercept −0.0933 ms. The growing gap is predominantly proportional over
these sizes, not a fixed 0.1–0.6 ms overhead. The fit is not a correction
factor.

The separate existing cross-clock diagnostic used 12 FC1 blocks of 5,000
launches after three timing warmup blocks. Median total durations were
**310.952 ms CUDA event**, **295.498 ms `perf_counter`**, and **311.049 ms
`CLOCK_MONOTONIC_RAW`**. The paired median event-minus-perf gap was
+15.462 ms/block; event-minus-RAW was **−0.094 ms/block**. Median RAW/perf
ratio was **1.05263**. The RAW host bracket contains timestamp and
synchronization-return overhead, so a slightly shorter event interval is
expected from its placement. This is compatibility, not a proof of absolute
timer accuracy.

The existing CPU-only diagnostic recorded 60 fixed-work or sleep blocks with
no GPU operations. Median RAW/perf ratio was **1.05531**, with observed range
**1.05386–1.05668**. `perf_counter` and `CLOCK_MONOTONIC` used the same clock
implementation. The approximately 5.5% difference therefore exists without
Triton, cuBLAS, graph capture, or CUDA synchronization.

Two new matched-window probe sessions each retained 96 blocks: eight rounds,
two identical A/A slots, two workloads, and three modes. The table uses
medians of **total block durations**, not normalized values:

| Session / workload / mode | CUDA event, ms | perf, ms | RAW, ms | Median event − perf, ms | Median event − RAW, ms |
|---|---:|---:|---:|---:|---:|
| v1 add, eager ×100 | 1.069 | 1.037 | 1.093 | +0.024 | −0.026 |
| v2 add, eager ×100 | 0.849 | 0.827 | 0.877 | +0.010 | −0.030 |
| v1 add, graph ×100 | 0.117 | 0.142 | 0.149 | −0.021 | −0.028 |
| v2 add, graph ×100 | 0.113 | 0.143 | 0.151 | −0.023 | −0.031 |
| v1 FC1, eager ×100 | 6.917 | 6.610 | 6.973 | +0.314 | −0.046 |
| v2 FC1, eager ×100 | 6.382 | 6.090 | 6.413 | +0.288 | −0.042 |
| v1 FC1, graph ×100 | 6.681 | 6.369 | 6.719 | +0.292 | −0.050 |
| v2 FC1, graph ×100 | 6.370 | 6.048 | 6.413 | +0.280 | −0.043 |

All 192 recorded probe blocks had event time below the enclosing RAW window.
All 64 FC1 eager/graph blocks had event time
above the perf window. The pointwise eager blocks were positive against perf
in 14/16 and 13/16 blocks; graph blocks were negative in both sessions because
their much shorter execution leaves host fixed costs prominent. The single
operation blocks likewise have large relative host overhead. Thus the rate
disagreement does not imply every short block must have a positive gap.
Graph replay retains the FC1 discrepancy with one host submission per 100
captured operations, so repeated Python launches are not required for it.

## Variability and paired decisions

The probe's A and B slots call the **same** function in alternating AB/BA
order. They are a null control, not two candidate kernels. For 100-operation
FC1 eager blocks, the paired A−B event median was +0.00289 ms/op in v1 and
+0.00183 ms/op in v2. Their ranges were −0.00156 to +0.00703 and −0.00780 to
+0.00469 ms/op. For FC1 graph blocks, the medians were +0.00295 and +0.00375
ms/op, with ranges −0.00473 to +0.00804 and −0.00623 to +0.00999 ms/op.
The A/A medians should not be interpreted as a real kernel difference. They
show that order, warm state, or noise at this measurement scale can create
apparent paired gaps comparable to MI-Dogfood-2's A−B medians of about
0.001–0.004 ms/op. This probe synchronizes before each block, unlike the
paired runner, so it is a diagnostic bound on confidence, not a direct
replacement for that experiment.

FC1 eager event medians differed between probe sessions (6.917 and 6.382
ms/100 operations); graph medians were 6.681 and 6.370. First-half and
second-half medians also shifted within sessions. `nvidia-smi` round snapshots
showed graphics clock mostly 2,565 MHz, one 2,760 MHz sample, memory clock
8,751 MHz, temperature 40 °C, and changing power/utilization. The preflight
compute-process checks were empty. Those coarse snapshots do not isolate a
cause of each timing change or rule out transient background work. The host
clock ratio itself moved between sessions. These effects require repeated
matched controls for future small performance claims.

## Explanation, correction, and limits

The supported explanation is **cross-clock rate disagreement on this WSL2
host**. The CPU-only reproduction eliminates CUDA work as a necessary cause.
Matching explicit streams and exact counts eliminate stream and normalization
mismatches as explanations for the reproduced gap. Graph replay eliminates
per-call Python dispatch as a necessary cause. GPU clock changes may affect
actual kernel latency, but cannot explain a CPU-only RAW/perf rate gap. The
precise system mechanism behind the host-clock rate difference remains
unproven; neither CUDA events nor RAW is declared an absolute oracle.

The CPU clock diagnostic's first invocation in the restricted sandbox could
not query `nvidia-smi`. Its error-reporting path then raised `NameError`
because that standalone script does not import `sys`. The GPU-access rerun
completed and supplied all 60 retained CPU blocks. This ancillary CLI error
path did not affect a measurement; adding `import sys` is a separate small
cleanup, not a timing correction.

No CUDA or benchmark implementation defect was found in launch counts,
synchronization order, or division. The **methodological defect** is treating
the event-versus-`perf_counter` inequality as a universal physical boundary
check when the clocks have measurably different rates. We did not modify the
production helper or historical artifacts. The smallest proposed correction
is additive and versioned: retain existing event and perf fields, also record
an enclosing RAW host interval where available, run a CPU host-clock-rate
preflight, and evaluate a new boundary rule only against validated comparable
clocks. On other platforms or when clock agreement cannot be shown, withhold
the ranking. New RAW evidence here would clear the boundary check for the
FC1 blocks, but the A/A variability still prevents a small speedup claim.
Historical bundles have no RAW field and must retain their original
`INCONCLUSIVE` decisions. Do not multiply old measurements by the fitted
ratio or silently reinterpret them.

The immediate recommendation is to keep eager and graph results separate,
record total block durations and exact counts, include an A/A null control,
repeat candidate comparisons across sessions, and require differences larger
than observed null variation before claiming improvement. Investigate the
host clock environment independently if absolute host latency is needed.

## Validation and reproduction

The diagnostic used only existing kernels and bounded runs. Reproduce from
the repository root using `.venv` and fresh output directories:

```bash
.venv/bin/python scripts/fc1_event_wall_scaling_diagnostic.py --output-root artifacts/mi-timing-1/source-scaling
.venv/bin/python scripts/fc1_cross_clock_diagnostic.py --output-root artifacts/mi-timing-1/source-cross-clock
.venv/bin/python scripts/host_clock_cpu_diagnostic.py --output-root artifacts/mi-timing-1/source-cpu-clock
.venv/bin/python scripts/mi_timing_1_probe.py --output artifacts/mi-timing-1/probe-v1
```

Each probe output is new; use another path for a repeat. The versioned bundle
contains byte-for-byte copies of the existing diagnostics' raw JSON/CSV,
their configurations and summaries, and sanitized provenance. It also
contains both new probe runs with all raw blocks and nested manifests. The
top manifest covers 31 files and has SHA-256
`bf563465beff88a2bc2cfc2c214c091fd1127a35bd0c551a9f4a2f4cd4632315`.
The probe can be replay-verified offline:

```bash
.venv/bin/python scripts/mi_timing_1_probe.py --verify artifacts/mi-timing-1/evidence-v1/probe-v1
.venv/bin/python scripts/mi_timing_1_probe.py --verify artifacts/mi-timing-1/evidence-v1/probe-v2
```

Both returned `PASS`, 96 blocks, six nested hashed files. The top manifest
rehash passed. The relevant CPU tests passed: **43 passed** from the new probe
tests plus `fc1_timing_diagnostic_test.py`, `experiment_runner_test.py`,
`experiment_pair_test.py`,
`experiment_pair_source_test.py`, and `experiment_compare_test.py` using
`.venv/bin/python -m pytest -q`. No framework benchmark code, kernel, search,
or tolerance was changed.
