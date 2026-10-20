# MI-Research-2B — Split-K latency variability

## Retained-evidence audit before new GPU work

The [MI-Research-2 case study](mi_research_2.md) compared complete two-launch
Split-2 against one-launch conventional GEMM on the RTX 4060 Ti. Its three
bundles and top-level checksum replayed successfully. The original low-output
shape was FP16 M=32, K=4096, N=2048. Both candidates passed the unchanged
three-seed PyTorch/cuBLAS gate. The two low-output sessions each had six
rounds favoring Split-2 and two favoring conventional A. Both saved
decisions were `INCONCLUSIVE`. None of those historical records is changed.

The exact slow B rounds were **5 and 6** in the first session (98.564 and
98.191 µs/operation) and **1 and 7** in the fresh repeat (68.165 and
69.079 µs/operation). Their A/B orders were AB, BA, AB, AB. The unranked
B controls were also slow at 97.3 and 68.9 µs/operation respectively. A's
corresponding rounds remained in its normal 57–65 µs range. Nearby
identical-A A/A samples did not reproduce the B jumps. The slow rounds
therefore do not align with one round index, A/B order, or a simultaneous
conventional-GEMM slowdown.

For each slow 100-call B block, synchronized `perf_counter` and RAW times
rose with CUDA-event time. The five-window host-clock preflights found no
resolved RAW/perf rate difference. More specifically, B host **submission**
time rose from about 2.7–3.8 ms per 100-call block in fast rounds to
9.88–9.89 ms in the first session's slow rounds and 6.71–6.89 ms in the
repeat's slow rounds. Event totals in those four blocks were 9.86, 9.82,
6.82, and 6.91 ms. Host submission is almost as long as the event window
there. This strongly suggests that launch/allocator pacing is involved in
the complete-operation slow state. CUDA-event intervals can include GPU
idle gaps while the host enqueues work; they are not a sum of pure kernel
execution durations. The saved aggregate blocks cannot identify whether
`torch.empty`, the partial launch, the reduction launch, or another host
condition caused the longer submission. The coarse pre/post `nvidia-smi`
snapshots cannot reconstruct clocks during individual rounds.

The FC1 control does not show the same B latency bimodality. Its B event
range was 69.806–81.252 µs/operation. Two FC1 B blocks had host submission
times above 6 ms, but their event totals remained about 7.4–7.6 ms per
100-call block. A longer GPU workload could mask host pacing in that shape;
this is an inference, not a measured mechanism.

The original `matmul_into` allocates a logical 0.5 MiB FP32 scratch tensor
on every call, launches `split_k_partial`, then launches
`reduce_partials` into preallocated FP16 output. The existing paired timer
starts before the call loop, ends after it, and synchronizes the end event.
Thus allocation and both launches are included in complete-operation host
wall and event-bracketed times. The event interval may include host-caused
gaps. The trusted-local callable returns only the reduction launch result,
so the original bundle has reduction PTX and metadata but not partial-GEMM
PTX. Source code specifies two launches; it does not prove profiler-observed
launch counts or achieved occupancy.

## Pre-run diagnostic plan

Recorded before any MI-Research-2B GPU execution. The primary question is
whether the B slow state tracks scratch allocation, partial launch
submission, reduction launch submission, or device execution. Reuse the
committed exact Split-2 source and the registered conventional kernel. Do
not edit either kernel or the normal paired benchmark engine.

Run **two fresh CLI processes** on the low-output shape. Each process uses
the same seeded inputs, the unchanged numerical gate, 30 warmups, eight
rounds, and 100 complete calls per measured block. Alternate the complete
per-call-scratch and preallocated-scratch order within every session. The
preallocated-scratch comparison is a **separate diagnostic**, not a
replacement for the original complete-operation result. Include a
conventional A block each round as a stability control. Characterize
partial-only and reduction-only blocks with preallocated scratch. Reverse
the outer phase order between the two sessions. Retain all three clock
totals, host submission times, exact call counts, and individual rounds.

In a separately labeled instrumented probe, retain per-call CPU durations
for scratch allocation, partial-launch submission, and reduction-launch
submission. That instrumentation perturbs the original wrapper, so it is
used to locate a host bottleneck, not to replace original latency numbers.
Where the environment supports it, capture the compiled partial and
reduction kernel PTX and metadata directly. A bounded CUDA Graph phase
probe may help distinguish device-phase work from eager host pacing; it
will be labeled as a different execution mode and will not be pooled with
the original eager paired timings.

The diagnostic stops after two sessions. No slow sample will be removed.
The result may support a host-path association while leaving the exact
allocator or GPU mechanism unresolved. A practical Split-2 optimization
claim would require correct output, complete-operation advantage with
stable paired directions across fresh sessions, tolerable latency variation,
and clock/phase evidence that does not hide the slow mode.

## Controlled-scratch addendum before the final two sessions

The first two diagnostic sessions are retained as exploratory evidence.
They independently show complete-operation slow blocks and expose both
kernel PTX artifacts. They also revealed a comparison flaw: the initial
scratch-reuse wrapper precomputed launch grids and used helper calls, while
the original allocating entrypoint recomputed its grids. That changes more
than scratch allocation, so their alloc-minus-reuse timing cannot isolate
allocation cost. This flaw does not alter the historical MI-Research-2
bundles or the first diagnostic's original per-call-scratch measurements.

The corrected reuse wrapper mirrors the committed `matmul_into` body,
including parameter reads, grid construction, launch arguments, and launch
order. Its sole intentional change is supplying preallocated FP32 scratch
instead of calling `torch.empty` each time. The corrected wrapper will pass
the same three-seed correctness gate before timing. Run exactly two more
fresh sessions, with the same eight rounds, 100 calls per block, balanced
alloc/reuse order, and reversed outer group order. Keep the first two
sessions and both script snapshots. No further variants or GPU sessions are
planned after this correction.

## Controlled diagnostic results

The two corrected sessions ran in separate processes on the RTX 4060 Ti.
Both passed the unchanged three-seed PyTorch/cuBLAS numerical gate for A,
allocating Split-2, and scratch-reuse Split-2. Each captured eight individual
100-call blocks per variant. The saved order alternates allocation and reuse
within each session. Session 2 reverses the outer measurement-group order.
The source snapshots match the committed MI-Research-2 Split-2 source. The
predeclared correction is recorded in `controlled_plan.json`. The two earlier
sessions and their original script snapshots remain in the bundle.

| Corrected session | Conventional event median; range | Split-2 alloc event median; range | Split-2 reuse event median; range | Paired alloc − reuse median; positive rounds |
| --- | ---: | ---: | ---: | ---: |
| 1 | 70.08; 65.16–72.57 µs | 45.51; 41.70–114.20 µs | 43.23; 35.83–137.08 µs | +5.44 µs; 5/8 |
| 2 | 62.72; 59.56–69.71 µs | 35.85; 31.65–69.87 µs | 38.79; 31.30–58.85 µs | −1.97 µs; 3/8 |

These are **diagnostic event-bracketed complete-operation times**, not a new
Matmul Inspector A/B ranking. The original paired decisions remain
`INCONCLUSIVE`. Session 1 had a shared slow state in round 7: alloc 114.20
and reuse 137.08 µs/call. Session 2 had a shared slow state in round 6:
69.87 and 58.85 µs/call. The corresponding alloc/reuse host-submission
totals were 11.50/13.74 ms and 7.03/5.91 ms per 100-call block. The
conventional blocks in those rounds were 68.27 and 63.20 µs/call. Scratch
reuse therefore did **not** remove the slow state. The direction of the
alloc-minus-reuse median also changed between sessions. Per-call scratch
allocation is measurable host work, but these results do not isolate it as
the cause of the large outliers or establish a stable reuse speedup.

| Slow block, 100 calls | CUDA event | Host submission | Synchronized `perf_counter` | Synchronized RAW |
| --- | ---: | ---: | ---: | ---: |
| Session 1 round 7, alloc | 11.420 ms | 11.505 ms | 11.551 ms | 11.551 ms |
| Session 1 round 7, reuse | 13.708 ms | 13.741 ms | 13.791 ms | 13.791 ms |
| Session 2 round 6, alloc | 6.987 ms | 7.035 ms | 7.037 ms | 7.037 ms |
| Session 2 round 6, reuse | 5.885 ms | 5.905 ms | 5.924 ms | 5.924 ms |

Each block uses the existing timer: read `perf_counter`, read RAW, record
CUDA start event, enqueue exactly 100 complete calls, record CUDA end event,
read host submission time, synchronize the end event, then read RAW and
`perf_counter` again. The event spans work and any host-caused gaps between
launches. Synchronized host time also includes the final wait. The start
and end clock reads are sequential rather than mathematically identical;
the raw files retain their exact totals before per-call normalization.

Across the corrected sessions, alloc event-block versus host-submission
Pearson correlations were 0.995 and 0.993. Reuse correlations were 0.989
and 0.961. These descriptive associations support host submission pacing as
a contributor to event-bracketed latency. They do not identify whether
allocation, Triton launch preparation, scheduling, or GPU backpressure is
the initiating cause. In the separately instrumented probe, median host
scratch allocation was 9.21 µs/call in session 1 and 2.23 µs/call in
session 2. Partial-launch submission was 45.51 and 11.48 µs/call;
reduction-launch submission was 25.07 and 9.64 µs/call. This probe adds
timestamps inside the call and is **not** directly comparable to the
uninstrumented complete-operation blocks. The entire host path changed
between sessions, not just `torch.empty`.

The phase-only eager event medians were 39.78/31.30 µs for partial GEMM
and 13.38/13.37 µs for reduction in the two corrected sessions. A reduction
block reached 34.61 µs in session 1 and 31.95 µs in session 2; each had
elevated host submission. Partial blocks also varied, but the shared
complete-operation outliers do not map to one consistently slow device
phase. The separately labeled 100-node CUDA Graph probes measured partial
nodes in roughly 27–34 µs and reduction nodes near 1.2 µs each. Graph
replay removes per-node Python submission and uses a different execution
mode. It makes a persistent 30 µs reduction-kernel execution time unlikely,
but cannot reconstruct every eager GPU interval or prove a single cause.

All four diagnostic sessions retained PTX and metadata for conventional
GEMM, Split-2 partial GEMM, and reduction. In the corrected sessions the
conventional and partial kernels used 24,576 bytes shared memory and 119
and 124 registers per thread respectively; reduction used no shared memory
and 22 registers. Each reported zero spills. PTX identities and exact
metadata are in `analysis.json` and the session raw files. This is compiler
evidence, not measured occupancy or decoded SASS. The configured grids are
16 conventional CTAs, 32 partial CTAs, and 64 reduction CTAs. More partial
CTAs than the conventional 16 is a plausible way to expose more K work to
the 34-SM GPU, but occupancy and utilization were not measured. Each
Split-2 call still writes a logical 0.5 MiB FP32 scratch tensor and enqueues
two kernels, compared with one conventional launch.

The corrected pre/post `nvidia-smi` snapshots showed 44–45 °C and idle or
low-utilization clock readings. They were taken outside the timed blocks
and cannot explain a particular round. All four diagnostic host-clock
preflights reported `NO_RESOLVED_DIFFERENCE`. RAW and `perf_counter` block
durations, as well as CUDA events, are preserved individually in `raw.json`
and copied into `analysis.json`. None of the slow samples was removed.

## Interpretation and next step

The original median advantage recurred in both historical paired sessions.
The corrected diagnostic also had lower *median* alloc Split-2 event time
than conventional A in both sessions. One exploratory session did not show
that median direction, and all four diagnostic sessions include material
variability. The unusually slow complete Split-2 blocks recur, including
with preallocated scratch. They coincide with longer host submission, but
no single allocation or kernel phase is established as their cause.

**Decision: the Split-2 practical optimization claim remains inconclusive.**
The historical paired decision rule is unchanged. The evidence justifies
further Split-K research only if it can control or explain the slow host
submission state and then show stable complete-operation paired advantages
across fresh sessions. A lower typical median alone is insufficient.
Eight rounds per session cannot support a reliable tail-latency percentile.
This study did not measure achieved occupancy, profiler-observed launch
counts, or in-round clocks, and it does not reinterpret historical results.

## Reproduction and verification

The bounded run used the existing `.venv` and these commands, one per fresh
process. The output directories were new and no kernel or benchmark engine
was modified:

```bash
.venv/bin/python scripts/run_mi_research_2b_diagnostic.py --session 1 --output artifacts/mi-research-2b/evidence-v1/controlled-s1
.venv/bin/python scripts/run_mi_research_2b_diagnostic.py --session 2 --output artifacts/mi-research-2b/evidence-v1/controlled-s2
```

For offline verification, run `--verify --output` against each of `s1`,
`s2`, `controlled-s1`, and `controlled-s2`, then run
`sha256sum -c artifacts/mi-research-2b/evidence-v1/SHA256SUMS` from the
repository root. The per-session verifier checks complete file manifests,
source/spec/script identity, PTX hashes, round order, operation counts,
correctness records, and deterministic summary replay. The top-level
checksum covers the planning record, historical extraction, all raw files,
telemetry, and the derived analysis.

Validation on the repository `.venv`: 43 relevant CPU tests passed. All four
MI-Research-2B session manifests and summaries replayed offline. The
MI-Research-2 historical top-level checksum and all three historical paired
bundles also verified. The MI-Research-2B top-level checksum passed for all
48 files it lists. No additional GPU runs are needed for this decision.
