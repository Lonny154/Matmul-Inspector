# MI-Timing-3 — A/A control warm-state isolation

## Pre-run plan

This plan was recorded before GPU sessions on 2026-10-20. The hypothesis is
that putting the identical-kernel A/A pair immediately before A/B may change
the next A/B timings through warm state, cache, GPU clock, or other execution
conditions. The null is that placement has no detectable effect at this
experiment's sensitivity. Neither mechanism is assumed.

Use the committed [MI-Research-1 eight-warp spec](../examples/mi_research_1_warps8.json)
for every run: A is the original three-stage `64×128×32`, four-warp FC1
kernel. B is the existing trusted-local eight-warp source with the same
tile, stages, workload, and `group_size_m=1`. Numerical checks remain
seeded FP16 M=512, K=768, N=3072 against PyTorch/cuBLAS at unchanged
`atol=rtol=0.01`. The normal v3 timing settings remain 30 warmups per
kernel, 100 eager preallocated calls per block, eight A/B rounds, and eight
A/A rounds. A/B order alternates AB/BA within every session. Each block
retains CUDA-event, `perf_counter`, and MONOTONIC_RAW durations and exact
call counts. An A/A pair uses A in both X and Y slots, alternating XY/YX.

The only manipulated factor is whether the pair of A/A blocks comes before
or after the A/B pair in each round. Four independent CLI processes will run
in a predeclared ABBA condition order:

| Session | A/A placement | Phase order in every round |
|---|---|---|
| 1 | before | A/A then A/B |
| 2 | after | A/B then A/A |
| 3 | after | A/B then A/A |
| 4 | before | A/A then A/B |

Each process uses one fresh output directory. Record `nvidia-smi` graphics
and memory clocks, temperature, power, and utilization immediately before
and after each CLI invocation where available. These snapshots cannot
reconstruct clocks during every block. Keep session order and all failed
attempts if any. No new kernel or custom benchmark loop is permitted.

The primary unit of comparison is the **session**, not 32 independent
rounds. Compare per-session A and B CUDA-event medians, the eight paired
A−B differences and their spread, and corresponding host-clock values.
Inspect within-session trends, A/A X−Y distributions, and movement between
sessions. A placement claim would need a consistent condition contrast in
both counterbalanced session pairs, without a similarly sized session-order
or telemetry contrast. Overlap, drift, or inconsistent directions support
an inconclusive classification. The maximum A/A gap is descriptive here,
not a significance cutoff. No cause such as cache or clock is attributed
without a control that isolates it.

The study will end with A (reproducible placement effect), B (no meaningful
effect detected within sensitivity), C (variability prevents distinction),
or D (blocked or invalid). Any recommendation to the normal workflow will
be narrow and evidence-backed. Historical timing bundles remain unchanged.

## Result

**Decision C: timing variability prevents distinguishing the two placements.**
Both A/A-after sessions had a negative median paired A−B difference (A
faster), while both A/A-before sessions were near zero. That direction is
consistent in the two outer-versus-inner session contrasts. It is a useful
observation, but four sessions do not isolate A/A placement from substantial
within-session movement and between-session latency shifts. No candidate
performance ranking is supported. All four individual comparisons report
`INCONCLUSIVE`.

## Execution and measurement windows

The four sessions ran in separate CLI processes, in the planned order, with
the same spec and trusted source. The sole runtime option was placement:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_1_warps8.json \
  --allow-trusted-local-source \
  --diagnostic-aa-placement before \
  --output artifacts/mi-timing-3/evidence-v1/s1-before
```

Sessions 2 and 3 used `after` and their corresponding output names;
session 4 used `before`. The diagnostic flag changes the phase order inside
each round. It does not change the timed block, candidate, input, reference,
warmup count, or decision rules. For each session, separate compile and
correctness work precedes 30 warmups per kernel and two unranked timing
controls. Eight rounds then contain one X/Y identical-A pair and one A/B
pair. Both pairs alternate their slot order each round. Thus every session
has 16 measured A/A blocks and 16 measured A/B blocks, each of exactly 100
eager preallocated GEMM wrapper calls. The two unranked controls also have
100 calls each. The saved `measurement_sequence` and per-block totals make
phase order and normalization replayable.

Each block uses the existing v3 window: `perf_counter` start, RAW start,
CUDA start event, 100 calls on the current stream, CUDA end event, host
submission timestamp, end-event synchronization, RAW end, then
`perf_counter` end. Event time brackets stream work. The enclosing host
windows include submission and synchronization. RAW and `perf_counter`
have slightly different timestamp-read boundaries, so they are reported
separately. Total block durations are retained before division by 100.
The optional RAW reader was available in every session. All five-window
preflights reported `NO_RESOLVED_DIFFERENCE`, with median RAW/perf ratios
between 0.999975 and 0.999987. None of the saved event windows exceeded
its enclosing host windows. This differs from the earlier MI-Timing-2
clock-rate episode; it does not prove that either host clock is universally
accurate.

All four sessions ran on the RTX 4060 Ti with driver 616.92, CUDA runtime
12.8, Python 3.13.11, PyTorch 2.11.0+cu128, and Triton 3.6.0. The spec SHA-256
was `5707026a020429a86a35bb0f922bac2c743450c896c5550cdfbe87773cb434cc`.
The trusted B source SHA-256 was
`bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
The retained A and B PTX hashes were identical across sessions. Compiler
metadata remained 24,576 shared-memory bytes for both kernels, with 121
registers/thread for A and 78 for B. No kernel source or compiler output
was changed for this study.

## Numerical gate and observed timing

Both kernels passed shape, FP16 dtype, finiteness, and the unchanged
PyTorch/cuBLAS `allclose` gate (`atol=rtol=0.01`) on seeds 0, 1, and 2 in
every session. Saved maximum absolute error was 0.0 for all checks.
The following values are **µs per GEMM operation**. A−B below zero favors
A. MAD is the median absolute deviation of the eight paired differences
within a session. The session, rather than each round, is the independent
comparison unit.

| Session | Placement | Event median A / B | Paired event A−B median; MAD; range | A-faster / B-faster rounds | A/A X−Y median; maximum absolute gap |
|---|---|---:|---:|---:|---:|
| 1 | before | 68.644 / 68.842 | +0.051; 1.273; −11.305 to +10.612 | 4 / 4 | −1.413; 14.177 |
| 2 | after | 69.559 / 74.296 | −2.542; 4.953; −16.804 to +5.622 | 6 / 2 | −2.148; 4.209 |
| 3 | after | 63.708 / 65.725 | −1.990; 2.129; −5.746 to +4.473 | 7 / 1 | −0.437; 5.243 |
| 4 | before | 68.298 / 68.597 | +0.068; 3.829; −18.555 to +7.688 | 4 / 4 | −3.164; 8.019 |

| Session | `perf_counter` median A / B | RAW median A / B | Paired `perf_counter` A−B median | Paired RAW A−B median |
|---|---:|---:|---:|---:|
| 1 before | 69.633 / 69.759 | 69.623 / 69.752 | −0.016 | −0.017 |
| 2 after | 71.078 / 75.010 | 71.067 / 75.001 | −2.976 | −2.967 |
| 3 after | 64.393 / 66.846 | 64.388 / 66.842 | −1.965 | −1.964 |
| 4 before | 69.032 / 69.267 | 69.020 / 69.260 | +0.491 | +0.489 |

The host clocks show the same broad paired pattern. Session 1's near-zero
event median changes sign at host precision, which reinforces that it has
no stable direction. The A/A maximum gaps are **descriptive only**. They
are not a statistical cutoff for this placement experiment. Their range
from 4.209 to 14.177 µs also shows that A/A does not give one transferable
noise value. A/B paired ranges reach 10–26 µs across sessions. A/A samples
local noise, but cannot be assumed to reproduce the noise or warm state
of the neighboring A/B pair.

The after-session paired medians are −2.542 and −1.990 µs; the before
medians are +0.051 and +0.068 µs. Yet the first-versus-last four-round
paired medians shift by about 6.3 µs in session 2 and 3.5 µs in session 3.
Individual A-event medians range from 63.708 to 69.559 µs between sessions;
B ranges from 65.725 to 74.296 µs. These session shifts are at least as
large as the apparent placement contrast. Session 2 has the widest paired
MAD (4.953 µs). The before sessions also contain large opposing outliers.
The ABBA session order counters a simple monotonic time trend, but two
replicates per condition cannot separate a placement effect from these
changes or from differences in background work.

The coarse `nvidia-smi` snapshots show temperatures from 41–50 °C and
graphics clocks from 210–2565 MHz at session boundaries. They are idle or
transitional snapshots, not clocks during the measured blocks. For example,
sessions 3 and 4 started with the same reported 2565 MHz graphics clock
but had different placement and timing profiles. The snapshots therefore
cannot identify a clock or cache mechanism. No cache residency was measured.
The normal benchmark's unranked controls were kept ahead of both conditions.

The earlier MI-Research-1 eight-warp comparison was also inconclusive.
This study explains one methodological uncertainty more clearly but does
not reclassify that historical result or establish an eight-warp speedup.
In particular, the after condition's paired signs do not show that removing
A/A from the immediate lead-in would produce a reproducible improvement.

## Evidence and verification

The [versioned evidence bundle](../artifacts/mi-timing-3/evidence-v1/)
contains the pre-run plan, four complete paired-output directories, all
three-clock raw block timings and exact operation counts, numerical records,
compiler/PTX captures, per-session manifests, an `analysis.json` summary,
eight pre/post GPU telemetry snapshots, and a top-level `SHA256SUMS` covering
82 files. The summary contains descriptive values only and preserves the
four individual `INCONCLUSIVE` decisions. It is derived from the raw records.
The original raw timing and historical evidence files were not edited.
The top-level checksum file has SHA-256
`4ddf503b7c6fa203094ff78e624f319520a2e62effa283024147e1a4555cbf4e`.

Verify the retained files and replay each session offline from the repo root:

```bash
(cd artifacts/mi-timing-3/evidence-v1 && sha256sum -c SHA256SUMS)
for session in s1-before s2-after s3-after s4-before; do
  .venv/bin/matmul-inspector-python experiment pair \
    --verify "artifacts/mi-timing-3/evidence-v1/$session"
done
```

The relevant CPU suite passed **69 tests** covering phase order, operation
counts, legacy behavior, invalid options, bundle replay, and tamper detection.
The historical MI-Timing-2 and MI-Research-1 eight-warp bundles also verified
offline without changing their results. The four GPU bundles
each verified as `PASS`, `INCONCLUSIVE`, eight rounds, and 17 hashed files.
The outer checksum covers the per-session manifests as well as every raw
artifact. The source tree was dirty because unrelated user work and this
diagnostic extension were present, so artifact and source hashes are the
reproducibility anchor.

## Recommendation and limits

**Keep the normal A/A-before workflow unchanged for now.** The narrowly
scoped diagnostic option permits a controlled placement check when an A/B
result is close to observed noise. Its placement and operation sequence are
machine-readable. This four-session study does not justify moving or
removing the normal A/A control. Treat small A/B differences as inconclusive
until they recur across independent sessions with stable timing conditions.

This result is limited to one workload, one pair of existing kernel
configurations, two sessions per condition, and one short measurement period.
Within-session order is balanced for A/B but the control position is the
experimental factor, so it is intentionally different. No mid-block clocks,
cache metrics, or independent background-work trace were collected. The
experiment distinguishes an observed directional pattern from a demonstrated
causal effect. It completes this bounded timing investigation without
starting a new optimization search.
