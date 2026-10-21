# HA-Schedule-5: measurement-calibrated execution schedule search

HA-Schedule-4 found that the frozen rank-8 recompute path can win during eager
execution even when its consumer kernels cost more. It removes one
materialization launch and its host/device orchestration boundary. Under CUDA
Graph replay, the eager advantage mostly disappears or reverses. Selective
Nsight counters did not establish lower physical DRAM traffic as the cause.
HA-Schedule-2's abstract HBM traffic therefore remains useful for describing
legal schedules, but not as a measured DRAM or latency predictor.

This milestone checks whether that measured mechanism improves *schedule
selection*. It does not fit a GPU latency regression, add decompositions,
modify CUDA kernels, or implement an autonomous search loop.

## Two-stage search boundary

The retained HA-Schedule-2 CP-SAT corpus supplies bounded fixed-order legal
schedules for conventional rank-8, Strassen rank-7, and frozen exact rank-8.
This command also builds the five HA-Schedule-1 manual policies for each DAG.
It deduplicates by semantic schedule hash. Every candidate is checked again
with the independent HA-Schedule-1 validator and evaluator at N=128, 256,
512, and 1024. The corpus contains 10, 12, and 17 distinct candidates per
size for the three algorithms. No new OR-Tools solve is claimed or needed to
rank this retained corpus. OR-Tools is optional and unavailable in the
validation environment used for this study.

The solver remains responsible for schedule legality and abstract capacity.
Calibration is an *offline ranking* stage. It does not change a schedule, its
hash, its legality, or a CP-SAT objective. Static ranking reproduces the
HA-Schedule-2 lexicographic order: temporary HBM traffic, recomputation
operations, then logical launch groups. The displayed hash only breaks exact
ties deterministically. Pareto membership still uses four static dimensions:
temporary traffic, recomputation operations, peak abstract local bytes, and
logical groups. A logical group is not a physical CUDA launch.

Each candidate retains its abstract traffic, peak ON_CHIP and combined local
bytes, recomputation count and operations, shared-form count, abstract HBM
input-form count, source policy, and hash. Only the four HA-Schedule-3 selected GPU
lowerings have an `expected_actual_launch_count`: conventional 9, Strassen 8,
rank-8 stored 10, and rank-8 recomputed 9. These mappings are approximate
relative to the abstract schedule. Their expected actual materialization
stage counts are 0, 0, 1, and 0 respectively. Other candidates have an unknown physical
launch count and `UNCALIBRATED_LOWERING` status. They remain in the candidate
set, static ranking, and Pareto diagnostics. A calibrated “best” is explicitly
the best **within the currently mapped subset**, not the best among every
legal schedule. Static-best over all legal schedules and static-best within
the mapped subset are both reported so that mapping coverage is not mistaken
for a measured improvement.

## Calibration records and uncertainty

`calibration_profiles.json` has one record for each retained size and each
execution mode, `EAGER` or `CUDA_GRAPH`. A record contains a content-hashed
profile ID, source artifact path and SHA-256 hashes, hardware/software
identity, dtype, session IDs, physical rank-8 launch counts, stage medians,
paired consumer delta, optional dummy-launch timing range, N=1024 compiler
register/shared-memory metadata and cubin hashes, paired complete-operation
ratios and bootstrap intervals, confidence, preference, and limitations.
The command verifies the complete HA-Schedule-1 through HA-Schedule-4 hash
manifests. Missing, altered, wrong-hardware, wrong-version, wrong-size, or
wrong-mode profile inputs are errors. Profiles are exact-size only. There is
no interpolation or extrapolation.

Stage CUDA-event intervals can contain enqueue gaps. They are not assumed
additive, and their sum is not a complete-operation estimate. The one-word
dummy launch gives context-dependent scale, not a per-launch coefficient.
Nsight replay counters and static SASS counts describe different evidence
layers. No term is converted to a predicted GPU latency.

`calibrated_schedule_cost` is an ordinal 0 or 1 for the *known rank-8 pair*:
0 for the mode/size-supported preference, 1 for the other member. A parity,
weak, or unstable profile gives both 0 and a tied calibrated rank. This is a
partial-order rule from retained paired confidence intervals, not a numeric
cost model. Conventional and Strassen each have only one mapped lowering, so
they are recorded as supported anchors with no comparative calibration.
Their rank-8 profile is never used to infer a benefit for another algorithm.

Eager confidence uses separate HA-Schedule-4 sessions A and B. Both intervals
must exclude parity in the same direction for a preference. Overlapping
intervals give `STRONG`; disjoint but same-direction intervals give
`MODERATE`. Disagreement or parity is retained as `WEAK`, `PARITY`, or
`UNSTABLE`. Graph evidence has only session A, so an interval excluding
parity is at most `MODERATE`. Eager launch penalties are never copied into
graph mode. Graph preference uses the graph-replay paired interval directly.

A leave-one-session-out diagnostic uses only A's paired eager interval to
predict B's direction, then reverses the roles. It does not optimize any
coefficient and does not call the held-out session training data. Graph mode
has no independent retained session and cannot make the same held-out claim.

## Reproduce

The command refuses an existing output directory. A CPU-only structural
smoke is available; it does not establish the final decision gate:

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_calibration.py \
  --output /tmp/ha-schedule5-smoke --skip-gpu

PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_calibration.py \
  --output artifacts/tensor-game/hardware-aware-schedule-calibration-v4
```

The full command starts one fresh GPU subprocess. At each retained size it
checks both production rank-8 paths against PyTorch on five fixed seeds,
checks graph replay output, then interleaves eager and graph CUDA-event pairs.
Defaults are 10 warmups, 40 operations per round, and 11 rounds. The
validation file retains per-round summaries, paired intervals, correctness,
process provenance, and raw event rows separately. Only already implemented
lowerings are benchmarked. Fresh disagreement is recorded; it downgrades or
invalidates confidence rather than altering HA-Schedule-4 evidence.

Artifacts include config, provenance, source manifests, profiles, all
candidate schedules and features, three rankings, static Pareto/rank
comparison, leave-one-session-out checks, fresh GPU validation, decision
summary, notes, and a SHA-256 manifest. No cubin or profiler binary is
committed.

## Results and decision

The final retained run is `hardware-aware-schedule-calibration-v4`, produced
from implementation commit `af53dce` on the same RTX 4060 Ti, CUDA 12.8,
PyTorch 2.11.0+cu128, and Triton 3.6.0 as HA-Schedule-4. It checked 156
size-specific legal candidates: 10 conventional, 12 Strassen, and 17 frozen
rank-8 per size. Only four per size have validated GPU-lowering templates.
No legality failure or source-hash mismatch occurred. Static-best **over all
legal candidates** is often not executable with the current lowering. Its
hash prefixes are:

| Algorithm | N=128 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|
| Conventional | `11b57877` | `3e5b6637` | `0a9ffe98` | `0dfaa6df` |
| Strassen | `0bb49cb2` | `c1caff99` | `4494d457` | `2aa01b2a` |
| Frozen rank-8 | `9ee979cc` | `1406976b` | `07199237` | `c5b9945` |

These are identifiers, not measured performance rankings. Full hashes and
feature vectors are in `ranking_comparison.json` and
`candidate_schedules.jsonl`. For conventional and Strassen, the sole mapped
reference lowering is the only calibrated anchor in either mode. There is no
measured choice between alternative schedules for those algorithms. For
frozen rank-8, the mapped subset is the HA-Schedule-3 stored/recomputed pair.

| N | Static-best mapped pair | Retained eager rank | Retained graph rank | Post-validation eager rank | Post-validation graph rank |
|---:|---|---|---|---|---|
| 128 | recompute | recompute (`STRONG`) | parity | recompute (`STRONG`) | parity (`WEAK`) |
| 256 | recompute | recompute (`STRONG`) | parity | recompute (`STRONG`) | parity |
| 512 | recompute | recompute (`MODERATE`) | stored (`MODERATE`) | recompute (`MODERATE`) | stored (`MODERATE`) |
| 1024 | recompute | parity | stored (`MODERATE`) | parity | stored (`MODERATE`) |

At N=256 the static pair has 1,572,864 versus 1,474,560 abstract temporary
HBM bytes (stored/recomputed), 0 versus 16,384 abstract recomputation
operations, and eight logical groups for both. Its physical lowerings have
10 versus 9 launches and one versus zero preparation kernels. The abstract
stored schedule has eight HBM input forms, but only one becomes the separate
physical preparation kernel in the measured lowering. This is why the
calibrated choice uses the validated lowering map instead of counting abstract
groups as launches.

Across the four frozen rank-8 sizes, seven pair-rank entries change between
static and the two retained-mode rankings; the count includes a tie replacing
a strict rank. Eager and graph mapped selections differ at all four sizes
if a parity set is treated as different from a single preferred schedule.
Static and eager mapped best agree at three sizes; static and graph mapped
best agree at none. These counts are **within the two mapped rank-8 choices**.
They do not establish that a calibrated winner beats every unmapped static
candidate.

All eight eager leave-one-session-out checks agree in *direction*: A predicts
B and B predicts A at all four sizes. The direction is recompute at
128–512 and parity at 1024. This does not validate graph mode, which has
only one retained HA-Schedule-4 session. It also does not erase the absolute
latency and N=512 effect-size variation between the two eager sessions.

### Fresh GPU validation and audit runs

The final run used a fresh process, five correctness seeds per eager path,
one checked graph input per path and size, and 11 interleaved rounds of 40
operations after 10 warmups. All 48 recorded correctness checks passed. Both
modes ran at all four sizes. Ratios are paired round-median
stored/recomputed ratios. Intervals are bootstrap 95% intervals over rounds;
above 1 favors recomputation.

| N | Fresh eager ratio (95% interval) | Fresh graph ratio (95% interval) | Validation outcome |
|---:|---:|---:|---|
| 128 | 1.067 (1.036–1.144) | 1.020 (1.010–1.046) | eager confirmed; graph shows a new weak recompute direction from retained parity |
| 256 | 1.105 (1.034–1.123) | 1.000 (0.825–1.024) | eager recompute; graph parity |
| 512 | 1.149 (1.031–1.166) | 0.982 (0.972–0.986) | eager recompute; graph stored |
| 1024 | 1.042 (0.989–1.073) | 0.972 (0.953–0.976) | eager parity; graph stored |

Three earlier fresh runs (`v1`–`v3`) are retained as audit evidence. Their
reporting code was superseded by later confidence/provenance fixes; their raw
GPU timings and correctness records were not rewritten. All four fresh runs
agree on eager recompute at 128 and 256, graph parity at 256, and graph stored
at 512 and 1024. Their edge cases vary: eager 512 includes parity in `v1`
and `v3`; graph 128 is parity in `v1`/`v2` but favors recompute in `v3`/`v4`;
eager 1024 is parity in three runs and weakly favors stored in `v2`. This
variation is a reason to keep parity and weak categories. It is not a reason
to refit the retained HA-Schedule-4 profile after seeing fresh data.

The final validated profiles link the untouched retained profile ID to the
fresh validation file hash. A fresh contradiction would become `UNSTABLE`;
an unconfirmed direction becomes `WEAK` and loses its hard pair preference.
The final `v4` run downgrades graph 128 to `WEAK` because the retained graph
interval included parity, even though this fresh interval favors recompute.
Earlier runs illustrate the same downgrade rule at eager 512. No calibration
is transferred to a different GPU, software version, dtype, or matrix size.

Primary decision: `MODE_SPECIFIC_SIGNAL`. Secondary:
`CALIBRATED_SEARCH_SIGNAL`, limited to the mapped rank-8 pair. The robust
N=256 outcome is the useful control: static and eager choose recompute, while
graph execution gives parity. At 512, graph favors stored and eager favors
recompute in the final run, although eager significance varied across fresh
runs. This supports mode-specific *ranking*, not a universal GPU cost formula.

Closed-loop readiness: `NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE`. Candidate
generation, legality checks, static characterization, correctness gates,
bounded GPU benchmarking, evidence hashing, confidence downgrade, and local
pair reselection are present. But 35 of 39 legal schedules per size lack a
validated GPU-lowering map, graph calibration has only one retained session,
and edge-size timing confidence varies. A prototype agent should wait for
broader lowering coverage and independent mode-specific validation. No agent
loop is implemented here.
