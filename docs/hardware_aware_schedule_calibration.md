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
  --output artifacts/tensor-game/hardware-aware-schedule-calibration-v1
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

To be completed after a committed implementation and fresh GPU validation.
