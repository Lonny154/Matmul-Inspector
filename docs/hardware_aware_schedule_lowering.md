# HA-Schedule-6: bounded schedule-family lowering

HA-Schedule-5 revalidated 39 legal schedules per size but only four had a GPU
mapping. Those four mappings were already **approximate**: the physical
implementation writes all rank products to global memory and reconstructs in
a later kernel, while the abstract schedules put some product and
reconstruction work in the same launch group. This milestone adds a narrow,
deterministic bridge from schedules to executable templates. It also repeats
the frozen rank-8 CUDA Graph comparison in a new process. It does not change
the decomposition, CP-SAT objective, or HA-3 production kernels.

## Reproduce

The command refuses an existing output directory. The CPU-only form generates
structural classifications but makes no new GPU-validation claim:

```bash
PYTHONPATH=src:. python scripts/run_hardware_aware_schedule_lowering_study.py \
  --output /tmp/ha6-cpu --skip-gpu
```

The retained GPU run used the CUDA-enabled Python environment on the same RTX
4060 Ti as HA-3 through HA-5:

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_lowering_study.py \
  --output artifacts/tensor-game/hardware-aware-schedule-lowering-v1
```

The runner checks the HA-5 source hash manifest, reconstructs each DAG and
schedule, validates the schedule, and records its full schedule hash. It starts
separate GPU processes for lowering validation and graph replication. The
graph worker reuses the unchanged HA-5 protocol: five correctness seeds per
rank-8 path, ten warmups, 40 operations per round, and eleven interleaved
rounds. It retains raw event rows. The lowering worker uses three fixed seeds
and the prior `rtol=0.005`, `atol=0.05` tolerance for each mapped hash.
The retained run also passed four `--audit-graph-artifact` inputs from
preliminary HA-6 runs made during development. The runner verified their
manifests, copied their graph summaries and raw event rows into
`graph_development_audit/`, and included them in the confidence aggregation.
They are preserved rather than silently discarded after implementation
changes. The command above runs one new session without those optional audit
inputs; its session count will differ from the retained multi-session summary.

## Classifier and lowering plan

`ScheduleLoweringPlan` is a bounded mapping record, not a general compiler IR.
It contains the exact schedule and decomposition hashes, family, ordered
launch roles/kernels, materialized/recomputed/fused values, reconstruction
strategy, expected launch count, mapping status, reasons, version, static and
calibration-feature hashes, and its own content hash. The classifier reads
the schedule's groups and locations. It never infers lowerability from equal
metric vectors. The CPU-only `lowerability_query` and capability manifest can
reject an unsupported schedule before touching CUDA. A separate correctness
gate decides whether a mapped schedule may be benchmarked.

There are two registered families:

| Family | Physical template | Claim |
|---|---|---|
| `MATERIALIZED_DAG` | One Triton launch for each singleton HBM form, product, accumulation, or output group, in abstract order | Exact mapping when topology agrees and correctness passes |
| `LEGACY_PRODUCT_RECONSTRUCTION` | The four HA-3 selected lowerings, keyed by their full retained schedule hashes | Approximate mapping only |

The materialized template is parameterized by decomposition factors, input
forms, product IDs, reconstruction coefficients, and output blocks. It reuses
one form kernel, one GEMM kernel, one accumulator kernel, and one output kernel
across the three algorithms. A form is explicitly written as FP16, then its
consuming GEMM reads that buffer. Products and accumulation intermediates are
FP32 HBM values. The final output is FP16. The template does not fuse a
multi-operation group or promise physical register/shared-memory residency for
abstract `ON_CHIP` values. The compiler may still alter actual memory
behavior below these source-level boundaries. The saved PTX and SASS for four
representative roles are separate compiler and machine-code evidence, not
measured traffic or a speedup claim.

`EXACT_MAPPING` means the generated execution follows the modeled
materialization and recomputation structure. `APPROXIMATE_MAPPING` remains a
useful exploratory implementation but is **not** execution of the exact
searched schedule. `UNSUPPORTED` means no current template preserves the
distinction. The four legacy mappings stay approximate even though their
outputs and physical topology pass validation.

## Coverage and correctness

The 39 normalized structures repeat across N=128, 256, 512, and 1024 with
distinct size-specific hashes. Each size has 10 conventional, 12 Strassen,
and 17 frozen rank-8 schedules. Structural classification gives five exact
template mappings, four existing approximate mappings, and 30 unsupported
hashes per size. Two of the five exact hashes are policy-identity duplicates
of another fully materialized structure: one conventional and one rank-8.
They share the same template; they are not independent GPU algorithms.

| Algorithm | Legal / size | Structural exact | Existing approximate | Unsupported | Validated exact at 128/256 | Validated exact at 512/1024 |
|---|---:|---:|---:|---:|---:|---:|
| Conventional | 10 | 2 | 1 | 7 | 2 | 2 |
| Strassen | 12 | 1 | 1 | 10 | 1 | 0 |
| Frozen rank-8 | 17 | 2 | 2 | 13 | 2 | 2 |
| Total | 39 | 5 | 4 | 30 | 5 | 4 |

The new materialized Strassen path fails the fixed correctness tolerance at
N=512 and 1024 for all three seeds. Its maximum absolute errors reach 0.09375
and 0.125 respectively. Explicit FP16 form materialization adds rounding
relative to the previously fused form path. This is a plausible numerical
cause, not proof of a particular instruction-level cause. The failures are
classified `LOWERING_INCORRECT` and excluded from timing. No tolerance was
widened. Across all 156 size-specific hashes, 18 exact mappings pass the
three-seed gate (11.5%), 16 approximate mappings pass (10.3%), two exact
template mappings fail, and 120 are structurally unsupported. The previous
mapped set had 16 approximate hashes and no exact ones.

The profiler recorded an actual kernel sequence for all 36 structurally
mapped hash/size records, including the two incorrect cases. All 36 launch
counts and kernel orders matched their plans. Exact materialized schedules
use 20, 33, or 30 launches for conventional, Strassen, or rank-8. The legacy
paths retain 9, 8, 10, and 9 launches. A topology match does not rescue a
numerically incorrect schedule. Seven N=256 candidates received a five-run
CUDA-event execution smoke: the three materialized controls plus the four
legacy paths. Those unpaired samples are not a performance ranking.

The largest unsupported gap is mixed GEMM/reconstruction grouping. Across the
four sizes, 120 unsupported records need a multi-operation group template,
116 include fused GEMM/reconstruction, 100 use unsupported ON_CHIP fusion,
64 co-locate HBM form preparation with other work, and 24 need recomputation.
Reasons overlap. The current template architecture cannot faithfully realize
most of the fixed-order solver's mixed groups; adding names for them would
not increase validated coverage.

## Independent graph session

The new graph worker ran in a fresh process and passed all 48 eager/graph
correctness checks. Ratios below are paired round-median **stored/recomputed**
ratios. Above one favors recomputation. Marginal medians need not yield the
same ratio because pairing and summary operations are different.

| N | Stored / recomputed graph median (µs) | Paired ratio (95% bootstrap interval) | New direction | Ten-session confidence |
|---:|---:|---:|---|---|
| 128 | 21.376 / 21.152 | 1.014 (1.008–1.045) | Recompute | `PARTIAL_AGREEMENT` |
| 256 | 32.352 / 32.400 | 0.998 (0.991–1.005) | Parity | `PARITY` |
| 512 | 71.552 / 72.704 | 0.982 (0.975–0.984) | Stored | `PARTIAL_AGREEMENT` |
| 1024 | 381.936 / 392.192 | 0.974 (0.973–0.974) | Stored | `AGREES` |

The ten sessions are HA-4 graph session A, HA-5 v1–v4 fresh validations, four
HA-6 development audit runs, and the final HA-6 run. Each retains its own
interval and provenance in `graph_evidence_summary.json`; no intervals or
medians are pooled. N=256 is parity in all ten. N=1024 favors stored in all
ten. N=512 has nine stored directions and one development-audit parity
interval, so its confidence is partial agreement rather than a unanimous
stored result. N=128 mixes four parity and six weak recompute directions.
The final graph round-median coefficients of variation were about 2% at 128,
and 6%/14% for recomputed/stored at 256. Earlier audit sessions were more
variable. None of this refits old profiles
or transfers an eager launch benefit into graph mode. It is specific to this
GPU, software stack, dtype, and these four sizes.

## Readiness and limitations

The primary decision is `LOWERING_COVERAGE_STILL_LIMITED`; closed-loop status
remains `NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE`. Candidate generation, legality,
static characterization, mode-specific calibration, deterministic
lowerability queries, automatic execution of the supported family, three-seed
correctness gates, bounded GPU smoke, profiler/compiler capture, failure
records, provenance, confidence downgrade, and local pair reselection are
available. But faithful passing coverage is only four or five of 39 per size,
far below the 20/39 target. The mixed-group fusion capability is the main
remaining blocker. This milestone does not implement an agent loop.

Higher lowering coverage does not imply better schedules. A lowerable
schedule need not be fast. The materialized template deliberately uses many
launches and is a correctness/architecture control, not an optimization.
Abstract HBM/ON_CHIP counts are not measured DRAM bytes or occupancy. PTX is
not SASS, and static SASS counts are not dynamic execution counts. Any future
agent must remain bounded, correctness-gated, provenance-aware, and subject
to explicit human review before broad experiments.
