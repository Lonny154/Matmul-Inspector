# HA-Schedule-8: intentional HBM value lifetimes

HA-Schedule-7 raised exact-passing coverage to 48/156, but left 21 legal
schedules per size structurally unsupported. All 21 carry a partial output
across execution groups. Eleven also carry an HBM GEMM product, and ten carry
an HBM prepared form. These counts overlap. At N=128, the four cross-group
value-class combinations are partial-only (4 schedules), form+partial (6),
product+partial (7), and all three (4). Their multi-operation groups have 54
distinct ordered shapes. This is why HBM materialization must be represented
as an intentional schedule choice, not treated only as a fallback after fusion
fails.

The new `HBM_BACKED_MIXED_DAG` family supports one GEMM per group, up to two
direct reconstruction targets, co-located form writes, intentional product
writes, and FP32 partial-output chains. It maps 11 of the 21 previously
unsupported hashes per size: one conventional, five frozen rank-8, and five
Strassen. This is a bounded template, not arbitrary DAG code generation.
Existing HA-6 materialized and approximate plans and HA-7 fused plans retain
their prior plan hashes and physical lowerings. No schedule-search objective,
decomposition, or candidate corpus changed.

## Reproduce

The command refuses an existing output directory and checks the retained
HA-5 and HA-7 hash manifests before classification.

```bash
PYTHONPATH=src:. python scripts/run_hardware_aware_hbm_lifetime_lowering_study.py \
  --output /tmp/ha8-cpu --skip-gpu
```

That form produces structural and reference-transition evidence only. It
does not claim new GPU-correct exact coverage. The retained GPU command was:

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_hbm_lifetime_lowering_study.py \
  --output artifacts/tensor-game/hardware-aware-hbm-lifetime-lowering-v1
```

The [retained artifact](../artifacts/tensor-game/hardware-aware-hbm-lifetime-lowering-v1)
contains all 156 schedule hashes, plans, lifetimes, buffers, seed-level
correctness, profiler topology, runtime buffer audits, compiler-resource
samples, bounded execution smoke, selected-schedule analysis, decisions, and
a SHA-256 file manifest. Prior HA-7 validation statuses are joined by full
schedule hash; old GPU paths were not remeasured for this milestone. The
`launch_topology.jsonl` file holds 44 new profiler captures.
`all_mapped_launch_topology.jsonl` joins them with 72 hash-verified HA-6/7
captures, with each record's source identified.

## Value and buffer contract

Each intentional HBM value has an `HBMValueLifetime` record with its class,
producer group and launch role, ordered consumer groups and nodes, typed read
semantics, first/final consumer, dtype, bytes, write semantics, state,
schedule hash, plan hash, and DAG provenance. The four classes are:

| Class | Write | Later cross-group read | Meaning |
|---|---|---|---|
| `PREPARED_FORM` | `WRITE_NEW` | `READ_FORM` | FP16-rounded linear input form |
| `GEMM_PRODUCT` | `WRITE_NEW` | `READ_PRODUCT` | FP32 rank-product block |
| `PARTIAL_OUTPUT` | `WRITE_PARTIAL` or `UPDATE_PARTIAL` | `READ_PARTIAL` | FP32 ordered reconstruction state |
| `FINAL_OUTPUT` | `WRITE_FINAL` | none | completed FP16 output quadrant |

A producer and consumer in the same launch use `LOCAL_FORWARD`; that does
not claim an HBM reload. A later launch must read the recorded HBM buffer.
The CPU audit reconstructs all DAG consumers independently. It rejects
missing, extra, mistyped, or out-of-order reads. It also checks that launch
writes and the plan's materialized values match the schedule exactly.

Every HBM intermediate gets one stable, full-block logical buffer in v1.
There is no aggressive reuse. Four `FINAL_OUTPUT` value records refer to
disjoint quadrants of one FP16 output matrix. The buffer plan records first
write and last use. Overlapping reuse is rejected unless final-output
quadrants are disjoint. The runtime audit compares every planned intermediate
ID with the actual allocated logical buffers and checks shape and dtype.
This is logical buffer accounting, not a claim about cache residency or
measured DRAM transactions.

## Partial-output state and physical lowering

For each output block, the CPU reference follows the canonical decomposition
term order and checks an `UNINITIALIZED → PARTIAL → FINAL` state chain. It
records the contribution IDs, coefficients, and deterministic synthetic
intermediate values. Every expected contribution must occur exactly once.
The output is finalized once and cannot be updated afterward. A first HBM
reconstruction writes a valid partial; a later one reads its predecessor and
writes an updated FP32 partial. A final contribution writes the FP16 output.
If the abstract group explicitly puts an HBM partial and `OUTPUT` together,
one kernel writes both the FP32 partial and final FP16 quadrant. It does not
silently eliminate the intentional partial value. No atomics or concurrent
same-block updates are used; launches execute in schedule order on one CUDA
stream.

The reusable GEMM-group Triton kernel computes a tiled product and can also
store a prepared form, store the FP32 product, and apply up to two direct
reconstruction targets. For an HBM form produced in the same group, one CTA
owns each form tile's HBM write. Other GEMM CTAs compute their operand tiles
locally. No CTA reads another CTA's newly written form in the same grid.
Later groups load the materialized form. A second bounded kernel handles a
single accumulation plus final output without a GEMM. Singleton HBM groups
reuse the HA-6 kernels. Groups with earlier independent reconstruction,
unrelated forms, duplicate targets, or larger fan-out remain unsupported.

An `HBMBackedPlan` wraps the unchanged `ScheduleLoweringPlan`. Its final hash
includes launch topology, typed lifetimes, buffers, and contribution order;
each lifetime links back to that hash. This avoids changing old plan hashes
or creating a circular hash dependency. Structural `EXACT_MAPPING` becomes
passing coverage only after the CPU transition audit, GPU buffer audit,
fixed-tolerance correctness, and actual launch-topology check all pass.

## Correctness and coverage

New mappings use FP16 inputs, FP32 product accumulation, and FP16 output.
Each was checked against PyTorch matmul on seeds 11, 23, and 37 with the
established `rtol=0.005`, `atol=0.05`. One representative per algorithm at
N=256 also used seeds 41 and 53: 138 new full-operation checks in total.
All 176 CPU reference-transition checks passed. All 44 new mappings matched
their profiled launch counts and kernel order, and all 44 runtime logical
buffer audits passed.

| N | HA-7 exact passing | New structural exact | New passing | HA-8 exact passing | Approximate passing | Unsupported | Incorrect exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 14/39 | 11 | 11 | 25/39 | 4/39 | 10/39 | 0 |
| 256 | 14/39 | 11 | 11 | 25/39 | 4/39 | 10/39 | 0 |
| 512 | 10/39 | 11 | 6 | 16/39 | 4/39 | 10/39 | 9 |
| 1024 | 10/39 | 11 | 6 | 16/39 | 4/39 | 10/39 | 9 |

Conventional now has six exact-passing hashes per size. Frozen rank-8 has
ten. Strassen has nine at 128/256 and none at 512/1024. The large-size
incorrect count includes four retained HA-6/7 Strassen mappings plus five
new HBM-backed ones per size. Each new large-size Strassen mapping failed all
three seeds. The tolerance was not changed, and these paths are excluded
from passing coverage and smoke timing. This recurring numerical issue is
separate from the HBM lifetime contract. The new path passing at small sizes
does not resolve the earlier issue. Across 156 schedule-size records,
exact-passing coverage rises from 48 to 82 (30.8% to 52.6%). The 16 prior
approximate mappings remain approximate and passing. No new approximate
mapping was added.

## Selected schedules, resources, and smoke

Applying the new correctness results to the frozen HA-5 *static* rankings
gives exact coverage of 8/12 top-1 selections, 21/36 top-3 selections, and
46/108 static Pareto hashes across algorithms and sizes. These are
schedule-hash counts, not measured performance wins. HA-5 eager and graph
calibration ranked only the older measured subset. Their recorded top-1 sets
have 0/13 and 0/14 exact-passing hashes, respectively, because those old
selected mappings remain approximate. The new lowerings have **not** been
calibrated or ranked in eager or graph mode. It would be wrong to infer their
mode-specific preference from static rank or the old rank-8 pair.

Representative N=256 HBM-group kernels compiled and executed with these
reported resources: conventional 40 registers/thread and 4,096 bytes static
shared memory; frozen rank-8 40 and 6,144; Strassen 44 and 8,192. The
artifact retains bounded TTIR/TTGIR/PTX/SASS text and cubin hashes, not
cubin binaries. No tested HBM mapping hit a compiler resource rejection.
Compiler metadata for one specialization is not a resource bound for all
variants, and PTX is not final SASS. The five-run CUDA-event smoke includes
one correct new path per algorithm at N=256 plus the conventional and
stored/recomputed rank-8 anchors. It checks execution and timing plumbing
only. The samples are unpaired and Python-orchestrated, not ranking evidence.
Higher lowering coverage is not a performance result; a correct HBM-backed
schedule can still be slow.

## Remaining semantics and decisions

Ten of 39 schedules per size remain unsupported. Reasons overlap. Across
four sizes, 36 have reconstruction work before the group's GEMM, 36 have
nonlocal reconstruction, 24 have nonlocal output work, 20 exceed the
two-target reconstruction bound, 12 have unrelated form work, and eight
have duplicate output targets. At N=128 the ten remaining schedules form
six distinct blocker combinations; nine have multiple interacting blockers.
The retained one-GEMM solver bound means there is no multi-GEMM same-group
case in this corpus. Schedule legality also rules out cross-group `ON_CHIP`
lifetimes and cycles. This study introduces no unsafe concurrent partial
updates or in-place read/write aliases.

The primary HA-8 decision is `HBM_LIFETIME_COVERAGE_UNLOCKED`: one bounded
contract adds 34 correctness-passing exact records and passes 50% overall.
The backend decision is nevertheless `GENERAL_LOWERING_IR_NEEDED`. The
remaining groups combine pre-GEMM reconstruction, nonlocal work, unrelated
forms, and fan-out in several shapes. More one-off templates would be hard
to maintain. The recommended next step is to design a small general
schedule-to-kernel IR before growing the template set. This milestone does
not implement that IR.

Closed-loop readiness remains `NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE`. Search,
legality, static accounting, deterministic plans, lowerability queries,
correctness gates, evidence retention, and benchmark plumbing exist. But
important calibrated selections are still approximate, the new exact
families lack mode-specific calibration, large Strassen mappings fail
numerically, and ten schedules per size need a more general backend. A
future loop must benchmark only correctness-passing mappings. A static HBM
lifetime does not guarantee a particular physical cache path; compiler and
runtime behavior can change actual placement and cost.
