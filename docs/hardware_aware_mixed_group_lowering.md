# HA-Schedule-7: mixed GEMM/reconstruction lowering

HA-Schedule-6 had 30 structurally unsupported legal schedules per size. Of those,
29 contain at least one group combining a GEMM product with reconstruction.
At N=128, they contain 212 mixed-group occurrences with 48 distinct signatures when
normalized by form, product, accumulation locations and output count. This is
not one homogeneous fusion problem. Some groups combine HBM form preparation,
HBM product writes, multiple reconstruction targets, or other work. The
remaining unsupported schedule without a mixed group is a Strassen input-form
fusion. The same 39 schedule structures recur at N=128, 256, 512, and 1024
under different size-specific hashes.

This milestone adds one deliberately bounded family,
`FUSED_GEMM_RECONSTRUCTION`. It does not change decompositions, CP-SAT search,
objectives, candidate selection, or the existing HA-6 mappings. One template
covers nine previously unsupported hashes per size: three conventional,
three Strassen, and three frozen rank-8. They include policy-identity variants;
the count is schedule hashes, not independent algorithms.

## Reproduce

The standalone runner refuses an existing output directory. It checks the
retained HA-5 corpus manifest and uses the 156 original schedule hashes.

```bash
PYTHONPATH=src:. python scripts/run_hardware_aware_mixed_group_lowering_study.py \
  --output /tmp/ha7-cpu --skip-gpu
```

That command gives structural classifications only. It does not validate new
exact mappings. The retained GPU study used the CUDA-enabled Python environment:

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_mixed_group_lowering_study.py \
  --output artifacts/tensor-game/hardware-aware-mixed-group-lowering-v1
```

The retained evidence is in
[`artifacts/tensor-game/hardware-aware-mixed-group-lowering-v1`](../artifacts/tensor-game/hardware-aware-mixed-group-lowering-v1).
It includes configuration, source hashes, every lowering plan and
classification, seed-level correctness, profiler topology, representative
compiler resources, bounded smoke, and a file hash manifest. A derived-reason
label correction after the GPU run was applied with `--refresh-derived`.
That mode first checks the artifact manifest and verifies every retained GPU
plan hash against the current classifier. It preserves correctness, topology,
resource, and smoke records. It does not rerun CUDA.

## Template and exactness

The CPU-only classifier validates the schedule and checks each launch group.
An eligible mixed group has exactly one `ON_CHIP` GEMM product, one or two
immediate accumulation targets, at most two local direct input forms, and no
co-located HBM form preparation. Input forms are either previously
materialized FP16 HBM values or local forms computed directly from the four
source blocks. Local form values are rounded to FP16 before the dot. A group
that requires nested recomputation, an HBM product, more than two targets,
duplicate output targets, or nonlocal work is unsupported.

Each plan keeps the full schedule and decomposition hashes, a stable plan
hash, an ordered launch sequence, fused GEMM ID, input sources, local forms,
target accumulation IDs and exact decomposition coefficients, prior partial
IDs, initialize/read-modify-write mode, partial FP32 or final FP16 write mode,
and expected kernel role. Existing `MATERIALIZED_DAG` and legacy approximate
plans are unchanged. The `mixed_lowerability_query` can reject a candidate
without importing CUDA. The new family creates no new approximate mapping.

The reusable Triton kernel computes one tiled GEMM and applies its accumulator
to up to two distinct output blocks in that same launch. The fused product has
no standalone HBM allocation. Prior partial sums are read from explicit FP32
buffers. A non-final result writes an FP32 partial buffer. A final result
writes the appropriate FP16 quadrant. Launches run in schedule order on one
CUDA stream. Each tile owns disjoint elements, so no atomic accumulation or
cross-kernel race is assumed. Singleton HBM groups continue to reuse the
HA-6 kernels. The runtime allocates only schedule-declared HBM intermediates.

`EXACT_MAPPING` is a structural claim gated by actual launch topology and
fixed-tolerance full-operation correctness. A compiler may still choose
register, shared-memory, and machine-code behavior below the modeled
source-level boundary. PTX/SASS hashes document compiled code; they do not
prove physical traffic. No product HBM workspace is hidden for fused groups.

## Correctness and coverage

Each new mixed mapping was tested on seeds 11, 23, and 37 at FP16 input,
FP32 product accumulation, FP16 output with the established `rtol=0.005`,
`atol=0.05`. One representative per algorithm at N=256 also used seeds 41
and 53. That is 114 new correctness checks. Every new mapping was profiled
for actual CUDA kernel count and order. All 36 mixed mappings matched their
plans and had zero fused-product HBM workspaces. The established HA-6
validation is carried forward by schedule hash; it was not rerun here.

| N | Before exact passing | New structural exact | New mixed passing | After exact passing | Approximate passing | Unsupported | Incorrect exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5/39 | 9 | 9 | 14/39 | 4/39 | 21/39 | 0 |
| 256 | 5/39 | 9 | 9 | 14/39 | 4/39 | 21/39 | 0 |
| 512 | 4/39 | 9 | 6 | 10/39 | 4/39 | 21/39 | 4 |
| 1024 | 4/39 | 9 | 6 | 10/39 | 4/39 | 21/39 | 4 |

The 512/1024 incorrect counts include the one retained HA-6 materialized
Strassen failure and three new mixed Strassen failures at each size. All three
new mixed Strassen hashes at each large size failed all three seeds. They are
`LOWERING_INCORRECT`, excluded from passing coverage and benchmark decisions.
The tolerance was not changed. At 128/256, Strassen has four exact-passing
hashes (one old, three new). Conventional and rank-8 each have five
exact-passing hashes per size (two old, three new). Across 156 records, exact
passing rises from 18 to 48 (11.5% to 30.8%). All 16 prior approximate
mappings remain approximate and passing. No approximate mapping was promoted.

## Resource and execution evidence

Representative N=256 fused kernels for all three algorithms compiled and
executed. Each selected mixed kernel used 40 registers per thread and 4,096
bytes of static shared memory. The prior representative HA-6 GEMM reported
the same 40 registers and 4,096 bytes. These are specific compiled variants,
not a bound on every fusion. The retained manifest records TTIR, TTGIR, PTX,
cubin, and SASS hashes where available, but does not commit cubin binaries.
No compilation or resource rejection occurred for the tested sizes.

A five-run CUDA-event execution smoke at N=256 includes one correct mixed
path per algorithm and the existing stored/recomputed rank-8 anchors. These
unpaired, Python-orchestrated samples only show that paths execute and timing
works. They are not comparative ranking evidence. More coverage does not
imply faster execution. Fusion can increase register or shared-memory
pressure, and a correct exact lowering can be slow.

## Remaining gap and decision

Twenty-one of 39 legal schedules per size remain structurally unsupported.
Reasons overlap. Across all four sizes, 72 remaining records require a
partial-output lifetime pattern outside this template, 68 place an HBM
product in a mixed group, 64 co-locate HBM form materialization, 36 require
nonlocal reconstruction, and 20 exceed the two-target reconstruction bound.
There are additional duplicate-target and non-operand-form patterns. The
most frequent single reason is partial-output lifetime (18 per size), but
most such schedules also need other capabilities. Only two unsupported
records per size have co-located HBM form materialization as their sole
reported blocker. Thus the backend is not one simple feature away from broad
exact coverage.

The primary result is `LOWERING_BACKEND_STILL_FRAGMENTED`. The new family
clearly helps, but 48/156 exact-passing coverage is below the practical 50%
threshold and below 20/39 at every size. Closed-loop readiness remains
`NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE`. Generation, legality, static metrics,
calibration, eager/graph separation, lowerability queries, plan generation,
GPU gating, bounded benchmarking, evidence retention, and candidate
reselection exist. Broad faithful backend coverage does not.

The next backend capability to study is a bounded **HBM-backed mixed-group
contract**: define when an HBM product and an HBM partial/final output can be
produced in one logical launch without an unsynchronized grid dependency or
hidden extra launch. The overlapping blocker counts make this a better next
question than assuming either label alone unlocks 17–18 schedules per size.
Co-located HBM form preparation is a separate issue. This milestone does not
validate the full schedule-search loop or authorize a closed-loop agent.
