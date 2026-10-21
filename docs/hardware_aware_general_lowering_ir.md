# HA-Schedule-9: bounded general schedule lowering IR

HA-Schedule-8 raised validated exact coverage to 82/156, but left ten legal
schedules per size unsupported. The remaining shapes combined reconstruction
before a group's GEMM, nonlocal work, unrelated form preparation, larger
fan-out, and repeat updates to one output. Another whole-schedule template
would duplicate the form, product, HBM, and output logic in HA-6/7/8. HA-9
introduces a small **schedule-semantics IR**, not a general GPU compiler IR.
It does not change decompositions, CP-SAT search, schedule hashes, numerical
tolerances, or earlier evidence.

## Reproduce and evidence boundary

The command verifies the retained HA-5 and HA-8 manifests, reconstructs the
same 156 schedule-size records, and refuses an existing output directory:

```bash
PYTHONPATH=src:. python scripts/run_hardware_aware_general_lowering_ir_study.py \
  --output /tmp/ha9-cpu --skip-gpu

PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_general_lowering_ir_study.py \
  --output artifacts/tensor-game/hardware-aware-general-lowering-ir-v1
```

The first form is CPU-only. It reports structural translation and planning,
not correctness-passing GPU coverage. The [retained artifact](../artifacts/tensor-game/hardware-aware-general-lowering-ir-v1)
contains full IR and plan records, individual correctness seeds, profiler
launch sequences, buffer audits, representative compiler text/hashes, bounded
execution smoke, high-ranked coverage, decisions, and a SHA-256 manifest.
`--refresh-derived` verifies that manifest and all retained GPU plan hashes
before rebuilding CPU joins or updating source-commit provenance. It does
not rerun GPU kernels.

The pre-implementation audit records full hashes and group signatures in
`unsupported_pre_analysis.json`. At each size, the ten HA-8 unsupported
records were three conventional, five frozen rank-8, and two Strassen. Nine
had reconstruction before a group's GEMM; nine had nonlocal reconstruction;
six had nonlocal output work; five exceeded the old two-target bound; three
had unrelated form work; and two revisited an output target within one
group. Counts overlap. The largest retained group has eight operations and
one GEMM. No retained legal schedule needs multi-GEMM fusion, cross-group
`ON_CHIP` lifetime, nested form dependencies, dynamic aliasing, or concurrent
partial updates.

## IR model and validation

`ScheduleLoweringIR` records its version, source schedule and DAG hashes,
algorithm/decomposition identity, execution-instance values, ordered
operations, explicit launch groups, HBM buffer plan, output transitions,
recomputation records, provenance, and an additional stable IR hash. Every
value has a source DAG node ID, producer and consumer operation IDs, class,
dtype, block shape and bytes, abstract location, and HBM buffer ID when
materialized. Classes are `INPUT`, `PREPARED_FORM`, `GEMM_PRODUCT`,
`PARTIAL_OUTPUT`, and `FINAL_OUTPUT`. Repeated eligible forms get distinct
execution-instance IDs; prior schedule and lowering hashes remain unchanged.

The executable operation set is `PREPARE_FORM`, `RECOMPUTE_FORM`, `GEMM`,
`RECONSTRUCT`, and `FINALIZE_OUTPUT`. Immutable inputs are `INPUT` values.
HBM loads/stores are typed value edges and placement records, not redundant
executable instructions. Group membership is an explicit launch boundary.
This captures the requested load, materialize, update-partial, and finalize
semantics without implying low-level Triton instructions or register/shared
memory placement.

The standalone CPU validator checks identity and hash, every source
execution and dependency, unique producers, producer-before-consumer,
cross-group HBM availability, local `ON_CHIP`/`EPHEMERAL` lifetimes, explicit
recomputation inputs, ordered group membership, every required reconstruction
contribution exactly once, all four final outputs, no update after final,
and HBM buffer identity and lifetime. A mutated or inconsistent IR cannot
reach primitive planning. The output chains remain deterministic
`UNINITIALIZED → PARTIAL → FINAL`; a later same-group contribution uses the
current tile's FP32 state and an HBM partial write is still performed when
the schedule requires it. Final FP16 quadrants are disjoint regions of one
output tensor. Every other HBM intermediate has one stable logical buffer;
there is no aggressive reuse.

## Primitive planning and physical lowering

The six-entry CPU registry matches groups, not whole schedules:

| Primitive | Retained group pattern | Physical implementation |
|---|---|---|
| `SINGLETON_FORM` | One prepared form | HA-6 form kernel |
| `SINGLETON_GEMM` | One GEMM | HA-6 FP32 GEMM kernel |
| `SINGLETON_RECON` | One contribution | HA-6 accumulator kernel |
| `SINGLETON_FINAL` | One output | HA-6 output kernel |
| `ORDERED_ONE_GEMM` | One GEMM within at most eight ordered operations and four contributions | One parameterized tiled Triton kernel |
| `ORDERED_RECON_FINAL` | One contribution followed by its final output | The same ordered Triton kernel |

The registry declares accepted patterns, value classes, dtype, integer
coefficients, HBM/recomputation support, fan bounds, partial-output support,
launch count, and implementation. The planner never splits an intended exact
group. A pattern outside the bounds returns `LOWERING_UNSUPPORTED` with an
explicit reason. The ordered primitive receives a compact operation
descriptor and resolved, planned buffer pointers. It is one reusable kernel
source, not per-schedule source generation. Each CTA owns a disjoint output
tile and executes that group's operations in source order. Operand forms
are rounded to FP16 before `tl.dot`; products and partial chains accumulate
in FP32. One CTA owns each HBM form tile write. Other CTAs compute their
operand tiles locally, so no CTA reads another CTA's newly written form in
the same grid. No atomics or concurrent same-block output writes are used.

The default planner/backend is `IR_PRIMITIVES`. `prefer_legacy=True` selects
the unchanged HA-6/7/8 physical path for hashes it already supports.
`allocate_backend` and `execute_backend` expose this selection. Old plans
remain available and their hashes are not replaced. The IR path emits one
launch per retained group. Kernel names and resource use may differ even
where the schedule semantics are identical. The comparison classifies 20
old exact records as `EQUIVALENT` in kernel order, 80 as
`SEMANTICALLY_EQUIVALENT_DIFFERENT_PLAN`, 16 old approximate paths as
`DIFFERENT`, and 40 newly covered records as `NOT_APPLICABLE` to a legacy
physical plan. Six representative old paths also passed direct IR/legacy
output agreement checks at N=256. Three were bit-identical on the tested
input. The stored/recomputed rank-8 anchors differed by at most 0.015625;
the conventional approximate reference by 0.03125. All passed the
established output comparison. One fully materialized control reused 20
identically named kernels; their 20 IR/legacy cubin hashes matched.

## Correctness and coverage

All 156 schedule-size records translated to `IR_VALID`; none were
`IR_UNSUPPORTED` or `IR_INVALID`. All 156 matched a primitive plan with
structural `EXACT_MAPPING`; there are no remaining approximate or
lowering-unsupported IR mappings in this frozen corpus. Each was run on
deterministic seeds 11, 23, and 37. Nine N=256 representatives also used
seeds 41 and 53: 486 full-operation checks. Inputs and output were FP16,
product accumulation FP32, with unchanged `rtol=0.005` and `atol=0.05`.
Every mapped plan passed its logical buffer audit and profiled kernel count
and order: 156/156 each.

| N | HA-8 exact passing | HA-9 exact passing | Incorrect exact | IR-valid / structural exact |
|---:|---:|---:|---:|---:|
| 128 | 25/39 | 39/39 | 0 | 39/39 |
| 256 | 25/39 | 39/39 | 0 | 39/39 |
| 512 | 16/39 | 27/39 | 12 | 39/39 |
| 1024 | 16/39 | 27/39 | 12 | 39/39 |

Validated exact coverage rises from 82/156 (52.6%) to **132/156 (84.6%)**.
Conventional passes all ten schedules per size and frozen rank-8 all
seventeen. Strassen passes all twelve at 128/256 and none at 512/1024.
All 24 large Strassen mappings remain `LOWERING_INCORRECT` under the fixed
tolerance. They are IR-valid, structurally lowerable, topology-correct, and
buffer-correct; the failure is numerical. This neither proves a single
rounding cause nor resolves the earlier HA-6/7/8 findings. Failed paths are
excluded from passing coverage and smoke timing.

Frozen HA-5 static selections are IR-valid and structurally lowerable at
12/12 top-1, 36/36 top-3, and 108/108 Pareto entries. Correctness-passing
counts are 10/12, 30/36, and 90/108, respectively. The old eager-mapped
top-1 set is 11/13 passing and the old graph-mapped top-1 set 12/14.
Those mode rankings include **only HA-5's older measured subset**. The
newly exact IR schedules have not been benchmark-calibrated or reranked in
eager or graph mode. The two non-passing top-1 entries in each mode are
large-size Strassen, not primitive-planning failures.

Nine bounded N=256 IR paths received five CUDA-event execution-smoke
samples, including three newly enabled schedules and conventional,
stored/recomputed rank-8, materialized, and Strassen controls. The samples check execution
plumbing and catastrophic behavior only; they are unpaired and are **not**
performance rankings. Representative ordered kernels used 40 registers and
4 KiB static shared memory for conventional, 40–46 registers and 6 KiB for
rank-8, and 48 registers and 8 KiB for Strassen. The artifact retains
TTIR/TTGIR/PTX/SASS text and cubin hashes for these specializations, not
cubin binaries. PTX is not final machine code, compiler metadata for one
specialization is not a universal resource bound, and higher lowering
coverage is not a speedup.

## Decisions and limits

The primary decision is `GENERAL_IR_COVERAGE_UNLOCKED`. A six-entry
primitive registry now composes the complete retained legal schedule space
without bespoke schedule source. The architecture decision is
`IR_ARCHITECTURE_VIABLE` for this bounded corpus. This does **not** establish
support for arbitrary DAGs, multi-GEMM groups, dynamic aliasing, larger
fan-out, or arbitrary Triton generation. IR validity does not guarantee
numerical correctness or physical HBM traffic. Static HBM placement may be
affected by caches, compiler lowering, and runtime behavior.

Closed-loop status remains `NOT_READY_FOR_CLOSED_LOOP_PROTOTYPE` despite
passing the 75% overall coverage target. The new paths lack fresh
mode-specific calibration and general result-driven candidate reselection.
Large Strassen paths also fail numerically. The next bounded milestone is a
calibration/reselection study over correctness-passing IR paths, while keeping
every execution correctness-gated. An agent loop is not implemented here.
