# HA-Schedule-1: exact execution DAG and static memory model

HA-Tensor-2A found `NO_STRUCTURAL_SIGNAL` in its bounded exact rank-8/9
operand-reuse search. Conventional rank-8 already has four distinct forms on
each input side. This milestone therefore holds exact decompositions fixed and
asks what changes when block computations are grouped, materialized, fused, or
recomputed. It does not search algebra or predict GPU latency.

## Representation

`tensor_game_schedule_model.py` builds a one-level 2×2 block DAG from the
project's `RankOneAction` factors. The conventional and Strassen references come
from the existing `reference_lowering` interface and pass the project's exact
decomposition validator. Each DAG contains eight `INPUT` blocks, shared
`LINEAR_FORM` nodes for non-identity u/v factors, one `MATMUL` per rank term,
an ordered `ACCUMULATE` chain for each output, and four `OUTPUT` nodes. Stable
node IDs, u/v/w coefficients, term and output provenance, dependencies, value
IDs, dtype, elements, and bytes are retained in `dag.json`. The DAG and each
schedule have content hashes. Generated DAGs are topologically validated.

For logical matrix dimension `N`, block dimension `B=N/2` and block bytes are
`B*B*dtype_bytes`. Inputs and input forms use the chosen input dtype. Products
and accumulators use the accumulation dtype. Final outputs use the output dtype.
This is an analytical payload model: no matrices are allocated. A negative
singleton input form is explicit, while a positive identity form aliases its
input and creates no preparation node. Arithmetic counts follow the existing
`linear_combination_cost` convention: additions and non-unit coefficient
scalings are counted per element, then multiplied by block element count.
Sign changes for unit coefficients are
represented in coefficients but are not a separate arithmetic count. GEMM
arithmetic is represented by a logical GEMM count, not scalar FLOPs.

## Schedule and legality

A schedule is an ordered list of launch groups. Each operation gives its node
and output location. `HBM` survives groups. `ON_CHIP` survives only within its
group. `EPHEMERAL` is consumed once and discarded; a later use requires another
execution. These are **abstract model locations**. `ON_CHIP` does not prove
register or shared-memory residency. A launch group is not guaranteed to map
one-to-one to an executable kernel, and multi-GEMM groups may be infeasible.

The validator checks DAG and schedule hashes, known nodes and locations,
dependency availability at the point of use, output completion, local lifetime,
explicit recompute permission, and orphaned work. An invalid schedule gets
structured issues and cannot receive a normal cost record. The evaluator
resolves each execution separately so a recomputed form has a new birth and
use. Peak temporary HBM and per-group ON_CHIP bytes are calculated by
allocating a result before releasing its last-use inputs. This intentionally
counts short-lived overlap. The number of simultaneously live intermediates
includes HBM, ON_CHIP, and EPHEMERAL values. These peaks are not occupancy or
compiler placement estimates.

Five policies are available:

| Policy | Abstract behavior |
| --- | --- |
| `MATERIALIZE_ALL` | HBM input forms, products, and accumulation intermediates; separate groups. |
| `FUSE_INPUT_PREPARATION` | Fuse single-use input forms into their consuming GEMM group. Shared forms remain HBM. |
| `FUSE_OUTPUT_RECONSTRUCTION` | Fuse each GEMM with its immediate accumulation uses. Cross-group partial sums remain HBM. |
| `FUSE_INPUT_AND_OUTPUT` | Combine the preceding two policies. |
| `RECOMPUTE_CHEAP_FORMS` | Start with combined fusion, then recompute shared forms with one or two modeled arithmetic operations as EPHEMERAL. |

For recomputation, records separate repeated executions, added form arithmetic,
extra source reads, a store-once HBM write avoided, and reload bytes avoided.
The last two are counterfactual components, **not** net traffic savings. Actual
net HBM read/write deltas use the full evaluated schedule. If a decomposition
has no shared cheap form, this policy legitimately matches combined fusion.

## Reproduce

The default command is fast and CPU-only:

```bash
PYTHONPATH=src python scripts/run_hardware_aware_schedule_study.py \
  --output /tmp/ha-schedule-smoke
```

The retained study uses four sizes and the frozen, exact project-new rank-8
candidate to exercise real recomputation:

```bash
PYTHONPATH=src python scripts/run_hardware_aware_schedule_study.py \
  --output artifacts/tensor-game/hardware-aware-schedule-v1 \
  --matrix-sizes 128 256 512 1024 --include-project-rank8
```

`--algorithm` can select `conventional`, `strassen`, or `all`. `--policies`
accepts any subset including `MATERIALIZE_ALL`. Input, accumulation, and
output dtype arguments accept `fp16`, `bf16`, and `fp32`. The command refuses
an existing output directory. It writes `config.json`, `provenance.json`,
`dag.json`, `schedules.json`, `schedule_results.jsonl`,
`aggregate_summary.json`, `study_notes.md`, and `hash_manifest.json`.
The aggregate compares each valid record with its own algorithm/size baseline;
invalid schedules are excluded and listed as validation failures. The optional
candidate is loaded from the frozen HA-Tensor-2A artifact and validated again.

## Initial static findings and decision

At `N=128`, input/output FP16 and accumulation FP32, one block has 4,096
elements, 8,192 input/output bytes, and 16,384 product/accumulator bytes.
The following are model counts, not measurements:

| Algorithm / policy | Temporary HBM traffic | Peak temporary HBM | Peak ON_CHIP | Groups |
| --- | ---: | ---: | ---: | ---: |
| Conventional / materialize all | 524,288 | 147,456 | 0 | 20 |
| Conventional / output or combined fusion | 131,072 | 32,768 | 32,768 | 8 |
| Strassen / materialize all | 868,352 | 147,456 | 0 | 33 |
| Strassen / input fusion | 704,512 | 147,456 | 16,384 | 23 |
| Strassen / output fusion | 425,984 | 131,072 | 32,768 | 17 |
| Strassen / combined fusion | 262,144 | 81,920 | 32,768 | 7 |
| Frozen project rank-8 / combined fusion | 393,216 | 114,688 | 32,768 | 16 |
| Frozen project rank-8 / recompute cheap forms | 368,640 | 114,688 | 32,768 | 15 |

Strassen's conservative materialization has 344,064 more temporary HBM bytes
of traffic than conventional's. Input fusion removes 163,840 temporary bytes
from Strassen's baseline. Output fusion removes 442,368; combined fusion
removes 606,208. Conventional has no input forms to fuse. The reference
Strassen DAG has no shared cheap forms, so recomputation adds no arithmetic
there. The frozen exact rank-8 candidate does: one extra form execution adds
4,096 elementwise arithmetic operations and 16,384 extra source-read bytes.
It avoids 8,192 HBM write bytes and 16,384 reload bytes relative to
store-once. Its net temporary traffic falls by 24,576 bytes relative to
combined fusion, while HBM reads remain equal because the extra source reads
replace form reloads. This is a real static tradeoff, not evidence of speed.

With fixed DAGs and policy topology, the same structural counts apply at
128, 256, 512, and 1024. Byte counts scale with `N²`. Different schedules
expose distinct temporary lifetimes and memory traffic. Decision:
`PROCEED_SCHEDULE_SEARCH`. A bounded HA-Schedule-2 can search legal grouping,
materialization, and recomputation subject to an explicit abstract ON_CHIP
capacity. Only selected, physically plausible schedules should later be
lowered and benchmarked.

Static byte counts are not measured GPU traffic. Reduced modeled traffic does
not prove reduced latency. Added arithmetic may or may not pay off on hardware.
No CUDA, Triton, Nsight, Trainium, or GPU benchmark was run for HA-Schedule-1.
