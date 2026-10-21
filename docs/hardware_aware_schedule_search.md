# HA-Schedule-2: bounded constraint-based schedule search

HA-Schedule-1 showed meaningful static traffic and lifetime differences among
legal hand-authored schedules. This milestone searches those decisions with
optional OR-Tools CP-SAT. The decomposition, exact DAG, validator, and cost
evaluator remain unchanged as the independent correctness and accounting
boundary. This is CPU-only. No Triton or GPU measurement occurs here.

## Fixed-order formulation

The solver uses one deterministic term-major topological order derived from
HA-Schedule-1 `TermPlan`: each distinct input form appears before its first
product, then each product is followed by its output accumulations and any
completed final outputs. This is fixed before solving. It differs from the
storage order of `dag.nodes`, which lists all products before reconstruction;
using that storage order would exclude the very fusion schedules under study.
There is no operation-order variable or arbitrary topological reordering.

For each executable operation position `i`, integer `group[i]` is
nondecreasing, starts at zero, and advances by at most one. This prevents empty
internal groups. By default each group can contain at most one logical GEMM.
That explicit, configurable limit prevents the abstract solver from choosing a
single giant multi-GEMM group as its default answer. It is a modeling bound,
not a proven hardware limit.

Each materializable intermediate has a Boolean HBM decision. If it is not in
HBM, each ordinary consumer must be in the producer's group. Eligible cheap
`LINEAR_FORM` nodes can instead be recomputed: they have immutable original
inputs, at least two consumers, and at most two modeled operations per block
element. The first execution serves the first consumer. If selected, every
later consumer gets an explicit `EPHEMERAL` execution just before use. This
all-or-none per-form rule avoids recursive recomputation and matches the
HA-Schedule-1 validator's consume-once semantics. Extra source reads,
arithmetic, and DAG executions are counted by that evaluator.

At every fixed-order position, the model tracks each non-HBM value from birth
through its final local use, including overlap with the consumer's newly born
result. Recomputed `EPHEMERAL` forms count in the same abstract capacity at
their consuming GEMM. The resulting `peak_abstract_local_bytes` is constrained
by `on_chip_capacity_bytes` and checked against the independent evaluator.
`peak_on_chip_live_bytes` alone excludes EPHEMERAL values, so capacity uses the
combined local metric. HBM temporary peak is modeled separately by birth and
last use. This is exact for the fixed abstract operation order and these
recomputation rules. It is not a compiler occupancy or placement estimate.

## Objectives and solver/evaluator gate

Available objectives are `MIN_TEMP_HBM_TRAFFIC`, `MIN_HBM_WRITES`,
`MIN_PEAK_HBM_TEMP`, `MIN_LAUNCH_GROUPS`, and `MIN_RECOMPUTE_OPS` under an
explicit temporary-traffic cap. `LEX_TEMP_RECOMPUTE_LAUNCH` solves three
stages: minimize temporary HBM traffic, fix its proven optimum, minimize
recomputation operations, fix that optimum, then minimize launch groups. There
are no weighted latency scores. An incomplete lexicographic solve is not
reported as a proven optimum. `UNKNOWN` and timeout remain distinct from
`INFEASIBLE`.

Recovered solutions become ordinary HA-Schedule-1 `Schedule` objects. Their
semantic schedule hash excludes solver settings; run provenance is stored
beside them. Every recovery is independently validated and evaluated. The
solver predicts temporary HBM traffic and writes, recomputation operations
and execution count, group count, peak temporary HBM, and peak abstract local
bytes. All seven integers must match the evaluator exactly. A mismatch or
invalid schedule is marked `MODEL_INVALID` and omitted from valid metrics.
Run records preserve OR-Tools version, seed, workers, timeout, deterministic
settings, status, stage values, variables, constraints, branches, conflicts,
wall time, and diagnostic failures.

## Reproduce

Install the existing optional `constraint-search` extra to run CP-SAT. The
default command is a small CPU-only smoke search:

```bash
PYTHONPATH=src python scripts/run_hardware_aware_schedule_search.py \
  --output /tmp/ha-schedule2-smoke --smoke
```

The retained bounded sweep uses capacities of 1, 2, 3, 4, 6, and 8 input-block
units, recomputation budgets of zero and one block of elementwise operations,
and both traffic-only and staged lexicographic objectives:

```bash
PYTHONPATH=src python scripts/run_hardware_aware_schedule_search.py \
  --output artifacts/tensor-game/hardware-aware-schedule-search-v1 \
  --full-study --timeout-seconds 2
```

The CLI also accepts algorithm, matrix sizes, all three dtypes, raw byte
capacities, raw operation budgets, objective selection, a temporary-traffic
cap, maximum groups, maximum GEMMs per group, seed, workers, timeout, and
maximum retained schedules. It refuses an existing output directory.

One block unit is one **input** block payload. At N=128 in the retained FP16
input/output and FP32 accumulation configuration, that is 8,192 bytes, while a
product or accumulator is 16,384 bytes. The solver therefore uses exact byte
sizes rather than treating all values as equal units. With block-relative
capacity and operation budgets, the CP-SAT problem has identical normalized
constraints at all four sizes. It is solved once at N=128 per algorithm and
sweep point, then its structural schedule is rebound to each size and checked
again by the independent evaluator. Raw-byte or raw-operation modes solve
each size separately.

Artifacts include config and provenance, DAG and model manifests, solver runs,
recovered schedules, independently evaluated metrics, manual controls, a
deterministic Pareto frontier, aggregate counts, notes, and SHA-256 hashes.
Pareto points use four static dimensions: temporary HBM traffic,
recomputation operations, peak abstract local bytes, and launch groups.
Different schedules with the same vector are listed as equivalent at one
frontier point. This is static dominance only.

## Retained controls and findings

The retained sweep used OR-Tools 9.15.6755, seed 20261018, one worker, and a
two-second per-solve timeout. All 72 solves were `OPTIMAL`; all 72 recovered
schedules passed independent validation and all predicted metrics agreed.
The models contained 158–343 variables and 253–533 constraints. Normalized
reuse yielded 288 valid size-specific metric records.

At N=128, capacity is shown in input-block units. Each table value is modeled
temporary HBM traffic in bytes under lexicographic optimization. The project
rank-8 columns show zero versus one block of recomputation budget.

| Capacity | Conventional | Strassen | Project rank-8, budget 0 | Project rank-8, budget 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 524,288 | 753,664 | 819,200 | 794,624 |
| 2 | 262,144 | 491,520 | 524,288 | 524,288 |
| 3 | 262,144 | 409,600 | 524,288 | 499,712 |
| 4 | 131,072 | 262,144 | 393,216 | 368,640 |
| 6 | 131,072 | 262,144 | 393,216 | 368,640 |
| 8 | 131,072 | 262,144 | 393,216 | 368,640 |

The four-block capacity recovers the hand-authored combined-fusion traffic
points for both references. Strassen gets a reduction already at one block
unit, but reaches its combined-fusion point at four units, the same threshold
as conventional under this model. The frozen rank-8 candidate selects the
known recompute-vs-store option at one, three, and four-or-more units when its
budget permits it. At four units that gives 368,640 instead of 393,216
temporary HBM bytes, with 4,096 extra elementwise arithmetic operations and
16,384 extra source-read bytes. All completed capacity-sweep traffic sequences
were checked for nonincrease; none reversed. These are not time measurements.

At N=128 there are five unique solver schedule hashes for conventional, seven
for Strassen, and twelve for the frozen rank-8 candidate. Deduplicating equal
metric vectors leaves Pareto frontiers of 3, 5, and 6 points respectively.
Some manual reference points remain nondominated because they use zero local
capacity or tie the solver's best traffic point. The solver also finds legal
points with fewer launch groups than several manual project-rank-8 schedules.
Every improvement here is on declared *static* dimensions only.

Decision: `PROCEED_GPU_LOWERING`. The small frontier contains distinct
capacity, traffic, and recomputation choices, and the one-GEMM-per-group bound
makes selected schedules plausible candidates for a controlled lowering study.
HA-Schedule-3 should implement only a few selected schedules, correctness-gate
them, then measure complete operations and inspect compiler/profiler evidence.
No HA-Schedule-3 code is included here.

ON_CHIP capacity remains abstract. CP-SAT does not model actual CUDA scheduling,
compiler register/shared-memory placement, cache traffic, launch overhead,
occupancy, or latency. Fixed topological order and all-or-none cheap-form
recomputation limit the search. Static HBM bytes are not measured DRAM traffic.
Recomputation arithmetic is modeled, not timed. Fewer modeled bytes do not
prove lower GPU latency. A solver-generated launch group may not map directly
to one Triton kernel. No accelerator measurements occurred in HA-Schedule-2.
