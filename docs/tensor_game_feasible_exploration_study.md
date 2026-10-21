# Feasibility-guided algebraic exploration

## Result

Exact late-depth pruning makes quota-based alternative exploration cheaper without changing solve outcomes in this controlled matrix. Rank 7 remains unsolved. Rank 8 remains 3/3 for all six primary configurations.

The interaction is real: the baseline provider exposes no infeasible rank-8 children, while quota multi-`w` exposes 81–87. Removing them avoids five or six future expansions per trial. Provenance-aware admission keeps alternative-origin edges alive deeper than global admission, but does not improve solve rate or pruning efficiency here.

All options remain explicit. The default provider, admission policy, and feasibility gate are unchanged.

## Integration architecture

The experiment composes existing components at their established boundaries:

1. `bounded_128` or `QuotaMultiWProvider` proposes actions using the existing residual score and exact top-k `w` implementation.
2. Quota proposals retain their existing `baseline` or `alternative` incoming-edge provenance.
3. `discover` constructs the child with exact integer subtraction.
4. The optional existing flattening-rank gate checks eligible children with at most three moves remaining.
5. Surviving children use the existing exact-residual deduplication and global or provenance-quota admission.

The lower bound is not a score. Passing it means feasibility is unknown. Duplicate residual states still consume one beam slot. Complete solutions are accepted only after exact reconstruction validation.

The only additional search instrumentation records feasibility rejections by provenance and optional beam-admission time. Phase timing is disabled by default to preserve stable ordinary diagnostics.

## Experimental matrix and timing method

The six primary configurations are the Cartesian product requested in the study:

| Provider | Admission | Gate |
|---|---|---|
| `bounded_128` | Global | Off / on |
| Quota residual 11/1 | Global | Off / on |
| Quota residual 11/1 | Provenance 11/1 | Off / on |

Every primary run uses rank 7 or 8, seeds 3, 11, and 29, node limit 256, beam width 12, and at most 12 returned actions. Gate order alternates across matched pairs. The baseline search ordering and providers are deterministic, so seeds reproduce the same algorithmic work. The three trials are useful interleaved timing rounds, not independent stochastic samples.

Provider settings are identical within every gate pair. The quota provider retains 11 baseline actions, reserves one residual-ranked alternative slot, uses pair width 128, and generates two exact `w` candidates per pair.

## Node-matched solve and work results

| Provider / admission | Rank | Off solves | On solves | Off nodes | On nodes | Nodes avoided | Mean paired time delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline / global | 7 | 0/3 | 0/3 | 207 | 135 | 72 | -0.726 s |
| Baseline / global | 8 | 3/3 | 3/3 | 225 | 225 | 0 | -0.016 s |
| Quota / global | 7 | 0/3 | 0/3 | 210 | 138 | 72 | -0.858 s |
| Quota / global | 8 | 3/3 | 3/3 | 246 | 228 | 18 | -0.210 s |
| Quota / provenance | 7 | 0/3 | 0/3 | 210 | 138 | 72 | -0.866 s |
| Quota / provenance | 8 | 3/3 | 3/3 | 246 | 231 | 15 | -0.176 s |

Provider calls equal expanded nodes in this search, so the same counts were avoided. Candidate edges avoided across the three pairs were 180 for baseline rank 7, 252 for quota-global rank 7, 261 for quota-provenance rank 7, and 54 for each quota rank-8 configuration.

The rank-7 runtime saving is larger with quota exploration because each avoided provider call performs broader complete-action generation. Mean provider-generation reductions were 0.737 seconds per baseline pair, 0.871 seconds per quota-global pair, and 0.877 seconds per quota-provenance pair. These are measured Python CPU times for this run, not general speedup claims.

The rank-8 baseline timing delta is noise: work counters are identical. Quota rank-8 has a small algorithmic reduction and consistently lower paired wall time in these three interleaved rounds. The sample is still too small and deterministic to establish a portable runtime ratio.

## Feasibility pruning by provenance

| Provider / admission | Rank | Checked | Pruned | Baseline pruned | Alternative pruned | Would enter beam |
|---|---:|---:|---:|---:|---:|---:|
| Baseline / global | 7 | 324 | 231 | 231 | 0 | 51 |
| Baseline / global | 8 | 324 | 0 | 0 | 0 | 0 |
| Quota / global | 7 | 396 | 300 | 231 | 69 | 51 |
| Quota / global | 8 | 432 | 81 | 0 | 81 | 18 |
| Quota / provenance | 7 | 399 | 309 | 240 | 69 | 51 |
| Quota / provenance | 8 | 435 | 87 | 6 | 81 | 21 |

Broader action exploration clearly exposes additional impossible children. At rank 8 every rejection under global quota admission is alternative-origin. Provenance admission retains a different frontier and exposes six additional impossible baseline-origin children. No rank-matched conventional or Strassen trajectory is rejected. The conventional rank-8 path remains available; the Strassen path is still unavailable at depth 1 because neither provider returns its first action.

The gate improves beam use when a rejected residual would otherwise occupy a slot. At rank 8 this happens 18 times under global quota admission and 21 times under provenance admission. These removals translate to 18 and 15 avoided expansions respectively. The counts differ because replacement survivors alter later frontiers.

## Alternative exploration and beam occupancy

Provenance labels incoming edges, not permanent ancestry. “Alternative depth” below is therefore the deepest admitted alternative-origin edge.

| Configuration | Rank | Alternative admitted off / on | Maximum alternative depth off / on |
|---|---:|---:|---:|
| Quota global | 7 | 3 / 3 | 1 / 1 |
| Quota provenance | 7 | 18 / 12 | 6 / 4 |
| Quota global | 8 | 21 / 3 | 7 / 1 |
| Quota provenance | 8 | 30 / 18 | 7 / 6 |

Global admission allows the single initial alternative but usually removes alternatives at the next layer. Provenance admission continues to preserve alternative edges after gating. It therefore adds exploration diversity when the gate is active, but that diversity costs three extra rank-8 expansions across the three trials and produces no additional solve.

The gate does not merely reduce total work indiscriminately. It removes 81 alternative-origin rank-8 children in both quota configurations while retaining the exact conventional solution path. However, this experiment cannot label the surviving alternatives as useful; deeper survival is not evidence of greater solve probability.

## Evaluation-cap approximation

A secondary run uses a 64-node cap for the two quota configurations. This approximates a complete-action evaluation cap because the provider has fixed per-call budgets. It is not exact equal compute: the gate can terminate a frontier before consuming all 64 calls.

For rank 7, gate-off consumes all 64 calls and scores about 30,950 complete actions. The gated runs stop after 46 calls and score about 22,000. For rank 8, all four runs consume 64 calls and score about 30,950 actions. All rank-8 runs stop at the node budget before solving, with or without the gate. The gate adds roughly 15 ms of rank checking and provides no downstream saving before that cap.

This secondary result separates two effects. Pruning saves compute when it can prevent future calls, as in rank 7. When a fixed cap is reached before pruned branches would be expanded, the check is overhead and does not improve the outcome.

## Interaction conclusions

1. Baseline rank-8 success remains 3/3.
2. Quota rank-8 success remains 3/3 under global and provenance admission.
3. Rank 7 remains 0/3. The gate proves the reachable frontier impossible sooner but does not improve algebraic coverage.
4. Alternative actions create additional infeasible states, including all 81 rank-8 rejections under global quota admission.
5. Provenance admission preserves deeper alternative exploration with the gate, but adds no solve benefit and slightly reduces work savings relative to global admission.
6. Runtime savings are larger with the quota provider because avoided provider calls cost more.
7. The combined mechanisms provide a benefit beyond either alone: quota exploration alone costs additional work, while the gate safely recovers part of that cost and preserves rank-8 success. It does not overcome the missing initial Strassen proposal.

## Limitations and next experiment

This is a small deterministic `4×4×4` search. Interleaving reduces order bias but does not create independent search samples. Phase timings are Python wall times. The evaluation-cap comparison is approximate. Provenance describes the latest edge rather than complete ancestry. The gate remains mathematically weak until late depth, and candidate coverage remains the rank-7 bottleneck.

The smallest justified next experiment is a matched width-128 beam run for quota global versus provenance admission with the gate on and off. It should use repeated interleaved rounds and profile duplicate rank calculations before adding a cache. Do not widen pair coverage or change scoring in the same experiment.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_feasible_exploration.py \
  --output artifacts/tensor-game/feasible-exploration-v1
```

The output directory must not exist. `trials.json` contains raw trials and per-depth counters. `summary.json` contains grouped and paired effects. `configuration.json` records the interleaved execution schedule and environment. `artifact_hashes.json` authenticates the other files.
