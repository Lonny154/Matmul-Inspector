# Experimental Tensor Game feasibility gate

## Result

The optional exact flattening-rank gate safely reduced downstream work in the frozen rank-7 search. It did not find a rank-7 solution. It preserved rank-8 reliability and is disabled by default.

Across three deterministic repetitions, the rank-7 gate rejected 231 generated children. Of those, 51 would otherwise have entered the beam. It avoided 72 later node expansions and provider calls. Rank-8 remained 3/3 with no rejected children. The result demonstrates useful late-depth pruning on this search path, not broader solve improvement.

## Exact rule and integration point

For residual `R` and post-action remaining budget `b`, the gate computes

```text
L(R) = max(rank(R_mode0), rank(R_mode1), rank(R_mode2)).
```

Ranks use the existing rational exact implementation. A child is removed only when `L(R) > b`. Passing the gate means feasibility is unknown.

`DiscoveryConfig.feasibility_gate` accepts `"off"` and `"flattening_rank"`. The default is `"off"`. The experimental path performs exact residual subtraction first, handles exact zero terminals, rejects nonterminal children at the final depth under the existing rule, and then applies the gate before residual deduplication and beam admission. Candidate scores and the ordering of surviving baseline-mode children are unchanged.

For a child at depth `d` under target rank `r`, the implementation uses `b = r - d`. Rank checks run only when `b <= 3`. No flattening is calculated with four or more moves remaining. This cutoff is exact for the current `4×4×4` residuals because their flattening ranks cannot exceed four.

## Correctness evidence

CPU tests cover strict inequality, inactive early depths, post-action budget accounting, exact zero handling, deterministic diagnostics, budget enforcement, explicit gate-off equivalence, and the unchanged provider interface. Direct trajectory tests apply the gate to every conventional rank-8 and Strassen rank-7 residual. None is rejected.

Search diagnostics also report zero gate losses on rank-matched known trajectories. Three diagnostic losses appear when the eight-term conventional trajectory is intentionally traced inside the rank-7 trials. Those suffixes require more terms than remain and are correctly rejected. The Strassen first action remains absent from `bounded_128`, so the gate cannot improve its proposal coverage.

## Matched configuration

The command used ranks 7 and 8, seeds 3, 11, and 29, a 256-node limit, beam width 12, 12 actions per state, global admission, and the unchanged `bounded_128` provider. Gate-off and gate-on trials differ only in `feasibility_gate`.

The baseline-mode ordering is deterministic and does not consume the seed. The three seeds therefore repeat the same search for reproducibility. They are not independent stochastic samples.

| Gate | Rank | Solves | Nodes | Provider calls | Generated children | CPU wall time |
|---|---:|---:|---:|---:|---:|---:|
| Off | 7 | 0/3 | 207 | 207 | 912 | 6.57 s |
| Flattening rank | 7 | 0/3 | 135 | 135 | 732 | 4.19 s |
| Off | 8 | 3/3 | 225 | 225 | 930 | 7.15 s |
| Flattening rank | 8 | 3/3 | 225 | 225 | 930 | 6.98 s |

Every search exhausted its reachable frontier before the 256-node limit. This is a matched-limit comparison, but not equal completed work: pruning is intended to change completed work. The runner does not support a cooperative elapsed-time limit, so no elapsed-time-matched search was attempted. Raw measured times are retained instead.

## Depth-specific pruning

| Rank and child depth | Remaining | Checked | Bound distribution | Pruned | Would enter beam |
|---|---:|---:|---|---:|---:|
| Rank 7, depth 4 | 3 | 180 | `3: 93, 4: 87` | 87 | 15 |
| Rank 7, depth 5 | 2 | 144 | `3: 144` | 144 | 36 |
| Rank 8, depth 5 | 3 | 144 | `3: 144` | 0 | 0 |
| Rank 8, depth 6 | 2 | 108 | `2: 108` | 0 | 0 |
| Rank 8, depth 7 | 1 | 72 | `1: 72` | 0 | 0 |

No earlier children were checked. At rank-7 depth 4, removing infeasible children changed admission and allowed 15 additional feasible-unknown children into the beam across the three runs. At depth 5, all 144 generated children were impossible with two moves remaining, so the gated frontier became empty.

The distinction between pruning counters matters:

- 231 children were mathematically rejected.
- 51 deduplicated rejected states would have occupied beam slots.
- 72 future expansions and provider calls were actually avoided relative to paired baseline runs.
- 180 later candidate edges were avoided.

High theoretical pruning therefore translated into reduced downstream work here. The baseline reached depth 7. The gated search stopped at depth 5 because it proved every reachable child at that layer infeasible. A smaller reported depth is not lost reliability in this case; it is earlier proof that this reachable frontier cannot complete within rank 7.

## Runtime and caching

Rank-7 exact checks took 40.34 ms across the three runs. Total rank-7 wall time fell by 2.38 seconds, or about 36%, because avoided provider calls dominated rank-check overhead. This timing used fixed baseline-before-gated ordering and a small Python workload, so it should not be treated as a general speedup.

Rank-8 checks took 37.82 ms and saved no work because every checked child lay on conventional-style feasible suffixes. Aggregate rank-8 wall time happened to be slightly lower with the gate, but the difference is not evidence of a speed benefit. The work counters are identical and the comparison was not interleaved.

No cache was added. Exact residual deduplication occurs after the gate, so repeated incoming paths can repeat a rank calculation. This experiment is too small to justify extra memory and complexity without a dedicated duplicate-check profile.

## Limitations and recommendation

The provider follows conventional residual reductions and never proposes the initial Strassen action. The experiment therefore tests faster rejection of an unsuccessful reachable rank-7 frontier, not improved access to a rank-7 solution. The beam is narrow, the workload is only `4×4×4`, and deterministic repetitions do not measure seed sensitivity. Rational-rank cost will grow with tensor dimensions and coefficient growth.

The result is positive for experimental pruning but insufficient for default integration. The smallest next experiment is a wider, interleaved matched run that preserves the same provider and compares gate off/on at beam width 128. It should profile duplicate rank checks before considering a cache. Keep the gate opt-in until rank-8 reliability and net wall-time benefit hold across broader state sets.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_feasibility_gate.py \
  --output artifacts/tensor-game/feasibility-gate-v1
```

The output directory must not exist. `trials.json` contains every trial and depth diagnostic. `summary.json` contains paired work deltas and depth aggregates. `configuration.json` records the exact command and environment. `artifact_hashes.json` authenticates the other files.
