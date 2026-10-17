# Tensor Game exact rank-feasibility study

## Result

Exact flattening rank is a useful late-depth impossibility test in this bounded sample. It is too weak for early rank-7 pruning. The study does not change search or candidate generation.

The original 2×2 multiplication tensor has exact flattening ranks `(4, 4, 4)`. The maximum lower bound is therefore 4. It cannot distinguish rank 7 from rank 8 at the root. It safely retained every tested state on known conventional and Strassen trajectories.

Among 21 deterministically sampled states from a newly generated failed rank-7 baseline search, 12 were provably infeasible within their remaining budgets. Among 89 bounded-provider children, 34 were provably infeasible. These are sample fractions, not estimates over every possible search state.

## Mathematical definition

For residual tensor `R` with shape `(I, J, K)`, the implementation constructs:

- mode 0: `I × (J*K)`, with columns ordered by `(j, k)`;
- mode 1: `J × (K*I)`, with columns ordered by `(k, i)`;
- mode 2: `K × (I*J)`, with columns ordered by `(i, j)`.

This follows the repository tensor indexing directly. It does not reinterpret concatenated action coefficients as a flattening.

Matrix rank is calculated by deterministic rational Gaussian elimination using Python integers and `Fraction`. There is no floating-point tolerance or fixed-width integer arithmetic. The tensor-rank lower bound is

```text
L(R) = max(rank(R_mode0), rank(R_mode1), rank(R_mode2)).
```

If `L(R) > b`, no sum of `b` rank-one tensors over the rationals can equal `R`. The state is therefore impossible for the solver too. If `L(R) <= b`, feasibility is unknown. In particular, this field-based condition cannot establish a completion using the solver's restricted ternary coefficients.

## Known-solution validation

The study evaluated forward, reverse, and even-then-odd term orders. This produced 51 trajectory states including endpoints.

| Family | Lower bounds by depth |
|---|---|
| Conventional rank 8 | `4, 4, 4, 4, 4, 3, 2, 1, 0` |
| Strassen rank 7 | `4, 4, 4, 4, 3, 2, 1, 0` |

All three orders gave the same bound sequence within each family. No known valid state was pruned. The zero tensor has ranks `(0, 0, 0)` and passes a zero remaining budget.

The result also shows the early-depth limitation. A `4×4×4` tensor flattening rank cannot exceed 4. The bound cannot reject any state with at least four multiplications remaining, regardless of how hard that residual is under the ternary domain.

## Failed-search states

Earlier search artifacts recorded aggregate traces and action identities but not exact residual tensors. This study therefore replayed the frozen deterministic baseline beam logic with rank 7, node budget 256, beam width 128, 12 actions per state, and the unchanged 128-pair provider. It expanded 247 nodes and generated 1,016 candidate edges without finding a solution.

Three states were selected deterministically at evenly spaced positions in the norm-ordered state list at every depth. They are labeled `new_frozen_baseline_rank7_sample`; they are not historical captures.

| Depth | Remaining budget | Sampled states | Bound histogram | Proven infeasible |
|---:|---:|---:|---|---:|
| 1 | 6 | 3 | `4: 3` | 0 |
| 2 | 5 | 3 | `4: 3` | 0 |
| 3 | 4 | 3 | `4: 3` | 0 |
| 4 | 3 | 3 | `4: 3` | 3 |
| 5 | 2 | 3 | `3: 3` | 3 |
| 6 | 1 | 3 | `2: 3` | 3 |
| 7 | 0 | 3 | `1: 3` | 3 |

The sample violation rate is `12/21`, or 57.1%. Every sampled state from depth 4 onward violated the bound. The selection intentionally covers depths and norm-ordered positions, so this fraction must not be generalized to the full frontier.

## Offline child analysis

The unchanged provider was evaluated at the root and at every sampled failed state with at least one move remaining. Each child was constructed with exact subtraction. The lower bound was then compared with the post-action budget. Nothing was filtered from production search.

| Child depth | Evaluated | Proven infeasible | Fraction |
|---:|---:|---:|---:|
| 1 | 8 | 0 | 0% |
| 2 | 21 | 0 | 0% |
| 3 | 18 | 0 | 0% |
| 4 | 15 | 7 | 46.7% |
| 5 | 12 | 12 | 100% |
| 6 | 9 | 9 | 100% |
| 7 | 6 | 6 | 100% |
| Total | 89 | 34 | 38.2% |

All 89 actions reduced squared norm by one. Residual reduction alone therefore did not distinguish the impossible children. The infeasible children had mean provider rank 2.26, compared with 3.78 for children that passed the bound. In this sample the mathematical check rejects many locally attractive actions, not merely low-ranked tail proposals.

There were three parent states whose candidate children had the same squared norm and remaining budget but different lower bounds. One depth-4 parent produced five norm-4 children with three lower-bound-3 states and two lower-bound-4 states. With three moves remaining, the former pass with unknown feasibility while the latter are rigorously impossible. This is direct information beyond residual magnitude.

## Runtime overhead

The 89 child rank checks took 13.34 ms total, or 149.8 microseconds per child on the recorded environment. Provider generation for the same parent calls took 578.22 ms. The measured rank-check time was about 2.3% of that provider-generation time. The complete script, including the failed-search replay and artifact construction, took 8.24 seconds.

These Python timings establish affordability only for this small CPU diagnostic. They do not predict overhead inside another search implementation or at larger tensor dimensions. Rational elimination scales with flattening dimensions and coefficient growth.

## Discriminative usefulness and limits

The bound answers an exact but narrow question. It detected no impossible state through depth 3 because every bound was at most the remaining budget. It became useful exactly when the remaining budget fell below the flattening ceiling of four. It retained all known successful trajectories and separated equal-norm siblings at depth 4.

Passing the test says nothing about ternary feasibility or closeness to a complete decomposition. Flattening rank can be far below true tensor rank. The selected failed states are deterministic but small and correlated with the conventional residual path favored by `bounded_128`. No matched search with pruning was run, so there is no measured reduction in nodes or wall time.

Stronger late-terminal diagnostics could include bounded exact completion enumeration for one or two remaining terms, restricted-domain linear constraints, or a carefully implemented substitution-method lower bound. Exact completion can exploit the finite ternary domain but becomes combinatorial. Substitution bounds require more algebra and careful field assumptions. Neither is implemented here.

## Recommendation

The evidence does not justify changing the default solver. It does justify a separately named offline or experimental late-depth gate. The smallest next experiment is a matched rank-7 baseline replay with the exact bound enabled only when three or fewer moves remain. Measure candidate edges, admitted states, wall time, and solution preservation under identical budgets. Keep it out of production until that experiment confirms net benefit and regression tests retain known solutions.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_rank_feasibility.py \
  --output artifacts/tensor-game/rank-feasibility-v1
```

The output directory must not exist. Evidence includes:

- `known_trajectories.json`: all exact known-path states and three orderings;
- `failed_states.json`: serialized sampled residuals and frozen-search audit;
- `child_states.json`: candidate-level bounds, timings, and provider ranks;
- `equal_norm_witnesses.json`: equal-norm and same-parent discrimination cases;
- `summary.json`, `configuration.json`, and `artifact_hashes.json`.
