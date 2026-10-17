# Tensor Game residual-subspace and two-step lookahead study

## Result

Bounded two-step residual scoring did not rescue Strassen's initial actions. Exact residual-subspace features sometimes changed their ranks at later Strassen residuals, but did not distinguish them at the original multiplication tensor. Proposal coverage remains the first limiting factor.

This is an offline CPU diagnostic. It does not change `bounded_128`, beam admission, search, the coefficient domain, or hardware ranking.

## Motivation and exact definitions

The current provider favors actions that immediately reduce squared residual norm. Every validated Strassen first action instead increases that norm. This study tests whether one additional action or exact residual structure reveals value hidden from the greedy score.

For integer residual tensor `R` and rank-one actions `a` and `b`, the evaluator computes

```text
Delta1(a)   = ||R||F^2 - ||R - a||F^2
Delta2(a,b) = ||R||F^2 - ||R - a - b||F^2
```

All transitions, norms, and reductions use integers. Matrix ranks use rational Gaussian elimination with `Fraction`. No floating-point rank tolerance is involved. Direct residual reconstruction tests the results.

The feature record contains:

- nonzero count, coefficient L1 norm, maximum magnitude, and negative count;
- the exact rank and nonzero row/column counts of all three tensor flattenings;
- the exact matrix rank of every slice along each mode;
- action coefficient, support, and arithmetic-cost descriptors.

A flattening rank is a lower bound on tensor rank. It is not tensor rank. A smaller flattening or slice rank does not guarantee an easier complete decomposition.

## Method

The controlled experiment used seven deterministic states: the original 2x2 multiplication tensor, conventional forward depths 1 and 4, Strassen forward depths 1, 3, and 5, and Strassen reverse depth 3. No serialized residual from an earlier failed search was available. The known-trajectory states are therefore representative probes, not replayed failures.

Two pools were kept separate:

- **Provider reachable:** up to 12 complete actions returned by the unchanged `bounded_128` provider with its 128-pair budget.
- **Expanded offline oracle:** the provider output plus the 8 validated conventional and 7 validated Strassen actions. Known coefficients appear only in this diagnostic pool. They never influence ordinary provider selection.

First actions are compared by immediate norm, best bounded two-step norm, and a fixed structural lexicographic key: flattening-rank sum, maximum flattening rank, nonzeros, L1 norm, squared norm, then stable action ID. This is an untuned diagnostic ordering. There are no learned or Strassen-fitted weights.

The default provider lookahead cap was 12 first actions, 12 continuations per first action, and 144 complete ordered-pair evaluations. The oracle cap was 32 first actions, 12 continuations, and 384 evaluations. Actual counts can be smaller when a provider returns fewer actions. A 12-evaluation comparison used 4 first actions and 3 continuations. A 48-evaluation comparison used 12 by 4.

## Known trajectories

All 56 ordered pairs of distinct conventional terms and all 42 ordered pairs of distinct Strassen terms were evaluated. Swapped pairs were retained only as ordered intermediate observations. Endpoint counts were deduplicated to 28 conventional and 21 Strassen term pairs.

| Family | First / second / total pattern | Ordered pairs | Positive two-step totals |
|---|---|---:|---:|
| Conventional | `+1 / +1 / +2` | 56 | 56 |
| Strassen | `-2 / -2 / -4` | 24 | 0 |
| Strassen | `-2 / 0 / -2` | 12 | 0 |
| Strassen | `-4 / 0 / -4` | 6 | 0 |

Here each pattern is immediate reduction, improvement from the second step, and total two-step reduction. No Strassen pair had a negative first step followed by positive second-step progress. The best second step was neutral for 18 ordered pairs. Every Strassen two-step total remained negative. Additivity made swapped endpoints identical, while the intermediate norm remained order-sensitive when unlike actions were paired.

## Candidate discrimination

At the original tensor the provider returned 8 unique actions and recalled 0 of 7 Strassen probes. In the 15-action oracle pool, Strassen immediate ranks were `9–15` except for the term with rank 15 ordering shown in the raw file. Two-step and structural ranks were identical. At a top-12 oracle cutoff, each method recalled 4 of 7. Thus the extra signals added no initial discrimination.

Results became mixed after known Strassen prefixes:

| State | Provider Strassen recall | Immediate top-12 | Two-step top-12 | Structural top-12 |
|---|---:|---:|---:|---:|
| target depth 0 | 0/7 | 4/7 | 4/7 | 4/7 |
| Strassen forward depth 1 | 0/7 | 1/7 | 0/7 | 1/7 |
| Strassen forward depth 3 | 1/7 | 3/7 | 3/7 | 4/7 |
| Strassen reverse depth 3 | 0/7 | 2/7 | 2/7 | 4/7 |
| Strassen forward depth 5 | 2/7 | 2/7 | 2/7 | 2/7 |

The structural ordering improved top-12 probe retention at the two depth-3 states. It also moved individual actions substantially. That is weak diagnostic evidence only: these states were created from known Strassen prefixes and the ranking did not help at the unbiased initial state.

## Budgets and overhead

At the initial state, provider-only lookahead evaluated 56 ordered pairs because only 8 first actions and 7 distinct provider continuations were available. The complete oracle diagnostic evaluated 180 pairs. Across states, provider lookahead used 12 to 144 evaluations and the oracle used 180 to 324. Total runtime was 10.54 seconds on the recorded environment.

The immediate oracle comparison scores 15 complete actions. The 180-evaluation oracle lookahead therefore costs 12 times as many complete action evaluations, excluding repeated provider generation. This is action-evaluation-equivalent accounting, not CPU-equivalent accounting: every continuation also invokes coarse provider generation, and exact structural ranks have a different cost from a norm score. A node-equivalent 12-by-12 expansion is 144 pair evaluations, but there is no exact baseline-equivalent CPU comparison because the baseline does not regenerate candidates at a second state.

Sensitivity at the original tensor was flat. The 12-evaluation setting retained no Strassen probe because it only admitted four greedily ranked first actions. Both 48 and 144 evaluations retained the same 4 of 7 oracle probes and the same four negative-immediate actions. More continuations did not improve first-action ranking.

## Limitations and implication

The oracle result is not deployable search performance. It grants explicit access to known decompositions and exists only to ask whether the scores would recognize them. Even under that favorable coverage, two-step norm did not improve initial Strassen ranks. Provider recall was zero initially and at most 2 of 7 in the sampled states. Coverage therefore remains limiting, but coverage alone is not sufficient: once exposed, a two-step squared-norm horizon still rejects Strassen pairs.

The smallest justified next change is another offline experiment, not a default-policy change. Test a bounded sequence feature that can express cancellation beyond two actions, while reserving a small, trajectory-agnostic quota for structurally distinct negative-immediate candidates. Validate it first on held-out synthetic decompositions. Do not tune it against the seven Strassen terms.

## Reproduction and evidence

Run from the repository root:

```sh
PYTHONPATH=src python scripts/run_tensor_game_residual_lookahead.py \
  --output artifacts/tensor-game/residual-lookahead-v1
```

The output directory must not exist. Raw state rankings and distributions are in `states.json`. Ordered trajectory evidence is in `trajectory_pairs.json`. `configuration.json` records the command, limits, environment, and source commit. `artifact_hashes.json` authenticates every other artifact.
