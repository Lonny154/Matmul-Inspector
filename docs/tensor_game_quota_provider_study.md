# Quota-preserving multi-w Tensor Game provider

## Research question

The previous multi-`w` provider improved known Strassen-action generation from 3/10 to 5/10 but removed useful baseline actions from the top-12 list, lost all rank-8 solutions, and scored 10,240 complete actions per call. This CPU study asks whether exact separable top-`k` generation and protected baseline quotas can retain the coverage gain, reduce proposal work, and restore rank-8 reliability.

The efficient generator succeeds and the quota improves returned recall. Baseline protection does not restore solver reliability. Adding one alternative per state changes the global beam frontier enough that rank-8 success remains 0/3.

## Separable objective

For fixed `u,v`, let `q = ||u||² ||v||²` and `d_j = sum_ij R_ijk u_i v_j`. The exact residual score is

```text
E(w) = ||R||² + sum_j (q w_j² - 2 d_j w_j).
```

Every output coordinate therefore contributes independently. The coefficient domain is `{-1,0,1}` and `w` has four coordinates. The only coupling constraint is that the final vector must not be all zero. There are no support, sign, or normalization constraints beyond the existing action validator.

For each coordinate, the implementation evaluates the three exact integer contributions and sorts by contribution then coefficient. A best-first Cartesian-product heap starts from the cheapest coordinate-choice tuple. Popping a tuple exposes neighbors formed by advancing one coordinate choice. Heap keys are total exact score, `w`, and choice indices. This preserves the existing score-first and lexicographic tie order. The all-zero result is skipped, and traversal continues until `k` valid vectors are returned or all 80 valid vectors are exhausted.

## Correctness validation

The test oracle enumerates all `3^4 - 1 = 80` valid vectors and sorts them by the same exact score and `w` tie break. Tests compare the heap output against that oracle for `k = 1, 2, 4, 8, 80, 100` across deterministic randomized contractions and denominators. Returned vectors, integer scores, ordering, ties, negative coefficients, zero coefficients, the nonzero constraint, and oversized `k` behavior match exactly.

The normal provider does not call the exhaustive oracle. With `k=2`, it evaluates 12 coordinate choices and only the heap prefix needed for two valid vectors, typically two or three complete vectors per pair.

## Provider architecture

`bounded_quota_multi_w` is optional. `bounded_128` and `bounded_multi_w` are unchanged.

The provider ranks the 6,400 `(u,v)` pairs once and shares the retained 128-pair prefix between baseline reconstruction and alternative generation. It then:

1. Reproduces the baseline candidate ranking from the shared pair prefix.
2. Protects the requested baseline prefix in its original order.
3. Generates two exact top-`k` vectors per retained pair.
4. Removes every action already present in the baseline returned set.
5. Deduplicates canonical complete actions.
6. Selects alternatives only for explicitly reserved slots.
7. Fills unused slots from remaining baseline actions.

Residual selection orders alternatives by exact score and coefficients. Diversity selection first restricts alternatives to within one residual-reduction unit of the best alternative. It then prefers new `w` support sizes, sign patterns, and residual reductions before exact reduction, preparation additions, and coefficients. No known solution or hardware-cost penalty participates in selection.

The total returned budget is at most 12. Some residuals have fewer than 12 legal baseline and quota candidates, matching existing provider behavior. The 12/0 configuration reproduces baseline top-12 output exactly.

## Quota recall

The study reuses the same 10 exact Strassen states as the two prior coverage experiments. Two `w` vectors are generated per retained pair.

| Configuration | Reachable through exact top-k | Returned exact actions | Excluded by slot budget |
|---|---:|---:|---:|
| Baseline 12/0 | 3/10 | 3/10 | 0 |
| Residual 11/1 | 5/10 | 4/10 | 1 |
| Residual 10/2 | 5/10 | 4/10 | 1 |
| Residual 8/4 | 5/10 | 4/10 | 1 |
| Diversity 11/1 | 5/10 | 4/10 | 1 |
| Diversity 10/2 | 5/10 | 4/10 | 1 |
| Diversity 8/4 | 5/10 | 4/10 | 1 |

The additional generated action at forward Strassen depth 3 remains outside every reserved-slot selection. The forward depth-4 action enters with one slot. Larger quotas change candidate distributions but do not improve exact recall. Five missing actions still have `(u,v)` ranks outside the retained 128 pairs.

## Proposal work and runtime

On the 10 recall states, top-`k` generation scores a mean 349.3 complete vectors across all 128 pairs. Including 128 baseline actions gives about 477 complete-action scores per call. The previous exhaustive provider scored 10,240.

This is a 95.3% reduction, or about 21.5× fewer complete-action evaluations per call. Looking only at alternative generation, 349.3 versus 10,240 is a 96.6% reduction.

The pair prefix is shared. This matters in practice: an initial implementation scanned all 6,400 pairs twice and took about 17.7 seconds per 256-node trial. The final implementation takes about 9.7 seconds.

Under the node-matched search:

- Baseline provider time is 23.97–24.10 seconds per three-trial rank group.
- Residual quota provider time is 28.63–28.73 seconds.
- Diversity quota provider time is 28.82–28.94 seconds.
- Previous exhaustive residual multi-`w` provider time was about 40.44–40.71 seconds.
- Previous exhaustive diversity multi-`w` provider time was about 57.69–57.77 seconds.

Quota generation is therefore about 29% faster than the previous residual implementation and about 50% faster than the previous diversity implementation in these runs. It remains about 20% slower than baseline. These are measured Python CPU times for this workload, not general complexity claims.

## Controlled search results

All node-matched trials use ranks 7 and 8, seeds 3, 11, and 29, beam width 128, top-12 maximum branching, and 256 nodes. Search uses the 11/1 quota because it provides the maximum observed returned recall with the largest protected baseline prefix.

| Provider | Rank | Success | Nodes | Actions scored | Edges | Duplicate paths | Max depth | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bounded_128` | 7 | 0/3 | 741 | 94,848 | 3,048 | 2,142 | 7 | 24.17 s |
| `bounded_128` | 8 | 3/3 | 765 | 97,920 | 3,072 | 2,286 | 8 | 24.31 s |
| Residual quota 11/1 | 7 | 0/3 | 768 | 369,069 | 4,809 | 3,039 | 5 | 28.92 s |
| Residual quota 11/1 | 8 | 0/3 | 768 | 369,069 | 4,809 | 3,039 | 5 | 29.02 s |
| Diversity quota 11/1 | 7 | 0/3 | 768 | 370,725 | 5,184 | 3,267 | 5 | 29.13 s |
| Diversity quota 11/1 | 8 | 0/3 | 768 | 370,725 | 5,184 | 3,267 | 5 | 29.25 s |

All quota trials hit the node budget. No exact rank-7 or rank-8 decomposition is found. Baseline rank-8 again returns the conventional decomposition in all three trials. Those solutions pass exact reconstruction and have zero preparation additions and four post-additions.

The approximately evaluation-matched comparison uses 64 quota nodes. It scores 90,909 residual or 91,884 diversity complete actions across three trials, close to the baseline groups' 94,848–97,920. Every quota trial stops at the node budget, reaches depth 4, and fails both ranks. This is not an elapsed-time match: quota wall time is about 7.3 seconds per three-trial group versus about 24 seconds for full baseline search.

The previous exhaustive multi-`w` node-matched results are included by hash and copied aggregate in the new summary. The source is the committed `multi-w-provider-v1/summary.json`; unsuccessful trials are retained.

## Candidate diversity

The compact evidence includes preparation additions, residual reductions, output support, and total support for every returned candidate on the 10 recall states. The 11/1 residual quota returns 109 instances with preparation costs 0–3, residual reductions 0, 1, 2, 4, and 8, and output supports 1–2. The 11/1 diversity quota has the same reduction range and output supports but shifts the preparation-cost distribution. Larger quotas introduce output-support-three actions and broader preparation costs without improving known-action recall.

This variation could support later reranking studies. It is not evidence of lower GPU cost, and no GPU experiment was run.

## Search-interference diagnosis

The conventional eight-term path was checked directly. At every depth, its expected next action is baseline rank 1 and appears as a protected baseline action in the 11/1 provider. Provider-level displacement is therefore not the rank-8 failure mechanism.

The extra alternative creates additional child residuals. Candidate edges rise from roughly 3,000 to 4,809–5,184 per three-trial group. Absolute duplicate paths also rise. Global beam sorting is based on child residual norm, residual value, and path identity rather than provider quota class. Alternative children consume the 128-state layer capacity and the 256-node budget. Quota trials reach only depth 5, while baseline reaches depth 8 and solves. Protected candidate availability does not preserve search scheduling.

## Artifacts and reproduction

```sh
PYTHONPATH=src python scripts/run_tensor_game_quota_provider.py \
  --output artifacts/tensor-game/quota-provider-v1
```

The command refuses to overwrite existing output. The archive contains:

- `recall.json`: 70 state/configuration recall records with returned candidate identities and costs.
- `trials.json`: all 30 current search trials, including failures, traces, counters, and exact solutions.
- `summary.json`: recall, cost, search, and prior-study comparison aggregates.
- `configuration.json`: exact command, seeds, and ranks.
- `artifact_hashes.json`: file sizes and SHA-256 hashes.

## Limitations and next experiment

The top-`k` derivation relies on the current uncoupled coefficient domain. A coupled normalization or support constraint would require a different method. The 10 recall states follow known Strassen trajectories. They do not measure arbitrary-state proposal quality. The evaluation-matched search is intentionally shallower and cannot establish full-budget solvability.

The primary objectives are only partly met. Exact top-`k` is correct and substantially cheaper. Baseline actions are protected. Returned Strassen recall improves to 4/10. Rank-8 reliability is not preserved because the beam scheduler ignores candidate provenance.

The smallest justified next experiment is provenance-aware beam admission, not another proposal expansion. Reserve part of each next-layer beam for children reached through protected baseline edges, while keeping the overall beam and node budgets fixed. First require exact 3/3 rank-8 baseline-path retention. Only then test whether a small alternative-child quota improves rank-7 search.
