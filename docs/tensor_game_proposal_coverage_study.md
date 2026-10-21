# Tensor Game proposal-coverage study

## Research question

The hardware-aware search study saw only zero-preparation actions with residual reduction one. This CPU-only diagnostic asks where that homogeneity enters the pipeline. It separates coefficient-domain representability, the provider's deterministic `w` choice, the 128-pair heuristic cutoff, exact scoring, and downstream search.

The result is sharper than “the width is too small.” Nonzero-cost actions exist in the provider-reachable space and in the width-128 prefix on Strassen residuals. However, 7 of 10 tested next Strassen actions are incompatible with the provider's single `w` choice. Increasing the pair width from 128 to all 6,400 pairs leaves Strassen next-action recall unchanged at 3/10.

## Existing candidate pipeline

`bounded_128` uses the following deterministic pipeline:

1. Enumerate the 80 nonzero vectors in `{-1,0,1}^4` for `u` and `v`. Their Cartesian product gives 6,400 contractions.
2. For each pair, compute `q = ||u||² ||v||²` and `d_k = sum_ij R_ijk u_i v_j`.
3. Rank pairs by the continuous-relaxation improvement `sum(d_k²) / q`. The implementation compares an exact common-denominator integer numerator. Lexicographic `u,v` order breaks ties.
4. Retain the first 128 pairs. The streaming implementation uses an exact fixed-size heap and matches the full sort.
5. Derive one `w` by independently quantizing each `d_k/q` to `{-1,0,1}`. An all-zero `w` is rejected.
6. Canonicalize sign-equivalent actions. Duplicate complete actions collapse to one record.
7. Compute the exact squared residual score. Sort by score and then canonical `u,v,w` coefficients.
8. Return at most the caller's requested action count. Fewer are returned when fewer unique legal actions survive.

The full `(u,v)` enumeration is not the complete ternary `(u,v,w)` action domain. There are 80 possible nonzero `w` vectors for every pair, or 512,000 raw triples before equivalence handling. The provider evaluates at most one derived `w` per pair. This study exhausts the provider-reachable complete actions. It does not enumerate all 512,000 triples.

There is no explicit preparation-cost rejection or sparsity term. Sparse pairs often rank well because the continuous relaxation favors them on sparse residuals. The local cost remains `max(nnz(u)-1,0) + max(nnz(v)-1,0)`. It is appropriate as an unshared, per-action input preparation descriptor. It is not a decomposition-wide cost because it omits reuse across terms.

## Diagnostic method

The new `tensor_game_proposal_coverage` module reuses the existing contraction ranking, quantizer, canonicalizer, exact score identity, action identity, residual update, and hardware-cost descriptor. For each pair it records the heuristic rank, contraction, chosen `w`, bounded-128 membership, exact score when valid, and whether it duplicates an equivalent action. For every unique reachable action it records exact rank, residual reduction, support, coefficient magnitude, preparation cost, and exclusion reason.

Counterfactual widths 32, 64, 128, 256, 512, 1,024, and 6,400 are slices of the same pair ranking. No search is rerun. Pareto analysis maximizes exact residual reduction and minimizes local preparation additions. Equal objective points remain nondominated.

Reproduce the run with:

```sh
PYTHONPATH=src python scripts/run_tensor_game_proposal_coverage.py \
  --output artifacts/tensor-game/proposal-coverage-v1
```

The command refuses to overwrite an existing output directory.

## Representative states

The run contains 21 exact states:

- All eight forward conventional prefix states.
- Three reverse-order conventional prefix states at depths 1, 4, and 7.
- All seven forward Strassen prefix states.
- Three reverse-order Strassen prefix states at depths 1, 3, and 6.

Each row stores the term order, depth, residual hash, and expected next action. Subtracting that action produces the recorded next-residual hash. The two depth-zero rows cover the initial multiplication tensor with different expected next actions.

Historical rank-7 dead-end residuals were not included. The prior hardware-aware artifacts contain traces and action IDs but do not serialize exact residual tensors. Fabricating them from incomplete evidence would not be reproducible.

## Known-solution recall

| Family | Tested next actions | In ternary complete-action domain | Reachable under chosen `w` | Retained at width 128 |
|---|---:|---:|---:|---:|
| Conventional | 11 | 11 | 11 | 11 |
| Strassen | 10 | 10 | 3 | 3 |
| Total | 21 | 21 | 14 | 14 |

All known actions have `(u,v)` pairs in the enumerated domain. The seven missing Strassen instances are therefore not coefficient-domain failures and are not filtered out by the 128 cutoff. Their pair exists, but quantizing `d/q` selects a different `w`. The three reachable Strassen actions occur late in the tested trajectories. They have heuristic rank 1, exact rank 1, exact residual reduction 4 or 8, and preparation cost 2.

At the initial tensor, the first forward Strassen term is legal and its pair has heuristic rank 3,378. Its exact residual reduction is 2. The provider derives a different `w`, so even width 6,400 cannot emit that complete action. The conventional next action is reachable at pair rank 34 and exact rank 1.

## Candidate distributions

Across 21 states, the evaluator processed 134,400 contractions and found 435 unique provider-reachable complete-action instances. Of those, 311 had nonzero local preparation cost (71.5%). Width 128 retained 186 unique instances, 126 with nonzero cost (67.7%). This diversity is concentrated in Strassen-prefix states. Every conventional-prefix state, including the initial tensor, exposes only zero-cost candidates.

The per-state artifact records exact histograms for residual reduction, preparation additions, input support, output support, and maximum coefficient magnitude. It also records distinct-value counts and Pareto sizes. Averaged over states, width 128 recalls 77.8% of the full provider-reachable Pareto frontier. Full width recalls 100% by definition.

This does not contradict the earlier search trace. That search stayed on conventional residuals, where the diagnostic also finds uniform zero-cost, unit-reduction actions. Hardware-aware reranking has candidate variation on known Strassen residuals, but it has none on the states the baseline search actually reaches.

## Counterfactual proposal widths

| Pair width | Mean unique actions | Mean nonzero-cost fraction | Mean full-Pareto recall | All known-action recall | Strassen recall |
|---:|---:|---:|---:|---:|---:|
| 32 | 4.52 | 42.9% | 44.5% | 12/21 | 3/10 |
| 64 | 6.95 | 41.8% | 63.4% | 13/21 | 3/10 |
| 128 | 8.86 | 41.6% | 77.8% | 14/21 | 3/10 |
| 256 | 12.67 | 39.6% | 91.3% | 14/21 | 3/10 |
| 512 | 15.48 | 38.7% | 97.9% | 14/21 | 3/10 |
| 1,024 | 19.19 | 37.5% | 100% | 14/21 | 3/10 |
| 6,400 | 20.71 | 34.3% | 100% | 14/21 | 3/10 |

The implied exact-scoring pair workload grows linearly with width. Width 6,400 costs 50 times the pair evaluations of width 128 without recovering another known Strassen action. Width alone improves general Pareto coverage, but not the missing known-solution coverage.

## Runtime, memory, and artifacts

The controlled run used Python 3.13.11 on Linux under WSL2. It completed in 8.17 seconds and recorded a 10.87 MB peak Python allocation through `tracemalloc`. Exact evaluation plus descriptor construction totaled about 1.95 seconds across states. Separate calls to the unchanged width-128 provider totaled about 3.02 seconds. These are diagnostic wall times, not kernel measurements.

Evidence is under `artifacts/tensor-game/proposal-coverage-v1/`:

- `candidates.jsonl`: all 435 unique reachable complete-action instances.
- `states.json`: compact per-state distributions, known-action results, and timings.
- `summary.json`: aggregate recall and width results.
- `configuration.json`: exact state labels and command.
- `artifact_hashes.json`: SHA-256 hashes and byte sizes.

The checked-in run uses compact summaries and candidate-level records. Pass
`--include-pair-records` to additionally write every contraction-level record.
That optional file is about 33 MB for this state set and is not needed to
reproduce the aggregate results.

## Failure classification and interpretation

Representability is not the limiting factor: all tested Strassen actions use legal ternary coefficients. The heuristic cutoff is also not the primary loss point: every Strassen action that the `w` rule can construct is already inside width 128. Exact scoring does not discard those reachable actions. The dominant failure occurs at deterministic `w` selection, before exact complete-action ranking.

Proposal presence still does not prove search solvability. The three recalled actions lie on known valid Strassen trajectories, so those specific suffixes have valid completions. This experiment does not show that an unguided solver would enter those states or choose the right continuation.

The hardware-aware objective has enough variation to rerank candidates after the solver reaches a Strassen-like residual. It cannot influence the conventional-only states seen in the previous search. More importantly, it cannot rank a known action that the `w` rule never creates.

## Limitations and next experiment

The state set is deterministic but small. It emphasizes two known decompositions and two term orders. It does not include historical failed residuals because exact states were unavailable. The full-space label refers to all 6,400 provider contractions and their deterministic `w` values, not every ternary complete action. Local preparation cost does not model shared input combinations or final hardware latency.

The smallest justified architectural experiment is a separately named diagnostic provider that keeps a bounded number of alternative ternary `w` vectors for promising `(u,v)` pairs. It should guarantee inclusion of the quantized choice, use exact residual scoring for alternatives, and keep the total complete-action budget fixed for a fair comparison. First measure known Strassen recall and conventional regression on CPU. Do not change the default `bounded_128` provider or claim better rank-7 solve rate until a matched search experiment demonstrates it.
