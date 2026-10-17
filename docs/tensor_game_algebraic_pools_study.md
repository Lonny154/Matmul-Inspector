# Algebraically informed Tensor Game action pools

## Result

Exact slice information produces useful new action coverage, but not an
independent rank-7 decomposition under the tested budgets. An algebraic-only
pool of 128 actions recalls six of seven held-out Strassen mode-pairs and one
complete Strassen rank-one tensor. Residual-ranked and structural baselines
recall five and three pairs, and neither recalls a complete term. The 50/50
hybrid recalls five pairs and no complete term.

This is a CPU-only offline study. It does not change `bounded_128`, multi-`w`,
beam admission, feasibility pruning, residual scoring, or production search.

## Domain and provider audit

Every factor uses the 80 nonzero vectors in `{-1,0,1}^4`. The complete raw
domain therefore contains `80^3 = 512,000` factor triples. Four two-factor sign
flips represent each nonzero rank-one tensor, leaving 128,000 canonical tensors.
Canonicalization selects the lexicographically smallest equivalent factor triple.

The 6,400 `(u,v)` pairs used by a provider are only pair coverage. Each pair has
80 legal `w` vectors. A deterministic single-`w` rule samples one completion and
does not cover the complete rank-one domain. The residual baseline in this study
ranks all 128,000 canonical tensors. The structural baseline selects 128 of 6,400
pairs and evaluates exact top-4 `w` choices.

## Slice and subspace definitions

For each mode, fixing one tensor coordinate gives four exact `4×4` slices. A
ternary vector `c` defines the contraction

```text
M(c) = sum_i c_i slice_i.
```

Ranks, reduced row spaces, column spaces, and right null spaces are computed over
the rationals. No floating-point tolerance is used. The root multiplication
tensor has rank-2 individual slices in every mode. Across all 80 nonzero ternary
combinations per mode, 32 have rank 2 and 48 have rank 4. None has rank 1. The
common ternary row-space and column-space intersections of all four slices are
empty in every mode.

This rules out the simplest proposed generator at the root: there is no
rank-one matrix in the tested ternary slice combinations to factor directly.
It does not prove that the complete rational slice span contains no rank-one
element, and it does not imply that useful decomposition factors must lie in a
slice span.

## Bounded algebraic generator

The implemented fallback uses exact rank-two combinations:

1. Enumerate 80 ternary combinations in each of three modes.
2. Retain the 96 combinations with exact rank two.
3. Extract normalized ternary directions actually present in their nonzero rows
   and columns.
4. Form mode-aware pairs of those directions and deduplicate them.
5. For every pair, score all 80 legal missing factors exactly.
6. Retain four missing factors per pair, then deduplicate complete actions.

This yields 192 pair proposals, 15,360 exact complete-action scores, and 662
unique actions. Generation took 0.50 seconds on the recorded host. Each action
is validated in the ternary domain and carries its mode, source combinations,
matrix ranks, pair identity, and missing-factor rank.

The row/column rule is a heuristic motivated by exact low-dimensional subspaces.
It is not a theorem that a rank-two contraction's observed row and column
directions occur in a tensor-rank decomposition. Four `w`-equivalent missing
factors avoid the deterministic single-factor bottleneck, but still sample only
5% of the legal completions for each pair.

## Matched pool construction

Pools are evaluated at sizes 20, 64, and 128. The hybrid reserves half of each
size for algebraic actions and fills the rest in full-domain residual rank order.
Known Strassen coefficients are not used for generation, filtering, ordering,
quota selection, or parameter selection.

| Pool, size 128 | Pair work | Exact scores | Generation time | Coarse structural signatures | Score range | Mean score |
|---|---:|---:|---:|---:|---:|---:|
| Residual ranked | 6,400 pairs / 512,000 raw actions | 128,000 | 4.29 s | 13 | 7–8 | 7.94 |
| Structural diverse | 6,400 pairs, 128 retained | 638 | 0.18 s | 59 | 7–20 | 11.22 |
| Algebraic only | 240 slice combinations, 192 pairs | 15,360 | 0.50 s | 19 | 7–10 | 8.96 |
| Hybrid algebraic/residual | Reuses both sources | Reuses both sources | negligible merge | 13 | 7–9 | 8.03 |

Signatures group actions by factor support sizes, unordered positive/negative
counts, and preparation additions. They are intentionally coarser than action
identity. Structural diversity is largest for the existing structural baseline;
the algebraic improvement is targeted pair coverage rather than generic novelty.

## Held-out Strassen coverage

| Pool | Size 20 pair/action | Size 64 pair/action | Size 128 pair/action | Tensor-equivalent recall at 128 |
|---|---:|---:|---:|---:|
| Residual ranked | 1/0 | 3/0 | 5/0 | 0 |
| Structural diverse | 1/0 | 2/0 | 3/0 | 0 |
| Algebraic only | 0/0 | 3/0 | **6/1** | **1** |
| Hybrid | 1/0 | 3/0 | 5/0 | 0 |

“Pair” means at least one sign-normalized pair across the three tensor modes is
present. “Action” uses the repository's four-way factor-sign canonicalization.
Exact stored coefficient identity is zero for every pool; the algebraic hit is a
sign-equivalent factorization of Strassen term 3 and represents the same tensor.

For algebraic-only at size 128, term 1 has no recalled mode-pair. Terms 2, 4,
and 5 expose one pair. Terms 3, 6, and 7 expose all three pairs. Only term 3's
missing factor survives the four-per-pair selection and pool cutoff. Pair
generation therefore improves materially, while missing-factor selection and
final pool ordering remain the main exclusion for five terms.

## Residual and subspace discrimination

The residual pool contains only scores 7 and 8. Algebraic-only includes 76
actions with scores 9 or 10, so exact slice structure admits locally harmful
actions excluded by immediate residual ranking. Its 19 coarse signatures exceed
the residual pool's 13, but remain far below the structural pool's 59.

The bounded first-32 diagnostic records exact before/after flattening ranks,
slice ranks, support, coefficient magnitude, preparation cost, and residual norm.
These features show structural differences without treating lower flattening
rank as proof of lower tensor rank. The raw records are preserved in
`candidate_diagnostics.json`.

## Meet-in-the-middle evaluation

For every strategy, the size-20 run completely enumerates 1,140 three-action and
4,845 four-action combinations. Size-64 runs use seeds 20261017 and 29 and sample
10,000 combinations per side. Storage is capped at 10,000 sums and each run has
a 30-second wall limit.

No configuration produced a complement match or exact decomposition. Complete
size-20 duplicate-sum counts were 6 residual, 0 structural, 48 algebraic, and 6
hybrid. Across size-64 samples, algebraic-only stored 9,919 and 9,928 unique left
sums; residual stored 9,745 and 9,713; structural stored 10,000 in both; hybrid
stored 9,781 and 9,751. Duplicate partial sums are not solutions.

Sampled runs took 9.96–10.30 seconds with peak traced memory below 19.8 MB.
Complete runs took 3.32–3.43 seconds with peak traced memory below 2.01 MB. These
are Python measurements with `tracemalloc` enabled, not portable speed claims.

## Failure analysis and next experiment

The strict rank-one slice-combination hypothesis fails at the root. The rank-two
fallback finds a broader and more relevant set of held-out pairs, proving exact
slice structure can add useful coverage. Its benefit arrives late in the pool:
size 64 has no complete-action improvement, and only one of six recalled terms
receives the needed missing factor by size 128. A 50/50 hybrid dilutes that term.
No tested partial-sum sample suggests a rank-7 completion.

The next experiment should keep the 192 algebraic pairs fixed and compare
missing-factor quotas or pair-balanced complete-action admission under one fixed
total score budget. It should be preregistered without inspecting further
Strassen identities. This directly tests the measured exclusion point. A larger
slice-combination coefficient domain or general rational subspace enumeration
would add much more work and is not yet justified.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_algebraic_pools.py \
  --output artifacts/tensor-game/algebraic-pools-v1
```

The output path must not exist. The archive contains configuration, exact slice
analysis, generation audits, pools with provenance, held-out coverage records,
bounded candidate diagnostics, raw meet-in-the-middle results, a summary, and
SHA-256 hashes. The oracle is evaluation-only. No oracle-assisted result is
reported as discovery.
