# Algebraic pair-to-output completion

## Result

Exact output-slice alignment improves complete-action coverage on the frozen
algebraic pair set, but local completion remains insufficient for rank-7
discovery. With two output vectors per pair and a final pool of 128 actions,
slice projection retains two of seven Strassen tensors. Exact residual top-2
retains one. No tested strategy retains more than two, and matched bounded
meet-in-the-middle runs find no complement or exact decomposition.

This is an offline CPU diagnostic. Existing pair generators, `bounded_128`,
multi-`w`, quota providers, search, feasibility pruning, and coefficient domains
are unchanged.

## Frozen pair set

The study reproduces the previous algebraic generator's 192 ordered mode-aware
factor-pair proposals. Each record preserves its source mode, exact coefficient
vectors, rank-two slice-combination provenance, continuous-relaxation heuristic
score, support descriptors, and original order. The deterministic manifest hash
is:

```text
3ce6b1414bc47e93a7bfa79145077b757397f5792adf25b32c8260740995bf2a
```

The prior study counted factor-pair recall across all three modes. Only 64 pairs
are native mode-2 `(u,v)` proposals. To apply one output-completion experiment to
the complete frozen set, this diagnostic treats every recorded coefficient pair
as `(u,v)` while retaining its source mode. That reinterpretation is a heuristic.
It does not assert that a mode-0 `(v,w)` or mode-1 `(u,w)` proposal has the same
tensor-mode meaning as an input pair.

As a sign-normalized coefficient-pair set, the manifest contains input pairs for
six of seven Strassen terms. Term 1 is absent, so complete-action recall cannot
exceed 6/7 under any output strategy.

## Output domain and exact score

`w` has four independent coordinates in `{-1,0,1}`. The all-zero vector is
prohibited because every rank-one factor must be nonzero. There are therefore
`3^4 - 1 = 80` valid vectors. There are no support, normalization, or sign
couplings between coordinates.

For fixed `u,v`, define

```text
q   = ||u||² ||v||²
d_k = <u outer v, residual output slice k>
E(w) = ||R||² + q ||w||² - 2 d dot w.
```

All quantities are integers. The existing exact top-`k` heap exploits coordinate
separability. Complete actions are canonicalized across the four factor-sign
representations. Exact coefficient identity can therefore differ while the
rank-one tensor is identical.

The required held-out output ranks under residual scoring are:

| Strassen term | Pair present | Residual `w` rank |
|---:|---:|---:|
| 1 | No | — |
| 2 | Yes | 8 |
| 3 | Yes | 10 |
| 4 | Yes | 2 |
| 5 | Yes | 3 |
| 6 | Yes | 2 |
| 7 | Yes | 4 |

This confirms that output completion is a limitation, but not the only one.

## Completion strategies

Every strategy receives the same 192 pairs in the same order. For controlled
accounting, every pair's full 80-vector domain is exact-scored even when the
existing top-`k` heap needs fewer complete-vector pops.

- **Residual:** reuse the existing exact top-`k` ordering by `E(w)`.
- **Structural:** anchor on the best residual vector, then greedily prefer new
  support sizes, sign patterns, maximum Hamming distance, exact score, and
  lexicographic coefficients.
- **Slice projection:** use the exact slice contractions `d`. Rank positive
  orientations by `(d dot w)^2 / ||w||²`, then by alignment, support, and
  coefficients. For ternary `w`, `||w||²` equals its support. This favors exact
  contraction direction rather than residual magnitude.
- **Hybrid:** reserve half of the local quota for residual top-`k`, then fill
  from structural maximin choices.

Projection is a local correlation heuristic. It does not prove global utility
or membership in a seven-term decomposition.

After local selection, canonical actions are deduplicated and all strategies use
the same exact residual-score final ranking. This isolates generated output
coverage while retaining a common final-pool rule.

## Controlled coverage

Per-pair budgets are 1, 2, 4, and 8. Final pools contain 20, 64, or 128 actions.
All smaller pools have zero complete-action recall. Results below show size 128:

| Strategy | `w=1` | `w=2` | `w=4` | `w=8` |
|---|---:|---:|---:|---:|
| Residual | 0 | 1 | 0 | 0 |
| Structural | 0 | 0 | 0 | 0 |
| Slice projection | 0 | **2** | 0 | 0 |
| Hybrid | 0 | 0 | 1 | 0 |

Tensor-equivalent recall equals canonical-action recall in every setting.
Slice top-2 retains terms 4 and 5 at final ranks 97 and 96. Residual top-2
retains term 4 at rank 127. Term 5 is the new recovery: its residual `w` rank is
3, so exact projection exposes a vector outside greedy top-2.

Coverage is non-monotonic in the per-pair budget. Adding candidates creates more
low-residual-score actions before the common 128-action cutoff and can displace a
held-out term. This is final-pool interference, not loss of local generation.
It also shows why “more `w` values” is not itself a useful selection policy.

## Candidate structure and work

At `w=2`, every strategy exact-scores `192 × 80 = 15,360` valid vectors for the
matched audit, selects 384 pair/vector instances, deduplicates to 256 canonical
actions, and retains 128.

| Strategy, `w=2` | Total completion | Matched exact scoring | Score range | Mean score | Distinct sign patterns |
|---|---:|---:|---:|---:|---:|
| Residual | 103.2 ms | 22.6 ms | 7–10 | 8.56 | 18 |
| Structural | 111.9 ms | 22.4 ms | 7–11 | 9.06 | 11 |
| Slice projection | 104.7 ms | 22.9 ms | 7–10 | 8.81 | 28 |
| Hybrid | 112.9 ms | 22.1 ms | 7–11 | 9.06 | 11 |

These are Python timings on the recorded host. Structural and hybrid selection
perform additional maximin work. Slice projection exposes more sign patterns and
retains 48 actions that increase squared residual by two, versus 32 for residual
top-2. The newly recovered term is verified against Strassen only after pool
construction. Its local harm is not independent evidence of global usefulness.

At `w=8`, strategies select 1,536 pair/vector instances and deduplicate to 1,024
actions. Residual completion takes about 278 ms, slice projection 283 ms, and
structural/hybrid about 418–421 ms. Comparing only elapsed time would be
misleading without these work counts.

## Meet-in-the-middle evaluation

Because slice projection materially improves recall, residual top-2 and slice
top-2 enter a matched bounded MITM comparison. Size-20 runs completely enumerate
1,140 left and 4,845 right combinations. Sizes 64 and 128 sample 10,000
combinations per side with seeds 20261017 and 29. Storage is capped at 10,000
sums and each run has a 30-second limit.

No run produces a complementary sum or exact decomposition. Complete size-20
left-side duplicate counts are 48 for residual and 80 for slice projection.
Sampled size-128 runs store 9,986–9,991 unique residual sums and 9,982–9,988
slice-projection sums. Peak traced memory stays below 19.9 MB. Sampled runs take
9.4–10.0 seconds. These null results cover only the recorded pools and samples.

## Failure analysis and recommendation

Output completion is a real bottleneck: five present pairs require residual
`w` ranks 2–10, and projection recovers one term excluded by residual top-2.
Still, local rules face three independent limits:

1. The frozen set lacks term 1's input pair.
2. Per-pair selection misses high residual-rank outputs.
3. Common global residual ranking displaces cancellation-heavy actions as the
   generated set grows.

The best recall remains 2/7 and produces no partial-sum complement. Structural
diversity alone is worse than residual top-`k`. More local candidates without a
joint objective increase interference.

The evidence does not justify integrating another local provider into search.
The next architecture should investigate a bounded joint decomposition model
over frozen pair/action variables, with exact reconstruction constraints and
strict CPU limits. A small integer-programming, SAT/SMT, or exact-cover
feasibility prototype would directly model cancellation across terms. It should
first use the current finite pools and remain separate from production search.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_algebraic_w_completion.py \
  --output artifacts/tensor-game/algebraic-w-completion-v1
```

The command refuses to overwrite evidence. The archive preserves the frozen
manifest, configuration, all completion/coverage records, retained pools, raw
MITM results, summary, and file hashes. Known Strassen terms are used only for
post-construction evaluation.
