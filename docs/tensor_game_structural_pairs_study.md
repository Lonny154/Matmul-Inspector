# Structural diversity in Tensor Game pair proposals

## Research question

The Strassen coverage study showed that five of seven possible first terms lie outside the residual-ranked 128-pair prefix. This CPU experiment asks whether a small trajectory-agnostic structural quota can recover useful `(u,v)` pairs without changing exact residual scoring, beam admission, or the coefficient domain.

The implemented selector increases measured structural diversity but does not improve exact known-pair recall. Rank-8 reliability remains 3/3 and rank 7 remains 0/3. The negative result separates generic feature diversity from the specific coverage needed by rank-7 decompositions.

## Existing limitation and insertion point

The unchanged provider enumerates 6,400 ternary pairs, ranks them by the continuous-relaxation bound, retains 128, derives `w`, canonicalizes complete actions, exactly scores them, and returns at most 12.

The optional `structural_pairs` policy changes only the 128-pair selection step. It preserves pair enumeration and heuristic scores. Downstream deterministic or top-k `w`, exact complete-action scoring, state updates, and global beam admission retain their existing semantics. `bounded_128`, multi-`w`, and quota-provider implementations are untouched.

## Structural features

Each pair receives these residual-derived features:

- `u` and `v` support and combined support.
- Unordered positive/negative coefficient counts for each input factor.
- Input sign patterns normalized so the first nonzero coefficient is positive.
- Unshared preparation additions.
- Contraction support, L1 magnitude, maximum magnitude, and sorted absolute-magnitude profile.
- The unchanged pair heuristic score for diagnostics.

The selection signature uses supports, normalized sign patterns, preparation additions, contraction support, and the absolute contraction profile. It is invariant to independent global sign flips of `u` and `v`. The heuristic bound and absolute contraction features are also sign invariant. Features are not invariant to swapping `u` and `v`, because they belong to distinct tensor modes. They are not invariant to arbitrary index permutations or general basis transformations. Contraction features depend on the current residual state.

## Selection algorithm

For a total budget of 128:

1. Keep the requested residual-ranked prefix in its original order.
2. Collapse remaining sign-equivalent structural signatures to their best residual-ranked representative.
3. Greedily select the signature with the largest minimum Hamming distance from already selected signatures.
4. Break distance ties by original residual rank.
5. Fill any unused slots from the original residual ranking.

No known Strassen coefficient or action identity participates in selection. The 128/0 configuration returns the original pair prefix exactly.

## Controlled pair coverage

The diagnostic reuses the same ten exact Strassen trajectory states as the quota-provider study. Top-k `w` is fixed at two for all complete-action comparisons.

| Residual/structural pairs | Pair recall | Newly exposed pairs | Complete actions generated | Returned top 12 | Mean signatures | Pair-selection time |
|---:|---:|---:|---:|---:|---:|---:|
| 128/0 | 5/10 | 0 | 5/10 | 4/10 | 39.1 | 0.31 s |
| 120/8 | 5/10 | 0 | 5/10 | 4/10 | 46.0 | 1.42 s |
| 112/16 | 5/10 | 0 | 5/10 | 4/10 | 51.5 | 1.48 s |
| 96/32 | 5/10 | 0 | 5/10 | 4/10 | 61.5 | 1.59 s |

Times are totals over ten states. The 128/0 diagnostic performs a full sort for comparability and is not the production streaming provider timing.

Initial-state signature counts rise from 64 at 128/0 to 68, 72, and 80. The 96/32 selector includes pairs with original residual ranks as low as 5,698. It broadens preparation costs and contraction supports. Initial preparation-cost-three and cost-five pairs are absent from 128/0 but appear under structural quotas. Contraction-support-three pairs also appear.

Despite this broader distribution, no structural configuration adds an exact known Strassen pair. The same five states are covered in every configuration. Structural similarity is not counted as exact recall.

## Complete-action coverage and residual quality

Because pair recall is unchanged, fixed top-2 `w` generation remains at 5/10 known complete actions and final top-12 return remains 4/10. The initial Strassen action is still unavailable.

Across the ten coverage states, none of the top-12 returned actions has negative immediate residual reduction. Structural exploration reaches locally worse pairs, but unchanged complete-action scoring removes their negative-reduction actions before the search interface. Pair diversity therefore does not pass through the final residual-ranked cutoff.

This is the central result: more structural signatures do not imply more useful exact actions, and pair-stage exploration alone cannot overcome the downstream local objective.

## Matched CPU search

Search uses ranks 7 and 8, seeds 3, 11, and 29, beam width 12, node budget 256, and maximum branching 12.

| Provider | Rank 7 | Rank 8 | Max depth | Nodes, rank 8 | Complete actions, rank 8 | CPU time, rank 8 | Negative returned actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original baseline | 0/3 | 3/3 | 8 | 225 | 28,800 | 7.10 s | 0 |
| Structural 96/32, deterministic `w` | 0/3 | 3/3 | 8 | 213 | 27,264 | 34.08 s | 0 |
| Structural 96/32, top-2 `w` | 0/3 | 3/3 | 8 | 255 | 92,697 | 42.57 s | 3 |

Every successful solution passes exact reconstruction. No configuration selectively receives more nodes or beam capacity.

The deterministic structural provider is about 4.8 times slower than baseline for the three rank-8 trials. Top-2 `w` is about 6.0 times slower and evaluates 3.2 times as many complete actions. These are Python CPU measurements for this implementation. They are not general complexity or hardware-performance claims.

Top-2 search does return three negative-reduction action instances across the three rank-8 trials. This proves such actions can cross the final interface in reached states. It does not improve rank-7 discovery and does not establish that those actions belong to a useful completion.

## Interpretation

The controlled structural quota meets the deterministic and baseline-preservation requirements. It also materially increases signature, preparation-cost, contraction-support, and original-rank diversity. It fails the primary coverage objective: exact known-pair recall does not improve at any tested quota.

The distance metric treats many algebraically different signatures as novel, but only 8–32 representatives can be selected from a much larger signature space. The required Strassen-compatible pairs are not singled out by this generic distance. Adding more arbitrary novelty would increase overhead without evidence that it targets useful contractions.

The downstream exact residual objective remains another bottleneck. Even if a locally harmful pair is selected, its complete action usually remains outside the top 12. The prior width-437/439 study found required complete-action ranks of 128–146 on full known trajectories. This experiment is consistent with that result.

## Reproduction and evidence

```sh
PYTHONPATH=src python scripts/run_tensor_game_structural_pairs.py \
  --output artifacts/tensor-game/structural-pairs-v1
```

The command refuses to overwrite existing output. Evidence includes:

- `coverage.json`: all 40 state/configuration pair and complete-action records.
- `trials.json`: all 18 matched search trials.
- `summary.json`: coverage and search aggregates.
- `configuration.json`: exact command, environment, budgets, and Git base.
- `artifact_hashes.json`: SHA-256 hashes and sizes.

No GPU benchmark, neural training, lookahead, beam change, or known-solution-directed selection was used.

## Recommended next experiment

Do not add a larger generic structural quota yet. The smallest justified next experiment is an offline, trajectory-agnostic contraction-quality objective that distinguishes algebraically complementary contractions from merely distant signatures. Candidate examples include residual-subspace coverage or a bounded two-step residual estimate. Evaluate it first on pair recall and complete-action rank with the total 128-pair and top-12 budgets fixed. Only integrate it into search if it exposes new exact actions without losing the rank-8 control.
