# Tensor Game meet-in-the-middle study

## Result

A bounded exact meet-in-the-middle diagnostic recovers Strassen when its seven
terms are supplied as an isolated oracle pool. It does not independently find a
rank-7 decomposition in residual-ranked, structurally diverse, or deterministic
mixed pools under the tested budgets.

This is a CPU-only offline result. It does not change the production provider,
beam search, feasibility gate, scoring rule, or ternary coefficient domain.

## Exact formulation

For the `4×4×4` matrix multiplication tensor `T`, the diagnostic seeks disjoint
sets of pool actions such that

```text
A = a1 + a2 + a3
B = b1 + b2 + b3 + b4
A = T - B
```

All additions and complement calculations use immutable integer tensors. A
SHA-256 digest selects a lookup bucket, then exact tensor equality confirms the
match. Action indices inside each side are increasing. The two sides must be
disjoint. Reconstructed seven-term sequences are validated by direct exact
summation. Solutions are deduplicated by their sorted canonical action IDs.

## Why the full domain is infeasible

There are 80 nonzero vectors in `{-1,0,1}^4` and 512,000 raw factor triples.
Every nonzero rank-one tensor has four equivalent two-factor sign flips, leaving
128,000 canonical rank-one tensors.

Even the unordered two-term multiset space has 8,192,064,000 entries. At an
optimistic 64 bytes per entry it needs at least 524,292,096,000 bytes. The
three-term and four-term multiset spaces contain 349,533,525,376,000 and
11,185,334,962,176,032,000 entries. A full-domain 3+4 search is not a practical
CPU experiment and was not attempted.

The implementation therefore exposes explicit pool size, left and right work,
stored-sum, wall-time, and random-seed limits. It enumerates a side completely
only when its combination count fits the work budget. Larger spaces use seeded
sampling without replacement. Repeated use of one pool action is excluded in
this version.

## Pool construction and leakage control

The three ordinary pools contain 64 canonical actions each:

- `residual_ranked` ranks the complete 128,000-action domain by the existing
  exact immediate residual norm and deterministic coefficient tie breaks.
- `structural_diverse` uses the existing structural pair selector and its
  signatures, taking one action per signature before score-ranked fill.
- `deterministic_mixed` interleaves the other two pools and removes duplicates.

None of these constructors reads Strassen terms or identities. The artifact
reports oracle overlap only after construction. The `strassen_oracle` pool is
separate and is used only to validate addition, hashing, complement lookup,
ordering, reconstruction, and deduplication.

## Controlled experiment

Each ordinary pool has two configurations. `complete_small` uses its first 20
actions and exhaustively checks all 1,140 three-action combinations and all
4,845 four-action combinations. `sampled_large` uses all 64 actions and samples
25,000 of 41,664 possible left combinations and 25,000 of 635,376 possible
right combinations. Seed 20261017 fixes sampling.

| Pool | Configuration | Left duplicate sums | Exact complements | Exact rank-7 decompositions | Runtime (s) | Peak traced memory |
|---|---|---:|---:|---:|---:|---:|
| Residual ranked | Complete 20 | 6 | 0 | 0 | 3.35 | 2.00 MB |
| Residual ranked | Sampled 64 | 1,121 | 0 | 0 | 25.53 | 48.66 MB |
| Structural diverse | Complete 20 | 0 | 0 | 0 | 3.46 | 2.01 MB |
| Structural diverse | Sampled 64 | 0 | 0 | 0 | 25.99 | 50.63 MB |
| Deterministic mixed | Complete 20 | 0 | 0 | 0 | 3.39 | 2.27 MB |
| Deterministic mixed | Sampled 64 | 156 | 0 | 0 | 25.70 | 50.46 MB |
| Strassen oracle | Complete 7 | 0 | 35 | 1 | 0.06 | 0.09 MB |

Runtime is Python wall time with `tracemalloc` enabled on the recorded host. It
is evidence for this run, not a performance claim. “Duplicate sums” means two
different left index sets produced the same tensor. Such a partial-sum collision
is not a decomposition. No hash collision was observed.

The oracle has 35 choices of three terms. Each has the complementary four terms,
so all 35 matches are expected. Canonical solution deduplication correctly turns
those partitions into one exact Strassen decomposition.

## Interpretation and limitations

Oracle recovery validates the mechanics but is not discovery. The ordinary
result is also limited: it rules out the complete 20-action prefixes and the
specific seeded samples from the 64-action pools. It does not rule out other
pools, unsampled combinations, repeated terms, or rank-7 decompositions in the
full ternary domain.

The optional residual-subspace restriction was not added. Earlier exact
flattening and slice features did not separate Strassen at the initial target,
and adding that restriction here would mix two experimental changes. A justified
next step is a second seed matrix or a rigorously defined subspace filter with
matched recall measurements. It should remain offline until it provides evidence
beyond the current exact pool baselines.

## Reproduction and evidence

```sh
PYTHONPATH=src python scripts/run_tensor_game_meet_in_middle.py \
  --output artifacts/tensor-game/meet-in-middle-v1
```

The output path must not exist. `configuration.json` records budgets and pool
audits. `domain_complexity.json` records exact domain sizes and storage lower
bounds. `pools.json` records actions and post-construction oracle overlap.
`experiments.json` contains raw counters, runtimes, memory, and reconstructed
solutions. `summary.json` separates oracle recovery, partial collisions, and
ordinary discovery. `artifact_hashes.json` authenticates the archive.
