# Tensor Game diversity-aware decomposition discovery

## Research question

This CPU-only study asks whether bounded diversity-aware search can find exact 2×2 matrix multiplication decompositions with arithmetic-cost profiles beyond the conventional and Strassen families in the Stage 1 dataset. It does not measure GPU performance or claim hardware efficiency.

## Existing search audit

`bounded_128` is a candidate provider. It is not a complete solver. For each exact residual it enumerates all 6,400 nonzero `u,v` pairs over `{-1, 0, 1}`, retains the best 128 by a continuous-relaxation bound, quantizes `w` into the same coefficient domain, and returns candidates ordered by exact squared residual norm. Coefficient tuples break score ties. The provider ignores its random-number argument and is deterministic.

`SampledMCTS` can use this provider at either a seven- or eight-action depth. It uses seeded PUCT, exact `(residual, remaining depth)` transpositions, node and simulation limits, and deterministic insertion order as the final selection tie-break. It continues simulations after a success but stores only one shortest solution. Thus its state table and result format do not retain alternate complete trajectories.

The experiment reuses the frozen `bounded_128` provider, exact tensor arithmetic, Stage 1 canonicalization, static cost descriptors, and Triton lowering metadata. It does not change the default solver or candidate policy.

## Discovery algorithm

The optional module performs bounded layer-by-layer beam search. Residual norm remains the primary ordering key. The baseline uses coefficient and content ordering for ties and retains one canonical prefix per exact residual. Diversity mode uses a deterministic seeded tie value and may retain up to four distinct canonical action-set prefixes per residual. Both modes expand the complete retained layer and collect every exact final trajectory instead of stopping at the first.

Every proposed solution is reconstructed with exact integer arithmetic. Empty, incomplete, or invalid trajectories are rejected before archive insertion. Exact representations are keyed by SHA-256 content hashes. Rank-term order and per-term sign-equivalent factorizations form the deliberately limited canonical identifier. The bounded archive prefers a previously unseen complete structural signature when duplicate signatures compete for space.

The diversity objective is applied only when retaining complete decompositions. Prefix diversity changes search tie handling but cannot make an invalid residual terminal. A different trajectory or term order is not counted as a new algorithm family.

## Search configuration

The command was:

```sh
PYTHONPATH=src python scripts/run_tensor_game_diversity_search.py \
  --output artifacts/tensor-game/diversity-search-v1 \
  --seeds 3 11 29 --ranks 7 8 \
  --node-budget 256 --beam-width 128 --max-actions 12 --archive-size 64
```

This produced 12 matched trials: two modes, two ranks, and three seeds. Baseline used one path per residual. Diversity mode allowed four. Compilation, GPU execution, training, and kernel tuning were not involved.

## Results

| Mode | Rank | Trials | Exact successes | Expanded nodes | Candidate edges | CPU wall time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7 | 3 | 0 | 741 | 3,048 | 22.04 s |
| Baseline | 8 | 3 | 3 | 765 | 3,072 | 23.18 s |
| Diversity | 7 | 3 | 0 | 741 | 3,048 | 22.43 s |
| Diversity | 8 | 3 | 3 | 765 | 3,072 | 23.42 s |

The complete experiment expanded 3,012 nodes in 91.07 seconds. No trial exhausted its 256-node limit. Rank 7 reached the requested depth but its best retained residual still required one conventional term. Rank 8 reconstructed the target in every trial.

Baseline produced one exact representation across its three successful trials. Diversity-aware tie-breaking produced three representations. All four differ only in term order. They collapse to one limited canonical class and match the conventional seed. The archive marks that relationship explicitly instead of labeling these as independent families.

The retained conventional class has rank 8, zero preparation additions, four reconstruction additions, zero nonunit coefficient scalings, and coefficient magnitudes of one. It has the same arithmetic-cost point as the conventional Stage 1 family. Candidate `12b28d005d3b0d787f2015cfa6f852e04652fdd9816b012d794cfc827d4cbebd` is a representative search output, not a new decomposition.

## Structural diversity and Stage 1 comparison

| Measure | Stage 1 | Search archive | New versus Stage 1 |
|---|---:|---:|---:|
| Exact coefficient representations | 128 | 4 | 4 |
| Limited canonical classes | 4 | 1 | 0 |
| Provenance families | 2 | 1 conventional match | 0 |
| Structural signatures | 11 | 1 | 1 |
| Primary arithmetic-cost points | 2 | 1 | 0 |

The one signature not present in Stage 1 reflects the sign-normalized representation emitted by `bounded_128`. It does not change rank, preparation additions, reconstruction additions, coefficient scaling count, or limited canonical family. It is representation-level variation, not meaningful new arithmetic structure.

Diversity-aware retention therefore yielded zero new canonical classes and zero new arithmetic-cost points per 1,506 expanded nodes. Its extra exact representations are not evidence of independent discoveries. The primary research question has a negative result under this budget and proposal set.

## Exact transformations

Stage 1 already implements rank-term permutation, valid per-term sign redistribution whose three signs multiply to +1, and simultaneous index swaps on the shared matrix-multiplication indices. These preserve each rank-one tensor or the complete multiplication tensor and are checked by exact reconstruction. General changes of basis would require applying mutually inverse transformations across the three tensor modes and may leave the `{-1, 0, 1}` coefficient domain. No new transformation was added here because a bounded integral change-of-basis family and canonical equivalence proof were outside this search experiment. Transformation-generated candidates remain separate from search outputs.

## Validation and artifacts

The archive contains full `U/V/W` coefficients, exact status, rank, content and canonical hashes, structural signatures, static costs, lowering metadata, search seed and budget, reproduction configuration, and known-family provenance. Serialization and ordering are deterministic. Existing Stage 1 files are untouched.

Raw evidence is in `artifacts/tensor-game/diversity-search-v1/`:

- `trials.json`: all successful and unsuccessful trials, effort counters, configurations, and timings.
- `candidates.jsonl`: the deduplicated exact archive.
- `summary.json`: aggregate counts and SHA-256 hashes. The recorded hashes are `45ee1aa5b882715ac26064126f9ba6030d133bc8e66c804747673ef40ad2eaf8` for `trials.json` and `d2840b33a95631417dedc4fa3d03a581c3e47d5a6a79eff54f1aa2c37097fe25` for `candidates.jsonl`.

## Limitations and next experiment

This is a small search over ternary factors. The 128-pair coarse prefix strongly favors sparse conventional residual reductions. Seeded tie-breaking cannot recover actions excluded before exact candidate ranking. Limited canonicalization does not classify index symmetries or general tensor isotopies. CPU wall time includes candidate generation and Python search bookkeeping and is not a kernel benchmark.

The next experiment should keep `bounded_128` as the required baseline but add a separately named exploratory proposal stream. It should sample residual-vetted candidates outside the fixed coarse prefix, then compare equal total proposal and node budgets. The acceptance archive should continue using exact reconstruction and complete-decomposition cost signatures. This directly tests whether proposal coverage, rather than archive retention, is the limiting factor.
