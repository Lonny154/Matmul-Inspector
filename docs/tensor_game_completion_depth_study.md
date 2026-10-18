# TG-Global-3: completion depth and algebraic recombination

## Question

TG-Global-2 found no exact decomposition or two-term-repairable best state in
its 12-seed discovery pilot. This study asks whether those states contain
larger exactly completable cores and whether combining terms from two imperfect
states can expose one. This is a CPU-only diagnostic of the existing finite
rank-7 pool.

## Exact completion hierarchy

A complete state is a sorted tuple of seven indices into the canonical action
pool. The audit freezes every five-term, four-term, and three-term subset of a
candidate, giving 21, 35, and 35 cores. It computes `target - frozen_sum`
using the repository's exact integer tensor operations.

For two missing terms, the required tensor is probed in the existing pair-sum
table. For three, each pool action is subtracted and the remainder is probed
against the pair table. For four, each pair sum is subtracted and the
complementary pair is looked up. The existing SHA-256 tensor digest selects a
bucket. Exact tensor equality, repeated-action legality, and full seven-term
reconstruction decide whether a match is verified. Unique solutions are
deduplicated by canonical action identifiers.

The pair table has limits for pool size, pair records, unique pair sums, and
construction time. Each candidate audit has a probe and time budget. A core
can return `COMPLETE`, `NO_COMPLETION`, `PAIR_TABLE_LIMIT`, `LOOKUP_LIMIT`,
or `TIME_BUDGET`. `NO_COMPLETION` means only that the configured finite pool
was exhausted for that core. A limited core remains unknown. The report keeps
raw hash-confirmed complements, domain-valid matches, independent
reconstructions, and unique exact decompositions separate.

## Algebraic recombination

One bounded operator selects distinct low-energy parents by a seeded pairing
of the top unique candidates. It enumerates two-term subsets from each parent,
forms legal four-term unions, and scores their exact tensor residuals. Only the
best bounded number of partial children receive three-term exact completion.
The operator records parent hashes, selected action IDs, partial tensor hashes,
residual components, work counts, and verification results. A low residual is
a ranking heuristic. It is not exact correctness.

## Inputs and controls

The discovery study rebuilds the frozen `algebraic_slice_projection_128` pool
and checks its hash against the retained TG-Global-2 pilot. It loads every
seed/variant final best state from `paired_runs.jsonl`. Identical states share
one audit, while the candidate manifest preserves every source record. Since
TG-Global-2 saved scores but not intermediate states, a small replay explicitly
retains actual initial and strict best-energy incumbents from two seeds. It
does not invent intermediate candidates from score history.

The oracle control uses known Strassen actions. A separate recombination
control adds conventional filler terms to two artificial parents. Both are
labeled oracle-assisted. Discovery mode rejects oracle candidate artifacts and
oracle pair tables. Pool hash checks prevent a cached lookup from another pool
being used silently.

## Reproduction

The default command is bounded and uses the retained TG-Global-2 pilot:

```sh
PYTHONPATH=src python scripts/run_tensor_game_completion_depth_study.py \
  --output artifacts/tensor-game/completion-depth-discovery-v1
```

Use `--recombine` for the bounded parent-pair pilot. The output directory must
not exist. Artifacts include configuration, provenance, candidate sources,
per-candidate depth records, recombination records, aggregates, exact
candidates, and SHA-256 file hashes.

## Interpretation and decision gate

`CONTINUE_RANK7` requires non-oracle deeper completion structure or verified
recombination evidence. `SEARCH_LIMITED` applies when the pool has useful
completion structure that search rarely reaches. `POOL_LIMITED` applies when
bounded evidence points strongly to pool coverage. `INCONCLUSIVE` applies when
resource limits leave the relevant deeper neighborhood unresolved. The final
classification and measured counts are recorded with the retained artifacts.

Exact completion inside an oracle pool is not discovery. Recombination of
known or equivalent Strassen structure is not a novel multiplication
algorithm. Lower residual is not exact correctness. Bounded failure does not
prove no completion exists, and failure within a finite pool does not prove
rank-7 impossibility.

This study does not demonstrate GPU acceleration. Its Python complement
queries are small and irregular, so these measurements do not justify a GPU
experiment. This milestone determines whether continued rank-7 global search
is warranted before a possible pivot to Hardware-Aware Tensor Search. If it
produces no compelling non-oracle signal, the recommended next research track
is exact rank-8 and rank-9 search with hardware-oriented objectives, followed
by measurement of promising representations on the RTX 4060 Ti. Rank-8 and
rank-9 search are outside TG-Global-3.
