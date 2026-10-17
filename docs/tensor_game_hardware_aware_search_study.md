# Hardware-aware Tensor Game search

## Research question

The RTX 4060 Ti calibration found strong within-rank relationships between complete-decomposition preparation additions and 1024 latency: Spearman 0.941 for rank 7 and 0.986 for rank 8. This CPU experiment asks whether an optional local preparation penalty can guide exact `bounded_128` search toward cheaper complete decompositions without reducing solve reliability.

The calibration evidence comes from change-of-basis variants of only conventional multiplication and Strassen. It does not establish a local search heuristic or justify a fitted penalty. The tested penalties are small experimental values, not latency predictions.

## Ranking objective

For each valid action already produced and exactly scored by `bounded_128`, weighted ranking uses

```text
score(action) = exact residual reduction - lambda * local preparation additions.
```

Local preparation additions are `max(nonzeros(u)-1,0) + max(nonzeros(v)-1,0)`. This is the cost of forming that action's two input combinations without reuse. It differs from complete-decomposition preparation cost, which deduplicates identical `u` and `v` combinations across all rank terms. A locally cheap sequence is therefore not guaranteed to minimize global cost.

The implementation uses exact rational penalty arithmetic and deterministic baseline-rank tie-breaking. `lambda=0` preserves the original residual ordering exactly. The default `BoundedResidualPolicy` and `create_residual_candidate_provider()` are unchanged.

The optional Pareto policy maximizes exact residual reduction and minimizes local preparation additions. It puts nondominated candidates first while preserving original residual rank within each group. Both policies rerank only after the frozen 128-pair coarse stage, exact `w` quantization, action validation, exact residual scoring, and canonical deduplication.

## Search implementation

The existing bounded beam discovery engine now supplies optional depth context to instrumented providers. Providers without `set_context()` behave exactly as before. Hardware-aware search requests up to 128 already exact-scored candidates, reranks them, and retains the same maximum of 12 actions per expanded state. This larger reranking pool is necessary because the beam expands all returned actions; reordering only the same 12 would not change exploration.

Diagnostics record residual reduction, local cost, weighted score, rank before and after reranking, Pareto status, selection, search depth, remaining moves, and residual squared norm. Traces are capped at eight expanded states per trial. Aggregate counters cover every provider call. Complete solutions still pass exact tensor reconstruction before export and retain the existing cost, structural-signature, lowering, provenance, and content-hash records.

## Controlled CPU experiment

The matched configurations were:

- Policies: residual baseline, weighted `lambda=0.25`, `0.5`, and `1.0`, plus Pareto selection.
- Ranks: 7 and 8.
- Seeds: 3, 11, and 29.
- Node budget: 256.
- Beam width: 128.
- Selected actions per state: at most 12.
- Exact-scored reranking pool: at most 128.
- Coefficient domain: `{-1,0,1}` inherited from `bounded_128`.

```sh
PYTHONPATH=src python scripts/run_tensor_game_hardware_aware_search.py \
  --output artifacts/tensor-game/hardware-aware-search-v1 \
  --seeds 3 11 29 --ranks 7 8 --penalties 0 0.25 0.5 1 \
  --node-budget 256 --beam-width 128 --max-actions 12 \
  --candidate-pool 128 --trace-limit 8
```

## Results

| Policy | Rank | Successes | Trials | Nodes | Unique solutions | Pre / post / support |
|---|---:|---:|---:|---:|---:|---:|
| Residual baseline | 7 | 0 | 3 | 741 | 0 | — |
| Weighted 0.25 | 7 | 0 | 3 | 741 | 0 | — |
| Weighted 0.5 | 7 | 0 | 3 | 741 | 0 | — |
| Weighted 1.0 | 7 | 0 | 3 | 741 | 0 | — |
| Pareto | 7 | 0 | 3 | 741 | 0 | — |
| Residual baseline | 8 | 3 | 3 | 765 | 1 | 0 / 4 / 24 |
| Weighted 0.25 | 8 | 3 | 3 | 765 | 1 | 0 / 4 / 24 |
| Weighted 0.5 | 8 | 3 | 3 | 765 | 1 | 0 / 4 / 24 |
| Weighted 1.0 | 8 | 3 | 3 | 765 | 1 | 0 / 4 / 24 |
| Pareto | 8 | 3 | 3 | 765 | 1 | 0 / 4 / 24 |

Every policy matches baseline success, node count, and solution content in all six rank/seed trials. Rank-7 searches terminate at the depth limit without an exact solution. Rank-8 searches reconstruct the target exactly and return the same conventional decomposition. Across the entire archive there is one exact representation, one limited canonical class, one structural signature, and one arithmetic-cost point.

Wall time varies from about 22.2 to 24.1 seconds per three-trial rank group. These small differences are ordinary CPU timing variation and Python diagnostic overhead. They do not reflect a search-efficiency change because nodes, candidate instances, and selected actions are identical.

## Why the policy had no effect

Each policy made 1,506 provider calls across its six trials and evaluated 6,120 candidate instances. Every candidate had:

```text
local preparation additions = 0
exact residual reduction = 1
```

There were zero reranked candidate instances and zero changed selections for every weighted and Pareto configuration. `bounded_128`'s coarse prefix exposes only sparse conventional basis terms on every state reached by this beam. The penalty is mathematically constant on that support, so sensitivity to `lambda` cannot appear. Pareto selection likewise sees every candidate at the same objective point.

This also explains the rank-7 failures. Low-cost preference did not create a new dead end; the baseline provider had already restricted the reachable graph to conventional one-block actions. A rank-7 Strassen path requires multi-block input combinations that were absent from every visited exact-scored candidate set.

## Answers to the experimental questions

1. Hardware-aware reranking preserves baseline solve success exactly, but only because it never changes a selection.
2. Nodes and search depth are identical. Runtime differences are not meaningful search-efficiency evidence.
3. It produces no lower-cost complete decomposition. Every success is conventional with zero preparation additions, four reconstruction additions, and support 24.
4. It adds no structural diversity.
5. Results are insensitive to `lambda` from 0.25 through 1.0 because local costs are uniformly zero.
6. Pareto selection is identical to weighted and residual selection on the observed candidate support.
7. The penalty neither creates new dead ends nor changes depth. Rank-7 failure predates reranking and is caused by proposal coverage.

## GPU evaluation

GPU evaluation was skipped. No policy produced a new representation, a different complete cost point, or any changed search trajectory. Benchmarking the same conventional decomposition again would not test the hardware-aware hypothesis and would add another GPU session without new information. The relevant conventional GPU anchor is already preserved in the calibration study.

## Artifacts and validation

Artifacts are in `artifacts/tensor-game/hardware-aware-search-v1/`:

- `trials.json` contains all 30 trials, exact configurations, bounded traces, aggregate diagnostics, termination reasons, solutions, failures, and CPU timings. SHA-256: `d000fc5900b5530035496e1bba0fae8ec8a9204799d00e54d7d8631ac4644cf1`.
- `candidates.jsonl` contains the one exact deduplicated solution. SHA-256: `ce9390f39c8773b653c6240f63f2e1def5a0e04cdc92d2aa1c297c09ba9f739c`.
- `summary.json` contains policy/rank outcomes and source hashes.
- `analysis.json` contains matched baseline comparisons and full-policy diagnostic totals.
- `artifact_hashes.json` records file sizes and hashes.

All accepted decompositions use exact integer validation. No partial or failed trajectory enters the archive. Serialization is stable and outputs are written into a new directory without modifying previous experiments.

## Limitations and next experiment

This is a negative but diagnostic result. It tests hardware-aware selection only inside the action support exposed by `bounded_128`. It does not show that preparation cost is useless for broader algebraic search. The local action cost also omits cross-term reuse, output reconstruction, coefficient scaling, register use, shared memory, and search-to-complete cost interactions.

The next experiment should keep this baseline and exact post-score reranker, but add a separately named bounded proposal stream that retains some multi-block `u/v` candidates outside the current coarse prefix. Use an explicit quota or stratification by preparation cost rather than a larger unbounded search. First verify on CPU that the expanded pool contains both zero- and nonzero-preparation actions and can still recover the conventional solution. Only then repeat matched rank-7/rank-8 trials. GPU evaluation remains conditional on finding a new exact complete cost point.
