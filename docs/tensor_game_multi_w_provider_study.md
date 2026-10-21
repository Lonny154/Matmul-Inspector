# Multi-w Tensor Game candidate provider

## Research hypothesis

The proposal-coverage study found that deterministic output-factor selection made 7 of 10 tested next Strassen actions unreachable. This experiment asks whether an optional bounded multi-`w` provider can expose those actions and improve exact rank-7 discovery without excessive CPU cost.

The coverage result is positive but the search result is negative. Multi-`w` raises exact-action generation recall from 3/10 to 5/10. Only 4/10 reach the normal top-12 search interface. Neither experimental strategy finds a rank-7 decomposition, and both lose the baseline rank-8 solution under the matched node budget.

## Existing limitation and output-vector domain

The unchanged `bounded_128` provider ranks 6,400 ternary `(u,v)` pairs, retains 128, and derives one `w` by independently quantizing each coordinate of `d/q` to `{-1,0,1}`. It then rejects all-zero `w`, canonicalizes the action, deduplicates, and exact-scores it.

For the 2×2 game, `w` has four coordinates. Its only action-domain restrictions are ternary coefficient membership and a nonzero vector. The domain therefore contains 80 valid vectors out of `3^4 = 81`. For fixed `(u,v)`, exact score is

```text
||R||² - 2 dot(d,w) + q ||w||².
```

The experimental provider is inserted after pair retention and before action canonicalization. The default provider and its public factory are unchanged.

## Alternative-generation algorithms

`bounded_multi_w` always includes the original quantized nonzero `w` first. It ranks the full 80-vector domain with the existing exact score identity, then keeps a bounded per-pair set.

- `residual` keeps the best exact-scored alternatives after the original.
- `diversity` reserves half of the alternative slots for the best exact-scored vectors. It fills the rest greedily using new output-support sizes, new sign patterns, Hamming distance, exact score, and lexicographic `w` order.

Both methods are deterministic. The seed setting is recorded but unused because this milestone introduces no randomized strategy. Canonical complete actions are deduplicated and globally ranked by exact residual score and coefficient tuples.

The configurable controls are pair width, `w` candidates per pair, strategy, retained complete-action budget, candidate pool, trace limit, and seed. The controlled experiment uses width 128, four retained `w` choices per pair, a 512-action pre-deduplication ceiling, and the unchanged top-12 search branching limit.

## Candidate budgets and overhead

Choosing exact best alternatives by exhaustive local ranking evaluates all 80 valid `w` vectors for each of 128 pairs, or 10,240 complete actions per provider call. Four per-pair choices retain at most 512 pair/action instances before canonical deduplication. `bounded_128` evaluates 128 complete actions per call.

This distinction is important: the retained pool grows by at most 4×, but exact alternative ranking grows the complete-action evaluation count by 80×. The implementation records pair-ranking time, alternative-generation time, selected-action exact-scoring time, total provider time, full-domain actions scored, retained pair/action instances, and returned actions separately.

## Known Strassen recall

The diagnostic reuses exactly the 10 Strassen states and denominator from the proposal-coverage study.

| Provider | `w` per pair | Exact action generated | Exact action returned top 12 | Equivalent child returned |
|---|---:|---:|---:|---:|
| `bounded_128` | 1 | 3/10 | 3/10 | 3/10 |
| residual multi-w | 2 | 5/10 | 4/10 | 4/10 |
| residual multi-w | 4 | 5/10 | 4/10 | 4/10 |
| residual multi-w | 8 | 5/10 | 4/10 | 4/10 |
| diversity multi-w | 2 | 5/10 | 4/10 | 4/10 |
| diversity multi-w | 4 | 5/10 | 4/10 | 4/10 |
| diversity multi-w | 8 | 5/10 | 4/10 | 4/10 |

The newly exposed actions are forward depths 3 and 4. Both have `(u,v)` pair ranks inside 128. The depth-4 action reaches global complete-action rank 3 and is returned. The depth-3 action is generated but ranks 19th, outside the unchanged top-12 interface. Larger per-pair counts add no known-action recall.

The other five missing actions have pair ranks outside 128. Multi-`w` cannot recover them without changing pair coverage. Exact-action and same-child recall are reported separately; they are equal here, so no gain comes from relaxed equivalence.

## Controlled CPU search

The search comparison uses ranks 7 and 8, seeds 3, 11, and 29, beam width 128, top-12 branching, exact integer state transitions, and exact decomposition validation. Node-matched runs use up to 256 nodes. Evaluation-matched runs give multi-`w` three nodes because each call scores 80 times as many complete actions.

### Node-matched results

| Provider | Rank | Success | Nodes | Complete actions scored | Candidate edges | Duplicate paths | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bounded_128` | 7 | 0/3 | 741 | 94,848 | 3,048 | 2,142 | 23.57 s |
| `bounded_128` | 8 | 3/3 | 765 | 97,920 | 3,072 | 2,286 | 24.23 s |
| residual multi-w | 7 | 0/3 | 768 | 7,864,320 | 9,216 | 3,732 | 40.44 s |
| residual multi-w | 8 | 0/3 | 768 | 7,864,320 | 9,216 | 3,732 | 40.71 s |
| diversity multi-w | 7 | 0/3 | 768 | 7,864,320 | 9,216 | 3,747 | 57.77 s |
| diversity multi-w | 8 | 0/3 | 768 | 7,864,320 | 9,216 | 3,747 | 57.69 s |

Baseline rank-7 trials exhaust the depth/frontier after 247 nodes per seed. Multi-`w` exhausts the 256-node budget. All providers return 12 actions per expanded node in these runs. The experimental providers therefore generate 9,216 candidate edges per three-trial group, versus about 3,000 for baseline.

The baseline rank-8 solutions all pass exact reconstruction and are the conventional decomposition with zero preparation additions and four post-additions. Neither multi-`w` strategy produces a complete solution at rank 7 or rank 8. Added exact-scored alternatives displace baseline actions in the top-12 set, causing search interference rather than a larger effective search horizon.

### Complete-action-evaluation-matched results

The baseline uses its original node budget. Multi-`w` receives three nodes per trial: 3 × 10,240 = 30,720 scored complete actions versus a nominal 256 × 128 = 32,768 baseline actions. Across three trials, each experimental rank/provider group scores 92,160 actions and generates 108 edges. No multi-`w` trial succeeds. The comparison is intentionally severe but makes the 80× proposal cost visible; node-matched results are not equal-compute results.

## Candidate-cost variation

Top-12 candidates on the 10 known Strassen states show meaningful variation for all providers. Multi-`w` adds output-support-three actions and exact-reduction-zero actions:

| Provider | Instances | Preparation additions | Residual reductions | `w` support |
|---|---:|---|---|---|
| `bounded_128` | 106 | 0:12, 1:29, 2:28, 3:37 | 1:14, 2:74, 4:17, 8:1 | 1:83, 2:23 |
| residual multi-w | 120 | 0:12, 1:47, 2:31, 3:30 | 0:5, 1:8, 2:89, 4:17, 8:1 | 1:76, 2:37, 3:7 |
| diversity multi-w | 120 | 0:12, 1:47, 2:29, 3:32 | 0:5, 1:8, 2:89, 4:17, 8:1 | 1:75, 2:39, 3:6 |

Hardware-aware reranking would have candidate variation on these states. That does not establish lower device cost. No GPU benchmark or calibration was run. There are no experimental complete decompositions whose decomposition-wide preparation cost could be compared.

## Failure analysis

The experimental provider meets the primary coverage criterion: it exposes exact Strassen-compatible actions that deterministic `w` selection could not produce. Coverage alone is insufficient for discovery for three reasons:

1. Five missing trajectory actions are blocked by the 128-pair cutoff rather than `w` selection.
2. One newly generated action ranks 19th and never reaches the top-12 search interface.
3. Globally mixing alternatives by residual score removes conventional baseline actions from top 12 and destroys rank-8 reliability.

The dominant remaining limitation is two-stage retention. Pair selection, per-pair alternative selection, and global top-12 selection each have independent coverage effects. Exhaustive local `w` ranking is also computationally disproportionate to the small recall gain.

## Reproduction and artifacts

```sh
PYTHONPATH=src python scripts/run_tensor_game_multi_w_provider.py \
  --output artifacts/tensor-game/multi-w-provider-v1
```

The command refuses to overwrite existing evidence. The compact archive contains:

- `recall.json`: all 70 provider/state recall records and candidate-cost descriptors.
- `trials.json`: all 36 search trials, configurations, traces, solutions, counters, and timings.
- `summary.json`: recall, search, and cost-distribution aggregates.
- `configuration.json`: command, ranks, and seeds.
- `artifact_hashes.json`: sizes and SHA-256 hashes.

## Limitations and next experiment

The state set follows known Strassen trajectories and does not estimate arbitrary-state recall. CPU timings include Python bookkeeping. The full 80-vector enumeration is feasible for this four-coordinate domain but does not scale to larger output dimensions. The diversity heuristic is structural and not learned or hardware-calibrated. Three-node evaluation-matched search is too shallow to test solvability, but it accurately exposes the current proposal-cost imbalance.

The smallest justified next experiment is a quota-preserving merge rather than more alternatives: reserve baseline top-12 slots, add a small number of multi-`w` actions through an equal-size replacement quota, and use a separable top-`k` `w` generator that does not score all 80 vectors. Audit conventional rank-8 retention before another rank-7 search. Broader `(u,v)` coverage should only be added after controlling this search interference.
