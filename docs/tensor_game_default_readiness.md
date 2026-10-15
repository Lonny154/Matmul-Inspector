# TG-4V default-provider readiness and independent regression study

## Result

The preregistered decision is **GO**. Exhaustive remains the production default. The study ran 144 trials on 24 independently generated targets. Exhaustive succeeded in 63/72 trials and integrated bounded residual succeeded in 63/72. Paired outcomes were {'both_fail': 9, 'both_succeed': 63}.

Mean end-to-end search time was 1.1277 seconds for exhaustive and 0.5223 seconds for bounded. That is a 2.16× speed ratio. Bounded was faster in 71/72 paired trials. Mean bounded Python allocation was 66.1 KiB peak and 13.8 KiB retained.

| Depth | Exhaustive success | Bounded success | Exhaustive mean s | Bounded mean s |
|---:|---:|---:|---:|---:|
| 1 | 18/18 | 18/18 | 0.0403 | 0.0291 |
| 2 | 18/18 | 18/18 | 0.5210 | 0.3515 |
| 3 | 15/18 | 15/18 | 1.6178 | 0.6310 |
| 4 | 12/18 | 12/18 | 2.3316 | 1.0776 |

Mean latency on successful trials was 0.5376 seconds exhaustive and 0.3137 bounded. On failed trials it was 5.2581 and 1.9828 seconds. There was one non-improving pair: tg4v-d3-03-f007161b7c63 seed 19 (+0.012886 seconds). This case had the same discovery outcome for both providers.

## Frozen decision gates

The criteria, target manifest, and evaluation spec were written and hashed before either provider ran. Hard gates cover reference and scoring correctness, exhaustive-only target regressions, timeouts, and integration compatibility. Promotion gates require at least 1.05× mean speedup, bounded faster in 75% of pairs, bounded mean allocation at most 100,000 peak and 30,000 retained bytes, and a zero-regression one-sided 95% upper bound at most 12.5%.

Failed gates: none.

## Corpus and limitations

The manifest contains six targets at each construction depth from one through four. Primary categories are {'dense': 5, 'intermediate': 4, 'repeated_term': 3, 'sign_equivalent': 3, 'sparse': 4, 'tie_heavy': 5}. Exact tensors and all six axis permutations were excluded against the listed historical manifests and within this corpus. This does not establish exclusion under arbitrary basis changes. Construction length is a verified upper bound, not proven tensor rank. “Near-equivalent decomposition” is structural stress labeling; distinct alternative decompositions were not proven.

Repeated seeds are paired repetitions. The independent unit for the regression bound is the 24 target tensors. A zero-event bound is uncertainty rather than proof of noninferiority. Timing is a single CPU-process observation and Python allocation tracking does not measure all native allocations.

Candidate scores agreed at all 1152 matched states where they were available. Candidate ordering or membership differed at 156 matched states, with no asymmetric discovery outcome. Root memory was measured separately from timing for all target/provider pairs. Process RSS deltas are allocator observations, not per-call peaks. Long-running repeated-search memory stability was not measured and is not inferred from those root samples.

## Reproduction

```sh
PYTHONPATH=src python examples/tensor_game_default_readiness.py prepare
PYTHONPATH=src python examples/tensor_game_default_readiness.py evaluate
```

## Recommended TG-4W milestone

Run a separately reviewed default-provider rollout decision using the frozen TG-4V gates plus a larger multi-machine latency corpus, without changing candidate semantics.
