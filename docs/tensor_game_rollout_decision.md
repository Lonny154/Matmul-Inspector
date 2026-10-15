# TG-4W default-provider rollout decision

## Decision

**CONDITIONAL**. Exhaustive remains the default. The bounded provider passed the measured discovery, runtime, correctness, timeout, memory, and installed-package gates on this host. The cross-machine reproducibility gate failed because only one genuine machine was available. A default change is therefore not yet recommended.

## Independent paired evidence

The frozen corpus contains 150 targets: {'1': 28, '2': 42, '3': 40, '4': 40} by constructed depth and {'dense': 25, 'intermediate': 25, 'repeated_term': 25, 'sign_equivalent': 25, 'sparse': 25, 'tie_heavy': 25} by primary category. Both providers ran once per target with seed 3, producing 300 trials and 150 independent target pairs. Paired outcomes were {'both_fail': 16, 'both_succeed': 134}. Exhaustive succeeded in 134/150; bounded succeeded in 134/150.

With zero exhaustive-only targets, the preregistered one-sided exact 95% upper regression-rate bound is 1.98%. This meets the 2% margin but does not prove equivalence.

| Provider | Mean s | Median s | p95 s | Mean proposal s | Mean non-proposal s | Timeouts |
|---|---:|---:|---:|---:|---:|---:|
| exhaustive | 1.0933 | 0.5451 | 4.6306 | 1.0891 | 0.0042 | 0 |
| bounded | 0.5148 | 0.3864 | 1.9074 | 0.5107 | 0.0042 | 0 |

Mean speedup was 2.12×. Bounded was faster in 149/150 pairs (99.3%). Raw per-depth, category, target, and seed results are in `summary.json`, `results.csv`, and `paired_outcomes.csv`.

## Memory, correctness, and compatibility

Each provider completed 500 repeated root proposals under `tracemalloc`, with checkpoint RSS measurements. Final rows are [{'call_index': 'final', 'candidate_count': 12, 'deterministic': True, 'provider': 'exhaustive', 'rss_bytes': 27283456, 'rss_growth_bytes': 155648, 'traced_current_bytes': 192886, 'traced_peak_bytes': 345103, 'traced_retained_growth_bytes': 192886}, {'call_index': 'final', 'candidate_count': 12, 'deterministic': True, 'provider': 'bounded_128_integrated', 'rss_bytes': 27357184, 'rss_growth_bytes': 73728, 'traced_current_bytes': 127758, 'traced_peak_bytes': 187523, 'traced_retained_growth_bytes': 127758}]. Python-traced growth and RSS stayed inside the frozen thresholds, and candidate sequences were deterministic. RSS remains an allocator-level observation rather than per-call ownership. Timed search did not run under `tracemalloc`.

Shared candidate scores agreed throughout the paired trace audit. Exact construction references replayed before evaluation. The installed API audit preserved the exhaustive default, verified explicit exhaustive and bounded selection, confirmed the streaming bounded class, retained exhaustive as fallback, and observed a clear `ValueError` for an invalid identifier. No dependency or interface was added.

## Environment and limitations

This was one process on `AMD Ryzen 7 7700X 8-Core Processor` under `Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39`, Python 3.13.11. No second machine was available. Repeated runs on this host are not presented as multi-machine reproduction. Construction length is an upper bound rather than proven rank. Duplicate exclusion covers exact tensors and six axis permutations, not arbitrary basis transformations. Distinct alternative decompositions were not proven.

The unchanged search result instrumentation records exact-success status and residual norm classification, but not the full final residual tensor. Full final residuals are therefore unavailable rather than reconstructed after the fact. Candidate correctness was audited at root states during resumed finalization because the first finalization attempt failed while serializing completed memory checkpoints. The 300 preserved search trials were not rerun.

## Reproduction

```sh
PYTHONPATH=src python examples/tensor_game_rollout_decision.py prepare
PYTHONPATH=src python examples/tensor_game_rollout_decision.py evaluate
```

## Recommended TG-4X milestone

Repeat the frozen TG-4W paired timing and compatibility protocol on one genuinely independent machine, then combine only the reliability counts and keep timing stratified by environment for the final default-change decision.
