# TG-4U streaming heap integration

## Result

The opt-in `bounded_128` policy now uses the streaming heap. Exhaustive remains default. All 51 permanent differential fixtures matched the full-list oracle exactly. The frozen TG-4S replay produced {'both_succeed': 33, 'both_fail': 15}, zero candidate divergences, and identical simulations, expansions, and proposal calls.

## Integration evidence

The public provider identifier and CLI flag are unchanged. `bounded_residual_candidates` remains the internal reference oracle. The integrated adapter calls `bounded_residual_candidates_memory_efficient`, retaining 128 rather than 6,400 coarse records. No dependency, MCTS, transition, scoring, or success-predicate change was made.

## Smoke measurements

| Metric | Reference | Integrated |
|---|---:|---:|
| Mean candidate latency | 33.07 ms | 32.23 ms |
| p95 candidate latency | 35.20 ms | 33.71 ms |
| Mean traced peak | 1.92 MB | 0.07 MB |
| Mean retained | 0.112 MB | 0.014 MB |
| Mean search latency | 0.9481 s | 0.9134 s |

Peak traced memory improved 29.0×. Candidate timing ratio was 1.03× and same-session search ratio was 1.04×. This is a smoke regression, not a new performance claim.

## Answers

1. Yes. The opt-in provider adapter calls the streaming heap implementation.
2. Yes. Factory and CLI defaults remain exhaustive.
3. Yes, across all 51 permanent fixtures.
4. Yes. All 48 paired search trajectories and discovery outcomes were preserved.
5. Yes. Mean traced peak improved 29.0× in the integration smoke test.
6. Yes. Candidate latency remained acceptable at 32.23 ms mean.
7. Yes. Permanent fixtures now include dense, repeated-term, zero, degenerate, tie-heavy, near-tie, TG-4Q, and TG-4S states.
8. A default decision still needs broader independent evidence, especially dense and repeated-term discovery workloads rather than candidate-only fixtures.

## Recommended TG-4V milestone

Run one preregistered default-provider decision study with dense, repeated-term, and larger independent target coverage. Do not change the exhaustive default automatically.
