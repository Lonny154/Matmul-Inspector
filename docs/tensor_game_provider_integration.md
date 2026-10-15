# TG-4S bounded residual provider integration

## Result

Users can select `bounded_128` through the installed `tensor-game-search` command or Python factory. Exhaustive remains the default. On 16 fresh targets and 48 paired target/seed comparisons, exhaustive succeeded in 33/48 trials and bounded succeeded in 33/48. Paired outcomes were {'both_succeed': 33, 'both_fail': 15}. Mean search latency changed from 1.1429 to 0.9217 seconds (1.24×).

## Integration

```sh
matmul-inspector tensor-game-search --target-json TARGETS.json --target-id ID
matmul-inspector tensor-game-search --target-json TARGETS.json --target-id ID --candidate-provider bounded_128
```

Python callers use `create_residual_candidate_provider()` or pass `"bounded_128"`. Omitting the identifier returns exhaustive. Unsupported identifiers raise `ValueError`. Candidate and success semantics are unchanged.

Current implementation note: after TG-4U, `bounded_128` uses a streaming
128-entry heap rather than retaining all 6,400 coarse records. The public API,
provider identifier, and exhaustive default are unchanged.

## Regression evidence

| Provider | Successes | Unique targets solved | Mean search s | Median s | p95 s | Mean proposal s | Calls | Exact pairs | Mean Python peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exhaustive | 33/48 | 11/16 | 1.1429 | 0.7199 | 2.6706 | 1.1373 | 1401 | 8966400 | 0.04 MB |
| bounded_128 | 33/48 | 11/16 | 0.9217 | 0.5591 | 2.0937 | 0.9158 | 1398 | 178944 | 1.93 MB |

Bounded was faster in 48/48 paired trials. Proposal speed ratio was 1.24×. Candidate sets or order differed at 756/1347 matched states. Python allocation tracking was separate from timed search. Process RSS deltas are noisy allocator observations, not per-call peaks.

Exact scores agreed for every candidate shared at a matched state. Of 1,332
matched states with at least two exhaustive candidates, 1,317 had a zero minimum
adjacent score gap. The frequent exact ties help explain ordering and membership
differences. They do not establish that every omitted action is interchangeable.

## Interpretation

There were 0 unique targets with an exhaustive-only success. If zero, the one-sided exact 95% upper bound on the independent-target regression probability is 17.1%; this is uncertainty, not proof of noninferiority. Depth 3–4 use a preregistered exploratory 64-simulation budget and failures there may reflect search budget rather than candidate impossibility.

Depths one and two solved 24/24 trials per provider. Depth three solved 9/12,
and depth four solved 0/12. The providers agreed on every outcome. The corpus
contained seven sparse and nine intermediate-density targets. The deterministic
generator produced no dense targets or repeated canonical terms, so this run
does not support a separate conclusion for those structures.

## Answers

1. Yes. CLI and Python API selection are documented above.
2. Yes. Exhaustive is still the default and explicit exhaustive matches the old provider.
3. Paired discovery results are {'both_succeed': 33, 'both_fail': 15}; repeated seeds are not treated as independent targets.
4. Mean end-to-end speedup was 1.24× and bounded was faster in 48/48 pairs.
5. No bounded-specific weak structure was observed. Both providers shared the depth-three and depth-four failures. Dense and repeated-term cases remain a corpus limitation.
6. Bounded Python allocation remains materially higher (1.93 versus 0.04 MB), but practical effects are limited to the recorded RSS and timeout evidence.
7. Bounded should remain opt-in. A default change requires a separate decision and memory optimization.

## Recommended TG-4T milestone

Optimize bounded-provider coarse-record memory without changing candidate results, then rerun the frozen TG-4S regression corpus before any default-provider decision.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_provider_regression.py prepare
PYTHONPATH=src .venv/bin/python examples/tensor_game_provider_regression.py evaluate
```
