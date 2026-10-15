# TG-4Q bounded residual candidate generation

## Result

`bounded_128` was the best measured coverage-first tradeoff: 28/30 states and 31/44 positives at 29.03 ms mean latency. Exhaustive reached 28/30 and 31/44 at 47.03 ms. The measured speed ratio is 1.62×. No search or held-out discovery was run.

## Exhaustive profile

Across 15 instrumented calls, mean component times were: contraction 21.76 ms, legal `w` selection 8.55 ms, canonicalization/object work 1.18 ms, tensor construction and exact scoring 14.47 ms, and ranking 0.04 ms. Instrumentation overhead means these components should not be compared directly with uninstrumented latency. The implementation is Python-loop based, not vectorized.

## Validation

| Configuration | State coverage | Positive recall | Top-12 overlap | Pairs exactly evaluated | Work fraction | Mean ms | Median ms | p95 ms | Dominated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exhaustive | 28/30 | 31/44 | 1.000 | 6400 | 1.000 | 47.03 | 45.88 | 71.57 | bounded_128;bounded_512;bounded_1024 |
| bounded_128 | 28/30 | 31/44 | 0.989 | 128 | 0.020 | 29.03 | 28.91 | 30.84 | no |
| bounded_512 | 28/30 | 31/44 | 1.000 | 512 | 0.080 | 29.83 | 29.78 | 31.86 | bounded_128 |
| bounded_1024 | 28/30 | 31/44 | 1.000 | 1024 | 0.160 | 30.91 | 30.94 | 33.20 | bounded_128;bounded_512 |

The continuous relaxation is a rigorous lower bound, but fixed-budget prefix retention is heuristic because discrete coefficient quantization can reorder pairs. All returned candidates use exact integer scores. `bounded_128` omitted five exhaustive candidates across three states and lost no verified positive. It also used about 1.94 MB mean peak traced Python memory versus 0.05 MB for exhaustive because it retains all coarse pair records.

## Answers

1. Contractions dominated at 21.76 ms, followed by tensor construction/exact scoring at 14.47 ms and legal `w` selection at 8.55 ms.
2. Yes on these validation states: the bound preserved 98.9% mean top-12 overlap at `B=128`; `B=512` and `B=1024` reproduced the exhaustive candidates exactly. This is empirical, not an exactness guarantee.
3. `B=128` avoided discrete exact processing for 98% of pairs, while still computing every contraction.
4. Yes. `B=128` reduced mean end-to-end latency from 47.03 to 29.03 ms, with separated timing ranges.
5. No verified-positive or state coverage was lost. Five non-positive exhaustive candidates were absent across three states.
6. The best measured cost-coverage tradeoff is `bounded_128`.
7. It is worth evaluating as an experimental search provider. The higher traced-memory footprint and lack of a fixed-budget exactness guarantee argue against production integration before held-out discovery evidence.

## Recommended TG-4R milestone

Run one preregistered held-out exact-discovery comparison of frozen `bounded_128` against the unchanged exhaustive residual provider, with identical search settings and candidate budgets.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_residual.py
```
