# TG-4R held-out bounded residual search

## Result

On 12 fresh targets, 72 provider trials formed 36 paired comparisons. Exhaustive succeeded in 36/36 trials and bounded `B=128` succeeded in 36/36. Paired outcomes were {'both_succeed': 36}. Mean full-search time was 0.2441 s exhaustive and 0.1991 s bounded, a 1.23× measured ratio. Equal observed success is not proof of general equivalence.

## Frozen protocol and integrity

The manifest `1b439fe6ba0f2579572d6af4f8fb3430a2fd43b30d626bddf9cfecda96b1f666` and evaluation specification `15b5b1e64b5cb0f7b26aaec2eb8825b11dc56634081cebb7f719f7a9f691948a` were frozen before provider execution. Targets used generator seed 20261016 and excluded exact and six-axis-permutation matches against the listed historical sources. This does not exclude every possible algebraic equivalence. Both providers used top-12, depth-one 16 simulations, depth-two 64 simulations, node limit 2,000, timeout 10 s, PUCT 1.5, and seeds 3, 19, 35.

## Discovery

| Depth | Provider | Trial successes | Targets solved |
|---:|---|---:|---:|
| 1 | exhaustive | 18/18 | 6/6 |
| 1 | bounded_128 | 18/18 | 6/6 |
| 2 | exhaustive | 18/18 | 6/6 |
| 2 | bounded_128 | 18/18 | 6/6 |
| all | exhaustive | 36/36 | 12/12 |
| all | bounded_128 | 36/36 | 12/12 |

## Cost

| Provider | Mean search s | Median search s | p95 search s | Mean proposal s | Mean non-proposal s | Proposal calls | Exact pair evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|
| exhaustive | 0.2441 | 0.2281 | 0.4981 | 0.2422 | 0.0019 | 252 | 1612800 |
| bounded_128 | 0.1991 | 0.1849 | 0.3793 | 0.1971 | 0.0020 | 249 | 31872 |

Proposal time improved by 1.23× and full search by 1.23× on mean latency. Bounded was faster in 36/36 paired trials. MCTS time excluding provider work is computed by subtraction from the unchanged search instrumentation. Wall time to first success and expansions to first success remain unavailable; simulations to first success are retained raw.

## Trajectory comparison

The providers shared 246 matched residual/depth proposal states. Candidate sets and order differed at 81 (32.9%) of them, with 21 first matched-state divergences across the 36 trial pairs. Only matched states enter this comparison. Later provider-specific states are not treated as paired.

## Answers

1. Bounded discovery outcomes are reported directly above; equal observed outcomes are limited-sample evidence, not equivalence proof.
2. The paired full-search timing result is 1.23× on mean latency.
3. Mean proposal time fell from 0.2422 to 0.1971 seconds (1.23×).
4. Candidate sets and order differed on 81/246 matched states and slightly changed expansion counts (252 versus 249 proposal calls), but not observed success.
5. Approximation-only lost successes: 0.
6. TG-4Q measured about 2 MB peak traced allocation per bounded call. TG-4R did not measure process-level peak memory, so it can only report that no timeout or node-limit event occurred.
7. Production integration is not authorized by this study. The recommendation below follows the measured success and cost evidence.

## Recommended TG-4S milestone

Integrate `bounded_128` behind an experimental, opt-in search-provider flag and add a larger preregistered regression corpus before considering a default change.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_heldout_bounded_search.py prepare
PYTHONPATH=src .venv/bin/python examples/tensor_game_heldout_bounded_search.py evaluate
```
