# TG-4T memory-efficient bounded residual ranking

## Result

The fixed-size heap preserved exact selected pairs and final candidates on all 49 differential states. Mean traced peak memory changed from 1.94 to 0.07 MB (29.13× reduction). Mean untraced candidate latency changed from 30.14 to 29.40 ms (1.03× ratio). The CLI remains on the original opt-in implementation and exhaustive remains default.

## Cause and design

`allocation_profile.json` attributes the dominant original allocation to the 6,400 retained `PairScore` records and contraction tuples. The heap retains 128 records and restores the original stable ordering after streaming. Integer arithmetic and exact rescoring are unchanged.

## Candidate benchmark

| Implementation | Mean ms | Median ms | p95 ms | Mean traced peak | Mean retained |
|---|---:|---:|---:|---:|---:|
| original | 30.14 | 29.77 | 32.45 | 1.94 MB | 0.104 MB |
| optimized | 29.40 | 29.23 | 31.18 | 0.07 MB | 0.014 MB |

`tracemalloc` measurements are separate from the uninstrumented timing samples. RSS deltas are allocator-level observations and not reliable per-call peaks.

## Frozen TG-4S replay

Original and optimized bounded providers produced {'both_succeed': 33, 'both_fail': 15} across 48 paired trials. Candidate divergences at matched states: 0. All deterministic trajectory fields matched: True. Mean search time was 0.9166 versus 0.8830 seconds (1.04× ratio). These are fresh same-session controls, not comparisons against historical timings.

Optimized search was faster in 38/48 paired trials. The mean paired difference
was -33.6 ms. This modest timing gain is smaller than the memory effect and
should not be generalized beyond this capture. Process RSS was unchanged in
29/30 optimized root calls; allocator reuse makes RSS deltas less diagnostic
than the traced allocation measurements.

## Answers

1. Retaining 6,400 Python records and contraction tuples caused the elevated peak.
2. Mean peak fell by 29.13×, from 1.94 to 0.07 MB.
3. Yes. Selected pair order, candidate identities, order, and integer scores matched exactly.
4. Candidate latency changed by a 1.03× original/optimized ratio.
5. Same-session search latency changed by a 1.04× ratio with identical trajectories.
6. Yes. The contained heap logic removes the dominant allocation and slightly improves rather than harms measured latency.
7. Yes, it is ready to replace the implementation behind the existing opt-in `bounded_128` identifier in a separate reviewable change. The current CLI was intentionally not changed in TG-4T.
8. A separate default-provider decision still needs broader dense/repeated-term coverage and explicit review.

## Recommended TG-4U milestone

Review and, if accepted, replace the opt-in `bounded_128` selector with the exactly equivalent heap implementation, then add dense and repeated-term cases to the permanent regression suite. Do not change the exhaustive default in that milestone.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_memory.py prepare
PYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_memory.py benchmark
```
