# TG-4P cost-aware adaptive candidate routing

## Result

The adaptive rule covered 28/30 validation states at a mean proposal cost of 50.78 ms. Frozen-only covered 7/30; residual-only covered 28/30. This is a validation-only candidate study. No search or held-out discovery was run.

## Frozen protocol

The rule was selected on 140 training states before decoding the 30 validation states. It sends a state to residual when `residual_nonzero >= 2` and otherwise to frozen. Runtime routing reads only the residual tensor. The sealed specification SHA-256 is `fa383822cba1675d081b9901511bfebd358402c7249adedeee91686f024dee9b`.

## Cost and coverage

| Strategy | Covered states | Positive recall | Mean ms | Median ms | p95 ms | Residual calls | Dominated by mean |
|---|---:|---:|---:|---:|---:|---:|---|
| frozen_only | 7/30 | 7/44 | 15.73 | 15.42 | 18.59 | 0 | no |
| residual_only | 28/30 | 31/44 | 50.31 | 48.65 | 75.83 | 150 | no |
| fixed_6_6 | 28/30 | 31/44 | 67.22 | 65.94 | 92.12 | 150 | residual_only;adaptive |
| fixed_9_3 | 27/30 | 30/44 | 67.30 | 67.82 | 92.56 | 150 | residual_only;fixed_6_6;adaptive |
| adaptive | 28/30 | 31/44 | 50.78 | 50.50 | 76.17 | 150 | residual_only |

Residual-only cost 3.20× frozen-only in these controlled samples. Each residual call evaluated all 6,400 `(u,v)` pairs. Therefore 6+6 and 9+3 did not reduce residual enumeration and paid for both providers. Mean-cost dominance is descriptive. Timing ranges and raw repetitions are retained because small differences inside run-to-run variability are not firm dominance evidence. Python `tracemalloc` peaks are separate diagnostics and were not included in timed samples.

## Routing signal

On training data, residual nonzero count had correlation -0.08959408065071749 with frozen state coverage and -0.020374429895384558 with useful residual fallback. These are selection-set associations, not calibrated probabilities. The validation comparison is the check against overinterpreting them.

## Answers

1. Residual generation cost 3.20× frozen inference on mean latency.
2. No. The observed training associations were near zero and the selected rule degenerated to residual-only.
3. Yes, from 7 to 28 of 30, but only by calling residual on every state.
4. No. Adaptive mean cost was 50.78 ms versus 50.31 ms and made the same number of residual calls.
5. No. Both fixed allocations paid full residual enumeration plus frozen inference and did not improve residual-only coverage.
6. Frozen-only and residual-only are the useful non-dominated endpoints. Residual achieved higher coverage and better covered-states per proposal second.
7. No new held-out exact-discovery evaluation is justified for this router. It is operationally residual-only, which TG-4G already evaluated.

## Recommended TG-4Q milestone

Implement and validate one bounded residual-ranking approximation against exhaustive residual-only candidate coverage and cost. Keep the same 12-action budget and validation labels. Do not run held-out search until it establishes a real cost reduction without material coverage loss.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_adaptive_routing.py
```
