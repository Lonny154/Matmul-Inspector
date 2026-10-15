# TG-4F search benchmark calibration

TG-4F calibrated search on the preserved TG-4E targets. It did not retrain a
model, change production MCTS, regenerate calibration targets, or evaluate learned
policies on the new TG-4G targets.

All 16 reference trajectories replayed through `TensorGame.step` and reached the
production exact-success predicate with zero final residual. Every recorded action
was valid. This rules out target construction, residual update, and exact-success
disagreement as explanations for the TG-4E floor.

Reference-only MCTS solved all depth-one through depth-three trials with four
simulations. Depth four required eight. Search can therefore execute known-valid
paths correctly.

The reference-augmented control retained frozen top-12 proposals and deterministically
replaced the last candidate with the reference action when needed. It used uniform
priors to avoid mixing privileged and model scores.

| Constructed depth | Simulations | Reference-augmented success |
|---:|---:|---:|
| 1 | 4 | 0/12 |
| 1 | 8 | 0/12 |
| 1 | 16 | 12/12 |
| 2 | 16 | 3/12 |
| 2 | 64 | 3/12 |
| 2 | 256 | 12/12 |
| 3 | 256 | 0/12 |
| 4 | 256 | 0/12 |

At depth one, the first failure is explicit: the reference action is present but
has zero visits at four and eight simulations. At 16 simulations MCTS explores it
and succeeds. At depths three and four, reference-only succeeds while the
12-candidate augmented tree exhausts 256 simulations. Branching and budget, not
transition correctness, dominate those failures.

The deterministic residual control exhaustively evaluates 6,400 nonzero `(u,v)`
pairs per state. It derives legal `w` coefficients from exact residual contractions,
deduplicates canonical actions, and sorts by exact squared norm after subtraction.
It is a bounded mathematical diagnostic, not an oracle or production proposal.
It solved all 12 depth-one trials at every tested budget, including four
simulations. This confirms that strong residual coverage eliminates the depth-one
floor.

The frozen regime for TG-4G uses depth one as primary with 16 simulations and
depth two as a stress group with 64 simulations. Both use 12 candidates, PUCT 1.5,
a 2,000-node cap, a ten-second cooperative between-simulation timeout, the
nonzero heuristic, and seeds 3,
19, and 35. Depths three and four are excluded because the augmented control
remained at zero through 256 simulations. Calibration outcomes are not unbiased
generalization estimates.

The new TG-4G manifest contains eight targets, four each at constructed depths one
and two. Generation used seed 20261015 and excluded exact and axis-permuted
equivalents from earlier corpora plus TG-4E. It was not evaluated during TG-4F.
Specification and manifest hashes are stored beside each JSON file.

The dominant learned-policy failure remains reference-action unavailability. A
larger search budget cannot recover actions outside the proposal set. The dominant
augmented-control failure at higher depth is simulation-budget exhaustion after
branching.

## TG-4G recommendation

Run the frozen specification once on the reserved manifest. Compare frozen,
trained, equal-budget hybrid, and residual-norm control. Treat reference-action
recall as a co-primary diagnostic. This can distinguish candidate quality at
depth one while testing limited branching at depth two. Do not tune the frozen
budgets from TG-4G outcomes.

Reproduction:

```sh
PYTHONPATH=src python examples/tensor_game_search_calibration.py \
  --targets artifacts/tensor-game/tg4e-out-of-sample-v1/targets.json \
  --checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \
  --output artifacts/tensor-game/tg4f-search-calibration-v1
```
