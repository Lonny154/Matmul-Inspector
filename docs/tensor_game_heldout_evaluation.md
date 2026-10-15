# TG-4G held-out policy and search evaluation

## Result

The learned candidate providers found no exact solutions on the eight reserved
targets. The existing exhaustive residual-norm control solved all eight. This is
evidence that candidate coverage, rather than the frozen search implementation,
was the limiting factor in this evaluation.

The run used commit `4a83b8d6677de426297226f2f72ab5c959bd3554` on branch
`research/tensor-game`. It ran on CPU with Python 3.13.11 and PyTorch 2.12.1+cpu.
The working tree was not clean. Existing changes were preserved.

## Frozen protocol and integrity

The evaluation specification and target manifest were verified before model
loading or output creation.

| Input | Verified SHA-256 |
|---|---|
| Evaluation specification | `4716b2531a9c8102778e49589891417df76335024717fbb95248525ba8705573` |
| Held-out target manifest | `dc5646879d626346ba0db08b13c0c375c0e0e52d5b2eec16249b02466aa66eb3` |
| Frozen checkpoint | `857cbf4aa5eec24033bb84180782e2675febec1edb8d4427d1fa9833db4cbbe5` |
| Trained checkpoint | `1c6f830d1c0915c89b0a2ebb00fdabc96ab4070c279ab70c67b27b5dbca70a79` |

The manifest contained four constructed-depth-one and four
constructed-depth-two targets. Constructed depth is an upper bound from the
known construction. It is not a proof of minimum rank. Each provider used seeds
3, 19, and 35. Depth-one trials used 16 simulations. Depth-two trials used 64.
All used a 12-candidate limit, 2,000-node limit, and 10-second timeout. All 96
expected target/provider/seed trials completed in the single preregistered run.

## Exact discovery

Unique-target and repeated-seed results are shown separately.

| Constructed depth | Provider | Trials | Trial successes | Targets solved |
|---:|---|---:|---:|---:|
| 1 | Frozen top-12 | 12 | 0 | 0/4 |
| 1 | Trained top-12 | 12 | 0 | 0/4 |
| 1 | Alternating hybrid top-12 | 12 | 0 | 0/4 |
| 1 | Residual-norm control | 12 | 12 | 4/4 |
| 2 | Frozen top-12 | 12 | 0 | 0/4 |
| 2 | Trained top-12 | 12 | 0 | 0/4 |
| 2 | Alternating hybrid top-12 | 12 | 0 | 0/4 |
| 2 | Residual-norm control | 12 | 12 | 4/4 |

Frozen versus trained and frozen versus hybrid each produced 24 shared failures.
Residual norm produced 24 failure-to-success transitions against both frozen and
trained. There were no regressions or learned-only successes. Repeated seeds are
paired trials, not independent target samples.

## Reference-action recall

Recall uses the single constructed reference trajectory for each target. It is
therefore **reference-action recall**, not exhaustive oracle recall. The
denominator is 12 unique reference states: four depth-one states and eight
depth-two states. Search seeds are not repeated in this denominator.

| Provider | Top-1 | Top-12 |
|---|---:|---:|
| Frozen top-12 | 1/12 | 1/12 |
| Trained top-12 | 0/12 | 0/12 |
| Alternating hybrid top-12 | 1/12 | 1/12 |
| Residual-norm control | 12/12 | 12/12 |

Training did not improve recall. The hybrid added no recalled reference actions
beyond frozen. No hybrid trial succeeded, so there is no successful-path source
attribution to report. The raw attribution table is retained with its schema and
zero observations.

## Search cost

Values below are means per trial. Proposal time is included in search wall time.

| Depth | Provider | Simulations | Expansions | Candidate evaluations | Model forwards | Proposal time (s) | Search time (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Frozen | 16 | 1 | 12.00 | 1 | 0.0158 | 0.0166 |
| 1 | Trained | 16 | 1 | 12.00 | 1 | 0.0155 | 0.0162 |
| 1 | Hybrid | 16 | 1 | 12.00 | 2 | 0.0299 | 0.0306 |
| 1 | Residual norm | 16 | 1 | 8.00 | 0 | 0.0394 | 0.0399 |
| 2 | Frozen | 64 | 13 | 156.00 | 13 | 0.2063 | 0.2108 |
| 2 | Trained | 64 | 13 | 156.00 | 13 | 0.2042 | 0.2088 |
| 2 | Hybrid | 64 | 13 | 156.00 | 26 | 0.4031 | 0.4077 |
| 2 | Residual norm | 64 | 13 | 145.75 | 0 | 0.4829 | 0.4858 |

Residual norm reached its first solution at simulation 2 for every depth-one
trial and simulation 14 for every depth-two trial. The production result does
not expose expansions at first success, so that field is recorded as unavailable
rather than inferred. Depth-one residual search time ranged from 0.0313 to
0.0504 seconds. Depth-two ranged from 0.4389 to 0.5974 seconds. Small timing
differences inside those ranges should not be interpreted.

The residual provider is the unchanged TG-4F diagnostic control. It exhaustively
scores the legal action pool by next-residual norm before returning at most 12
actions. Its zero model-forward count does not make it free. Candidate scoring
dominates its measured runtime, and its wall time exceeds either learned model.

## Failure evidence and interpretation

All 72 learned-provider failures exhausted their simulation budgets. In every
case, the first action on the constructed reference path was absent from the
root candidate set. That classification is direct execution evidence. It does
not prove that every possible solution was absent, but it supports candidate
coverage as the observed failure mechanism. There were no timeouts, node-limit
terminations, invalid transitions, or success-predicate disagreements.

Measured answers to the preregistered questions are:

1. Trained did not beat frozen. Both solved 0/8 targets and 0/24 trials.
2. Trained did not improve reference-action recall. It scored 0/12 versus 1/12.
3. Hybrid recovered no discoveries and no additional reference actions.
4. Residual-informed proposals outperformed learned proposals: 8/8 targets and
   24/24 trials versus zero for each learned provider.
5. Hybrid roughly doubled learned-model inference and wall cost. Exhaustive
   residual scoring cost more wall time despite using no model forwards.
6. Missing constructed reference actions was the dominant observed learned
   failure mechanism.
7. The current training objective showed no measurable end-to-end benefit under
   this frozen regime.

These eight targets are enough for this preregistered comparison but too few for
a universal provider ranking. The residual result removes the TG-4E all-provider
floor within this target set. It does not establish that residual norm will solve
other target distributions or larger ranks.

Raw trial records and aggregate tables are in
`artifacts/tensor-game/tg4g-heldout-evaluation-v1/`.

## Recommended TG-4H milestone

Run one preregistered candidate-generation study that tests whether a bounded,
CPU-compatible approximation to residual-norm ranking can retain its held-out
coverage while reducing exhaustive proposal cost. Keep policy weights, search,
and the TG-4G evidence frozen. Do not start that study as part of TG-4G.
