# TG-4H residual-guided policy supervision feasibility

## Result

Residual-guided supervision is feasible with the current policy representation
and is cheap enough to generate offline at this scale. It should not be treated
as a globally optimal action oracle. A small controlled mixed-supervision
experiment is justified.

The study selected states before teacher scoring: the first two target IDs from
TG-4B.2A training and TG-4E calibration at each constructed depth from one to
four. This produced 16 root states. The eight reserved TG-4G targets were used
only for an exclusion check. No target or residual collided with them.

The run used commit `4a83b8d6677de426297226f2f72ab5c959bd3554` on branch
`research/tensor-game`. The working tree was dirty and existing changes were
preserved. Measurements used CPU, Python 3.13.11, and PyTorch 2.11.0+cu128.

## Residual teacher

The unchanged TG-4F algorithm enumerates all 80 nonzero ternary `u` vectors and
all 80 nonzero ternary `v` vectors. For each of 6,400 pairs it derives `w` by
quantizing the exact residual contraction divided by
`||u||² ||v||²`. It canonicalizes and deduplicates valid actions, scores each by
the exact squared norm after subtraction, then sorts by score and lexicographic
`u`, `v`, and `w` tie-breaks.

This is exhaustive over `u,v` with a deterministic derived `w`. It is not
exhaustive over all `80³` rank-one triples. Immediate residual scores can tie.
The implementation materializes all unique canonical candidates before sorting.
The observed count ranged from 1 to 601, with mean 188.2. The coefficient and
score payload alone has a 104-byte-per-candidate lower bound, or about 62.5 KiB
at 601 candidates. Actual Python memory is higher due to objects and hash-table
overhead.

Seven timed observations followed a warmup on each of four states per depth:

| Constructed depth | Observations | Median | Range |
|---:|---:|---:|---:|
| 1 | 28 | 39.85 ms | 32.08–65.32 ms |
| 2 | 28 | 43.69 ms | 38.11–53.72 ms |
| 3 | 28 | 55.33 ms | 51.82–63.29 ms |
| 4 | 28 | 74.67 ms | 65.63–103.79 ms |

Across the same states, learned top-12 decoding took a median 17.53 ms frozen
and 17.14 ms trained. The teacher averaged 55.91 ms. Its 6,400 exact contraction
evaluations make it unattractive as a direct production replacement, especially
as shape grows. It is practical as an offline labeler.

## Representation and decoder compatibility

The audit covered every returned teacher action, 181 in total. Some states had
fewer than 12 unique legal candidates.

| Check | Result |
|---|---:|
| Legal ternary rank-one action | 181/181 |
| Exact encode/canonicalize/decode round trip | 181/181 |
| Structurally expressible by factorized policy output | 181/181 |
| Present in frozen checkpoint top-256 | 57/181 |
| Present in trained checkpoint top-256 | 0/181 |

The architecture emits one of `{-1,0,1}` at each of the 12 factor positions.
There is no token mask that structurally excludes these actions. The bounded
beam decoder rejects zero factors and canonical duplicates, neither of which
applies to the audited actions. The actions are representable but poorly ranked,
not unrepresentable. The trained checkpoint's 0/181 bounded recall is a model
ranking result, not proof that the decoder could never emit them after training.

## Existing versus residual supervision

Existing MCTS supports were available for the eight training roots. Their 85
teacher actions overlapped the selected MCTS supports only six times. Thus 79/85
teacher actions were absent rather than merely assigned low target probability.
Mean MCTS target mass on teacher actions was 0.1645 per training state, driven by
a few overlaps. The constructed reference action appeared in teacher top-12 on
6/8 training roots but in MCTS support on only 2/8.

Across all 16 training and calibration roots, teacher top-12 included the
constructed reference action on 12/16 states. By depth the counts were 4/4,
4/4, 3/4, and 1/4. Frozen top-256 included it on 5/16. Trained top-256 included
it on 0/16. These are constructed-reference comparisons, not exhaustive oracle
labels.

The disagreement supports supervision mismatch. It does not establish that every
teacher action is preferable to every MCTS action. MCTS contains multi-step visit
information, while the teacher optimizes one exact local score.

## Multi-step evidence

For each state, the audit took up to the top three teacher actions and continued
greedily with the same teacher within the known construction-length budget.
Fourteen of 46 tested first actions reached exact zero. Teacher top-1 reached
exact zero on 13/16 states:

| Depth | Top-1 continuation success | Top-1 equals construction action |
|---:|---:|---:|
| 1 | 4/4 | 4/4 |
| 2 | 4/4 | 3/4 |
| 3 | 2/4 | 1/4 |
| 4 | 3/4 | 1/4 |

Three counterexamples are preserved in `multistep_analysis.csv`. On those states,
the best immediate residual action left squared residual 3, 2, or 6 after the
remaining greedy steps. One counterexample had a two-way best-score tie. This
shows useful multi-step signal and a clear limit: immediate norm reduction is not
a proof of the best decomposition or globally optimal action.

## Offline generation feasibility

The 16 full prototype records occupy 90,496 bytes, or 5,656 bytes per state.
Using the measured mean without claiming parallel speedup, labeling 1,400 states
would take about 78 seconds and roughly 7.9 MB in this verbose JSONL format.
Labeling all 2,000 recorded TG-4B.2A decisions would project to about 112 seconds
and 11.3 MB. These are linear projections, not measured full-dataset runs.

Exact duplicate residuals can be cached by state digest. States are independent
and can be parallelized. Vectorized or batched contractions could reduce Python
overhead. None of those optimizations was implemented in TG-4H.

## Objective assessment

Pure residual imitation is easy to construct using a softmax over negative score
deltas. Temperature and ties require validation. It risks teaching greedy
failures. A positive-action likelihood term protects individual actions but does
not directly enforce top-k separation. Pairwise or margin-based top-k losses do,
and multiple-positive targets avoid selecting arbitrary equivalents.

The most direct response to the measured regression is mixed MCTS and residual
supervision with a multiple-positive coverage term. Retain MCTS visit
cross-entropy. Add known construction actions and teacher actions with verified
bounded continuations as positives. Compare them against model hard negatives.
Do not choose a mixing weight from this study. Validate exact discovery and
decoded top-k recall rather than relying on conditional cross-entropy alone.

Detailed designs are in
`artifacts/tensor-game/tg4h-residual-supervision-v1/objective_design.md`.

## Answers and limitations

1. Residual proposals succeed because exact state-dependent scoring supplies
   useful actions omitted by learned bounded decoding.
2. The current factorized ternary policy can represent every audited action.
3. On training roots, 79/85 teacher actions were absent from MCTS support.
4. Immediate residual guidance is useful through depth four but failed on 3/16
   top-1 bounded greedy continuations.
5. Teacher labeling averaged 55.91 ms per sampled state and is practical offline.
6. A multiple-positive top-k coverage term most directly targets the regression.
7. A controlled training experiment is justified, with multi-step filtering and
   MCTS retention.

This is a 16-state feasibility sample chosen deterministically, not a performance
estimate over a broad target distribution. Constructed depths are upper bounds,
not proven minimum ranks. Greedy success does not prove global optimality. No
training or checkpoint modification occurred.

## Recommended TG-4I experiment

Run one small preregistered policy-only ablation on TG-4B.2A training states:
existing MCTS cross-entropy versus the same loss plus a multiple-positive
coverage term built from construction actions and teacher actions with verified
bounded continuations. Freeze the architecture, decoder, optimizer budget, and
search evaluation. Select any loss coefficient on validation targets only, then
compare decoded top-12 recall and exact discovery on untouched test targets. Do
not use TG-4G targets for training or selection.
