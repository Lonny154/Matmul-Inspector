# TG-4L final-ranking-aware policy coverage

## Result

Final-ranking supervision fixed the narrow TG-4K training-set failure but did
not improve validation top-12 recall over the matched prefix control. The
combined model placed 7/9 verified training positives in the actual width-384
top-12, compared with 0/9 for prefix-only. Both models recalled 1/44 validation
positives from 1/30 states.

This was a bounded training and validation study. It did not access held-out
discovery targets, run search discovery, or change the production decoder.

## Diagnosis of the recovered positives

TG-4K's saved prefix checkpoint put all nine training positives in its top-256
decode but none in top-12. Exact width-384 tracing shows that all nine survive
to complete actions. Their observed raw beam ranks are 108 to 244. Their
teacher-forced score margins against the twelfth accepted proposal range from
-6.796 to -1.236 nats, with median -2.480. They fail because other accepted
valid canonical actions rank ahead of them, not because of prefix pruning,
invalidity, or sign canonicalization.

The saved TG-4K set checkpoint has a different profile. Eight of nine positives
are selected in top-12. The remaining positive is pruned before completion.
This reproduces its reported 8/9 recall and prevents the two checkpoints from
being conflated.

## Objective and controls

The state-level ranking hinge is:

`relu(0.1 + score(twelfth accepted action) - score(best verified positive))`

Scores use the production decoder's cumulative token log-softmax semantics and
the best of four sign-equivalent sequences. A fresh width-384 decode regenerates
the hard negative every update. Selection is detached while the chosen scores
remain attached. A pruned positive is teacher-forced, so lower loss does not by
itself establish decoder inclusion.

The state-level maximum protects at least one useful continuation. Per-positive
diagnostics remain separate to expose collapse. The unchanged TG-4K prefix loss
handles survival and shared prefixes.

The preregistration fixed eight training roots, seed 42, frozen initialization,
policy-head-only AdamW at 0.001, 100 updates, margins 0.1, and unit loss weights.
Each configuration uses the same deterministic replay support and MCTS loss:

- `matched_set`: MCTS plus whole-action set coverage.
- `matched_prefix`: MCTS plus prefix survival.
- `matched_combined`: MCTS plus prefix survival plus final ranking.

## Tiny training results

| Configuration | Top-12 | Top-256 | Final MCTS loss | Final prefix loss | Final rank loss | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Matched set | 8/9 | 8/9 | 3.4538 | 0.0530 | 0.0000 diagnostic | 41.98 s |
| Matched prefix | 0/9 | 1/9 | 2.2632 | 0.0991 | 2.9068 diagnostic | 45.33 s |
| Matched combined | 7/9 | 8/9 | 2.8256 | 0.0062 | 0.3795 | 47.62 s |

The combined model raised the median training positive-to-cutoff margin from
-1.858 to +0.118 nats. Eight positives survived width 384 and seven entered
top-12. One survivor remained excluded and one positive was still pruned. The
state-level formulation does not guarantee every positive is protected.

All three runtimes include calculation of every component for comparable metric
traces, even when a component was diagnostic rather than part of that model's
total loss. Combined training was about 5% slower than the matched prefix run in
this small experiment. The Python beam reconstruction remains the main cost.

## Validation-only comparison

| Configuration | Top-1 | Top-12 | Top-32 | Top-256 | State top-12 | MCTS cross-entropy |
|---|---:|---:|---:|---:|---:|---:|
| TG-4K tiny set | 0/44 | 1/44 | 1/44 | 3/44 | 1/30 | 13.2011 |
| TG-4K tiny prefix | 0/44 | 0/44 | 1/44 | 2/44 | 0/30 | 10.4288 |
| Matched set | 0/44 | 0/44 | 1/44 | 5/44 | 0/30 | 8.8912 |
| Matched prefix | 0/44 | 1/44 | 2/44 | 5/44 | 1/30 | 7.0284 |
| Matched combined | 0/44 | 1/44 | 3/44 | 5/44 | 1/30 | 7.2666 |

The combined and prefix models selected the same single validation state at
top-12. Combined gained one state at top-32. At top-256 it gained one state and
lost one, leaving the total unchanged. Its validation MCTS cross-entropy was
0.2382 worse than matched prefix. These paired results provide no validation
evidence for a ranking benefit.

For the combined model, 39/44 validation positives were prefix-pruned, four
survived but ranked outside top-12, and one was selected. The remaining failure
is primarily generalization of prefix survival, not final filtering.

## Answers and TG-4M recommendation

1. The nine TG-4K prefix positives survived width 384 but ranked 1.236 to 6.796
   nats below the actual accepted-action cutoff.
2. Ranking supervision improved training inclusion from 0/9 to 7/9.
3. Combined supervision clearly outperformed prefix-only on the training roots.
4. Validation top-12 did not improve. Both matched models reached 1/44.
5. Combined validation MCTS cross-entropy was 7.2666 versus 7.0284 for prefix.
6. The small added runtime is manageable, but the overall decoder-aware loss is
   still expensive and has no validation benefit.
7. A larger controlled run is not warranted from this evidence.

TG-4M should be a training-only prefix-generalization study. Use a larger but
still bounded training-root sample and cached or vectorized beam membership.
Measure whether prefix survival transfers to untouched validation roots before
adding more ranking weight or opening any held-out discovery set. Do not start
TG-4M as part of this milestone.
