# TG-4J coverage loss and policy retention investigation

## Result

The alternative objectives did not improve exact discovery over TG-4I set
coverage. Per-positive supervision improved the MCTS-imitation tradeoff, and
selective retention modestly improved several ranks, but neither produced better
top-12 coverage or new held-out successes.

The study ran on commit `4a83b8d6677de426297226f2f72ab5c959bd3554`
and branch `research/tensor-game`. Existing changes and every historical
checkpoint were preserved.

## Diagnostic evidence

The diagnostic used all 44 verified positives from the 30 TG-4I validation
roots. It compared frozen, reproduced MCTS-only, and TG-4I set coverage using
canonical whole-action log probabilities and actual bounded decoding.

| Classification | Positives |
|---|---:|
| Early sequence pruning | 33 |
| Positive-set competition | 5 |
| Poor relative ranking after surviving the beam | 2 |
| Covered at top-12 | 4 |

TG-4I set coverage increased mean positive log score by 3.743 nats versus frozen
and 1.499 versus MCTS-only. It nevertheless placed only 4/44 positives in top-12
and 11/44 in top-256. A higher sequence score was therefore insufficient to
survive prefix pruning and outrank competing complete actions.

Twenty-seven of 44 actions came from multiple-positive states. Five excluded
actions were in states where more than 90% of modeled positive-set mass
concentrated on one positive. The set loss can stop improving a harder positive
after an easier member carries the set probability.

Absent top-256 actions are reported without invented rank 257 values. Every
action remains structurally representable, so the supported mechanisms are
early pruning, relative ranking, and positive competition rather than an action
vocabulary limitation.

## Objectives and protocol

The preregistration hash is
`cb739c798f12f6a1bdfa54ebe244d9fb31429e4dc615624b618e8e83c512292f`.
The new held-out hash is
`a85a936f96bbebca86601711f4615fdc4b436ba1a646172175cbb457bfccae46`.
It was generated and sealed before training.

The individual objective averages canonical sequence negative log probability
over every verified positive. Unlike set-mass loss, each positive contributes
directly. Its weight was fixed at 0.5. The retention run added weight-1 token KL
from frozen to student.

Retention applied only to 31 labeled roots where frozen top-12 contained a
verified positive. Frozen logits were detached and frozen parameters received no
gradients. This protects observed useful behavior without distilling all frozen
predictions.

Both new runs matched the corrected TG-4I setup: frozen TG-3B initialization,
policy head only, AdamW, learning rate 0.001, batch 128, seed 42, and 30 fixed
epochs. Per-positive training took 80.2 seconds. The retention run took 81.3
seconds. Retention added about one second in this run.

## Validation

| Configuration | Top-1 | Top-12 | Top-32 | Top-256 | MCTS cross-entropy |
|---|---:|---:|---:|---:|---:|
| Frozen | 2/44 | 7/44 | 9/44 | 15/44 | 8.2523 historical |
| MCTS-only | 0/44 | 0/44 | 0/44 | 0/44 | 2.6490 |
| TG-4I set coverage | 0/44 | **4/44** | 7/44 | 11/44 | 3.8763 |
| Per-positive | 0/44 | 2/44 | 3/44 | 11/44 | 3.2299 |
| Per-positive + retention | 1/44 | 3/44 | 4/44 | 11/44 | 3.2832 |

Per-positive supervision recovered less top-12 coverage than set loss but
preserved more MCTS imitation. Retention improved top-1, top-12, and top-32 by
one action over per-positive alone. It did not change top-256 recall.

On the four historical TG-4C critical actions, set coverage retained 2/4 in
top-12. Each new model retained 1/4. Retention improved two excluded ranks from
162 to 19 and from 87 to 52 relative to per-positive alone. Those improvements
were real but did not cross the operational top-12 boundary. The fourth action
remained absent from top-256.

## Fresh held-out evaluation

The sealed set contains four constructed-depth-one and four depth-two targets.
Every policy used candidate budget 12, simulation budgets 16 and 64, node limit
2,000, timeout 10 seconds, PUCT 1.5, heuristic value, and seeds 3, 19, and 35.

| Configuration | Targets solved | Trial successes | Reference top-12 | Mean search time |
|---|---:|---:|---:|---:|
| Frozen | 3/8 | 9/24 | 2/8 | 0.1084 s |
| MCTS-only | 0/8 | 0/24 | 0/8 | 0.1083 s |
| TG-4I set coverage | 1/8 | 3/24 | 2/8 | 0.1083 s |
| Per-positive | 1/8 | 3/24 | 1/8 | 0.1085 s |
| Per-positive + retention | 1/8 | 3/24 | 1/8 | 0.1063 s |

Both new models produced the same three successful trials as set coverage. There
were no improvement or regression transitions against set coverage. Per-positive
and retained per-positive solved one depth-one target and no depth-two targets.
Frozen also solved one depth-two target. All unsuccessful trials exhausted their
simulation budgets. Timing differences are small and do not support an
efficiency claim.

Eight unique targets are too few for useful target-level uncertainty intervals.
Repeated seeds are paired trials, not independent target samples.

## Answers

1. TG-4I failed to preserve every positive because set probability could
   concentrate on one action and most excluded actions were pruned by bounded
   prefix search despite improved whole-action scores.
2. Per-positive supervision did not improve top-12 recall. It reached 2/44
   versus 4/44 for set coverage.
3. Retention modestly preserved useful rankings, reaching 3/44 and improving
   historical ranks, but it remained below set coverage and frozen.
4. Neither new objective improved fresh exact discovery. Both matched set
   coverage at 1/8 targets and 3/24 trials.
5. New-model MCTS cross-entropy, 3.2299 and 3.2832, improved over set coverage's
   3.8763 but remained worse than MCTS-only's 2.6490.
6. Training and inference costs were modest, but no discovery benefit justifies
   preferring these objectives as tested.

## Recommended TG-4K milestone

Run one validation-only decoder-alignment study before further held-out training.
Test a prefix-aware loss that protects every verified positive's sign-equivalent
prefixes through the fixed 256-beam stages, and measure whether this predicts
actual top-12 entry. Keep model architecture and production decoding unchanged.
The diagnostic evidence points to early prefix pruning as the dominant remaining
mechanism. Do not start TG-4K as part of this milestone.
