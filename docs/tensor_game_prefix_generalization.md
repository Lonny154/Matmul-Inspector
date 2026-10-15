# TG-4M prefix-survival generalization and training coverage

## Result

Broader verified-positive supervision produced a small validation improvement
over TG-4L, but it did not recover the frozen policy's coverage. Actual top-12
recall increased from 1/44 positives on 1/30 validation states to 2/44 on 2/30
states. The change was one paired state gain and no losses. Frozen remained much
stronger at 7/44 positives on 7/30 states.

The broader model improved width-384 survival from 5/44 to 9/44 relative to
TG-4L. Thirty-five positives were still prefix-pruned, seven survived but ranked
outside top-12, and two were selected. This is evidence that training coverage
matters, but the effect is too small to justify a larger training campaign.

No held-out discovery targets were accessed. Production decoding, MCTS, and the
exact-success predicate were unchanged.

## Dataset comparison

The existing verified-label artifact contains 140 training roots with 216
positives and 30 validation roots with 44 positives. The broader preregistered
sample contains 32 training roots, including all eight TG-4L roots, with eight
roots at each constructed depth. It contains 44 positives.

| Measure | All training | Validation |
|---|---:|---:|
| States | 140 | 30 |
| Verified positives | 216 | 44 |
| Mean positives per state | 1.543 | 1.467 |
| States with 1 / 2 / 3 positives | 72 / 60 / 8 | 17 / 12 / 1 |
| Mean residual nonzeros | 19.54 | 20.20 |
| Positive actions in MCTS support | 34/216 | 5/44 |

Token distributions were similar. Training positives contained 35.76% `-1`,
44.71% zero, and 19.52% `+1`; validation contained 36.55%, 42.99%, and 20.45%.
Constructed depths were also closely balanced. There were no repeated action IDs
within either split and no state-digest or target-ID overlap across splits.

These are modest measured differences, not evidence of a strong distribution
shift. Validation has 0.66 more nonzero residual entries on average and 4.4
percentage points less MCTS-support overlap, but action-token frequencies and
positive multiplicity are close.

## Surrogate audit

The same TG-4L combined checkpoint was audited on every training and validation
root. The trace reproduced exact width-384 scoring, width, sign forms, and
lexicographic tie-breaking. Modeled final survival disagreed with observed beam
membership on 0/260 positives.

| TG-4L checkpoint | Training | Validation |
|---|---:|---:|
| Positives | 216 | 44 |
| Complete width-384 survivors | 32 | 5 |
| Final top-12 | 8 | 1 |
| Median minimum prefix margin | -5.212 | -5.796 |
| Median complete-action margin | -7.824 | -7.189 |

The objective is therefore modeling the intended discrete beam boundary. The
main issue is not a scoring mismatch. It is correct surrogate optimization on
sampled roots with weak transfer to other roots. The loss includes every full
beam stage through first failure. Zero-margin or already-safe stages contribute
no hinge penalty, while genuinely at-risk stages remain active. Shared prefixes
and sign-equivalent actions retain the TG-4K behavior.

## Beam-supervision cost

Caching immutable sign-equivalent teacher encodings preserved loss values and
gradients exactly. Five batch-eight measurements were 0.26451 seconds per loss
without the cache and 0.26322 seconds with it, a 1.005x speedup. This is
measurable but negligible. Model-dependent cutoffs were not cached.

The result identifies Python beam construction and exact decoding as the real
cost. Target conversion is not a useful optimization target. Production code
was not changed.

## Preregistered comparison

The broader run retained TG-4L's frozen initialization, policy-head-only AdamW,
learning rate 0.001, batch size 8, seed 42, 100 updates, unit MCTS/prefix/rank
weights, and exact decoder semantics. Four fixed lexicographic batches were
cycled for the same 100-update budget. The run used 110.43 seconds and peaked at
728,280 KiB process RSS, below the sealed 180-second and 2-GiB limits.

| Configuration | Train top-12 | Validation top-12 | Validation states | Top-32 | Top-256 | Validation MCTS CE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 5/44 | **7/44** | **7/30** | 9/44 | 15/44 | 8.2523 |
| TG-4L combined | 7/44 | 1/44 | 1/30 | 3/44 | 5/44 | 7.2666 |
| Broader combined | **10/44** | 2/44 | 2/30 | 4/44 | 8/44 | **6.3043** |

Against TG-4L, broader training gained one validation state with no loss at
top-12 and top-32. At top-256 it gained four states and lost two. MCTS
cross-entropy improved by 0.9623.

Training dynamics limit the interpretation. After the first update the model
still recalled 7/44 validation positives. Recall then fell to 6 at update 10,
four at updates 20--60, two at update 70, three at update 80, and two at the
preregistered final update. No post-hoc checkpoint was selected. Broader labels
reduce the TG-4L regression but do not prevent useful frozen behavior from being
overwritten.

## Remaining failures and decision

The broader model converted four TG-4L prefix failures into complete survivors,
but only one additional state crossed top-12. Most failures remain early beam
pruning. All audited actions are canonical legal ternary rank-one actions, so
there is no demonstrated vocabulary limitation. The frozen model's 7/44 recall
also shows that this architecture can express materially better validation
coverage than either trained model.

Answers:

1. Training and validation differ modestly in residual density and MCTS overlap;
   action-token and depth distributions are close.
2. The prefix surrogate matched observed beam survival on all 260 positives.
3. Teacher-encoding caching preserved semantics but improved runtime only 0.5%.
4. Broader coverage improved validation prefix survival from 5/44 to 9/44.
5. Actual top-12 improved from 1/44 to 2/44, one paired state gain.
6. That gain does not justify 110 seconds of training or a larger campaign while
   frozen remains at 7/44.
7. The architecture remains a plausible basis because frozen coverage is much
   stronger and every label is representable. The current update objective is
   the weaker link demonstrated here.

## Recommended next milestone

Run exactly one TG-4N policy-update interference investigation. Measure which
token logits and frozen top-12 actions are displaced by MCTS, prefix, and rank
gradients on broader batches. Test whether the independent-token factorization
creates conflicting gradients across sign-equivalent and multi-positive labels.
This should be diagnostic, not another loss-weight sweep or held-out discovery
run. Do not start TG-4N as part of this milestone.
