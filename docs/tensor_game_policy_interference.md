# TG-4N policy update and action-factorization investigation

## Result

The evidence does not support simple destructive-gradient interference. Fresh
single updates usually increased verified-positive scores and relative cutoff
margins. MCTS and coverage gradients had positive cosine similarity on every
sampled batch. Yet the historical MCTS-only checkpoint excluded all 33 sampled
positives from the width-384 beam despite improving their mean teacher-forced
complete score by 4.622 nats.

The strongest measured mechanism is a mismatch between optimizing additive
token marginals and preserving useful combinations through bounded beam search.
Long-run updates reshape component ranks and competitor prefixes. This can erase
complete-action recall even while absolute target scores and imitation loss
improve.

This was diagnostic only. It used existing training and validation labels, did
not access held-out discovery targets, and did not modify the production model,
decoder, MCTS, or historical checkpoints.

## Architecture audit

The policy is not autoregressive. A two-layer residual-conditioned MLP produces
12 independent three-class logits for the four `u`, four `v`, and four `w`
coefficients. Given the shared residual representation, a sequence score is the
sum of 12 token log-softmax scores. No selected coefficient conditions a later
prediction.

The shared torso can learn residual-dependent marginal correlations, but the
output distribution contains no action-level interaction term. Production
decoding combines these marginals with a width-384 cumulative-score beam for
top-12, rejects invalid complete factors, and canonicalizes the four equivalent
factor-sign forms.

The model has 91,941 parameters, including 9,252 in the policy head. Historical
training froze the torso and value head for these policy-only comparisons.

## Checkpoint comparison

The deterministic audit used the first three target IDs at each constructed
depth: 12 training states and 12 validation states, containing 33 verified
positive actions. Only four positives appeared in recorded MCTS supports.

| Checkpoint | Mean complete score | Width-384 survivors | Top-12 | Change from frozen |
|---|---:|---:|---:|---:|
| Frozen | -17.004 | 11/33 | 4/33 | reference |
| MCTS-only | -12.382 | 0/33 | 0/33 | lost all four frozen hits |
| TG-4I coverage | -9.830 | 13/33 | 6/33 | three gains, one loss |
| TG-4M broader prefix/rank | -12.803 | 13/33 | 4/33 | one gain, one loss |

Absolute complete scores improved under every trained checkpoint. MCTS-only
still moved every positive outside the width-384 beam. Its median component
ranks degraded from frozen `u/v/w` ranks 11/8/9 to 37/24/16. Pruning was
concentrated at positions seven through nine.

Unsupported positives were not disproportionately suppressed in absolute score.
Across the 29 unsupported actions, mean score changes versus frozen were +4.747
nats for MCTS-only, +6.966 for coverage, and +3.960 for broader prefix/rank.
The four supported actions changed by +3.719, +8.686, and +5.952 nats. Medians
were heterogeneous, and four supported actions are too few for a strong subgroup
conclusion.

Construction-provenance actions improved more than teacher-only actions under
all three trained checkpoints. That association is descriptive because action
patterns and support membership are not randomized.

## Controlled single-update experiment

Three independent batches covered 24 balanced training roots and 34 positives.
Each objective started from the exact frozen checkpoint and used a fresh AdamW
optimizer at 0.001. Historical optimizer state was unavailable, so this is a
matched local diagnostic rather than an exact replay.

| Objective | Mean score delta | Scores reduced | Mean cutoff-margin delta | Margin reduced | Top-12 gains/losses |
|---|---:|---:|---:|---:|---:|
| MCTS only | +0.253 | 10/34 | +0.264 | 10/34 | 1 / 0 |
| MCTS + set coverage | +0.517 | 3/34 | +0.521 | 5/34 | 2 / 0 |
| MCTS + prefix/rank | +0.479 | 4/34 | +0.501 | 5/34 | 1 / 0 |

MCTS-only increased the five supported actions by 0.013 nats on average and the
29 unsupported actions by 0.294. Coverage objectives increased both groups more.
No objective produced a top-12 loss in one step.

First-order predicted positive-score directions were positive for all nine
batch/objective observations and agreed with the sign of actual mean changes.
The MCTS versus set-coverage gradient cosines were 0.384, 0.115, and 0.278. MCTS
versus prefix/rank cosines were 0.299, 0.074, and 0.252. These weakly positive
values do not show direct gradient opposition. Per-action regressions still
occur because an aggregate gradient can help some positives and hurt others.

## Factorization audit

Individually strong factors frequently failed to form a selected complete
action:

| Checkpoint | All three component ranks ≤12 | Still outside complete top-12 | Complete top-12 |
|---|---:|---:|---:|
| Frozen | 13/33 | 9/33 | 4/33 |
| MCTS-only | 0/33 | 0/33 | 0/33 |
| TG-4I coverage | 18/33 | 12/33 | 6/33 |
| TG-4M broader prefix/rank | 12/33 | 8/33 | 4/33 |

This is the expected combinatorial behavior of an additive independent-token
score: good marginal ranks do not ensure that one useful `u,v,w` combination
outranks the many cross-combinations of other plausible coefficients. It is
concrete evidence of factorization-related dilution. It is not proof that a
conditional architecture would improve discovery.

The independently highest-ranked complete action was a verified positive on
only one frozen state, three coverage states, and no broader-prefix/rank states
in this sample. MCTS-only had none.

## Synthetic counterfactuals

On 18 validation positives, one segment of TG-4M logits was replaced with frozen
logits without changing model weights. Replacing `u`, `v`, `w`, or all segments
changed mean target score by -0.553, -0.109, -1.695, and -2.384 nats. Positive
score changes occurred for 7, 9, 8, and 5 actions respectively. Final top-12
counts were 1, 1, 2, and 1.

No single segment consistently repairs the trained policy. Frozen `w` logits
helped one additional action cross top-12, but worsened mean score. These are
synthetic interventions, not realizable trained-policy gains, and they do not
identify one universally failing component.

## Interpretation

Answers:

1. Training updates do not systematically reduce absolute positive scores. The
   historical MCTS checkpoint improves mean scores while destroying beam recall.
2. Unsupported actions were not disproportionately harmed in this sample.
3. MCTS and coverage gradients were weakly aligned, not opposed, on all batches.
4. Independent additive factorization creates measured combinatorial dilution:
   8--12 actions per non-MCTS checkpoint had three top-12 components but no
   complete top-12 inclusion.
5. MCTS-only failures concentrate at early full-beam positions seven through
   nine, but counterfactuals do not isolate one factor segment.
6. The best-supported explanation is long-run marginal and competitor-prefix
   reshaping under an independent-token head, not immediate negative gradients
   on verified actions.
7. More unconstrained training of this head is not justified. Frozen remains the
   strongest learned provider and prior loss changes have not preserved it.

## Recommended TG-4O milestone

Run exactly one structured conditional-proposal feasibility study. Add no
production architecture yet. Compare offline scoring of a small conditional
`u -> v -> w` diagnostic head against the frozen independent head on the same
training and validation fixtures, with exact decoder-cost accounting. The goal
is to determine whether conditional factor dependencies reduce combinatorial
dilution before authorizing architecture training or held-out discovery.
