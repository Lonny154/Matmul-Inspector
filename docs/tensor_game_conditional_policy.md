# TG-4O conditional TensorGame policy

## Result

The experimental conditional policy is correct and trainable, but it did not
outperform a matched independent control on validation candidate recall. It
slightly improved training recall, then generalized worse while making candidate
generation about five times slower.

| Model | Train top-12 | Validation top-12 | Validation states | Validation beam survival | MCTS CE |
|---|---:|---:|---:|---:|---:|
| Frozen independent | 5/44 | **7/44** | **7/30** | **16/44** | 8.2523 |
| Matched independent | 16/44 | 2/44 | 2/30 | 8/44 | 6.5703 |
| Conditional | **17/44** | 1/44 | 1/30 | 8/44 | 6.5536 |
| Conditional, adapters disabled after training | 15/44 | 3/44 | 3/30 | 8/44 | 6.5241 |

The primary paired comparison had zero validation state gains, one loss, and 29
ties for conditional versus independent. The conditional prototype therefore
does not support the central hypothesis on this bounded experiment.

This study did not access held-out discovery targets or modify production model,
decoder, MCTS, or exact-success behavior.

## Experimental architecture

The isolated prototype factorizes:

`p(u,v,w | R) = p(u | R) p(v | u,R) p(w | u,v,R)`

It copies the frozen two-layer residual encoder and splits the frozen 12-token
policy head into `u`, `v`, and `w` base heads. A zero-initialized linear adapter
maps a 12-value one-hot encoding of `u` into the `v` logits. A second adapter
maps the 24-value concatenated `u,v` encoding into the `w` logits.

At initialization, every teacher-forced sequence score and sampled-support loss
matches the frozen independent policy. The conditional model has 92,397
parameters versus 91,941 for the control, a difference of 456 parameters
(0.50%). Trainable counts are 9,708 and 9,252. The residual encoder and value
head remain frozen.

The coefficient vocabulary, 12-token action representation, validity rules, and
four sign-equivalent canonical forms are unchanged.

## Conditional decoder

The separate experimental decoder uses cumulative conditional log probability,
beam width 384, lexicographic tie-breaking, validity filtering, canonical sign
deduplication, and at most 12 accepted actions. It does not call the independent
token beam implementation.

Each validation decode expanded the same 8,004 beam candidates as the
independent decoder. Conditional decoding required an average of 467 cached
context evaluations, compared with one independent policy evaluation.

| Model | Mean validation decode | Median | Relative mean |
|---|---:|---:|---:|
| Frozen independent | 13.02 ms | 13.01 ms | 1.02x control |
| Matched independent | 12.72 ms | 12.61 ms | 1.00x |
| Conditional | 64.25 ms | 63.06 ms | 5.05x |

Equal candidate and beam budgets therefore did not imply equal compute.

Targeted tests established tensor dimensions, normalization, initial score and
sampled-loss equivalence, deterministic legal decoding, teacher/decoder score
agreement, sign deduplication, gradient flow through all heads, checkpoint
round trips, and a ten-update one-action overfit reduction.

## Matched protocol

The preregistration used the 32 TG-4M training roots and 30 disjoint validation
roots. Both trainable models started from the same frozen independent behavior,
used the same four fixed batches, MCTS plus verified-positive set loss, AdamW at
0.001, batch size 8, seed 42, 100 updates, gradient clipping, and final-update
selection.

Training took 2.50 seconds for the independent control and 2.57 seconds for the
conditional model inside the interleaved 5.07-second run. The conditional
training overhead was about 2.9%. Peak process RSS was 670,652 KiB, below the
2-GiB limit.

Both models began with MCTS loss 9.272 and coverage loss 20.144. At update 100,
independent losses were 3.895 and 5.803. Conditional losses were slightly lower
at 3.803 and 5.583. Lower matched training losses did not translate to better
validation recall.

The earlier first attempt used unnormalized conditional MCTS mass rather than
the established support-normalized loss. Initial equivalence exposed the error.
That run is preserved locally as nonconforming and excluded. The conforming run
adds a regression test requiring exact initial sampled-support loss agreement.

## Factorization effects

On training roots, conditional decoding gained two positive actions and lost one
relative to the matched independent model. Beam survival improved by two. This
shows that the adapters can learn useful combinations on supervised roots.

On validation roots, conditional and independent models had the same 8/44 beam
survivors. Conditional gained no top-12 action and lost one. Mean conditional
positive score was 0.042 nats lower. Candidate diversity increased from mean
pairwise token distance 2.433 to 2.553, but the extra diversity was not useful
verified-positive coverage.

As a post-training diagnostic, setting the learned adapters to zero yielded
3/44 validation recall, one state better than the independent control and two
better than the active conditional model. It did not change total beam survival.
This ablation was not a separately trained preregistered model. It indicates
that learned conditional terms harmed final selection on two validation states;
it does not establish that disabling adapters is a deployable improvement.

## Answers and decision

1. The architecture can learn verified combinations: training top-12 reached
   17/44 and exceeded the matched control by one.
2. It did not outperform the control on validation: 1/44 versus 2/44.
3. State coverage was also worse, 1/30 versus 2/30.
4. It reduced dilution on two additional training actions but not on validation.
5. Training overhead was 2.9%; decoding overhead was 5.05x with 467 context
   evaluations per state.
6. No validation improvement justifies a larger conditional training run.
7. This prototype is not a credible production replacement as tested.

## Recommended TG-4P milestone

Run exactly one validation-only adaptive proposal-routing study. Use the frozen
independent policy and existing residual-informed provider as immutable candidate
sources, then test whether auditable residual features can select or interleave
them within the same 12-action budget. Do not train another policy architecture
or open held-out discovery targets until routing improves validation coverage at
controlled candidate-generation cost.
