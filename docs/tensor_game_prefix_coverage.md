# TG-4K prefix-aware coverage investigation

## Result

The prefix-survival loss correctly models the fixed decoder's cumulative beam
scores and can be overfit to zero. It did not improve actual top-12 recall. It
instead exposed a second bottleneck: surviving the width-384 beam is necessary
but does not ensure entry into the final 12 valid canonical actions.

This was a validation-only study. It did not generate or inspect a new held-out
set, run discovery search, or change the production decoder.

## Exact decoder semantics

The production decoder processes 12 ternary tokens in `u`, `v`, then `w` order.
Each prefix score is the sum of token log-softmax probabilities. At every stage,
candidates sort by descending cumulative score and then lexicographic prefix.

Beam width is `max(256, 32 × requested_actions)`. Top-12 decoding therefore uses
width 384. Top-256 uses width 8,192. Complete actions with zero `u`, `v`, or `w`
are rejected, then sign-equivalent actions are canonicalized and deduplicated.
The first 12 accepted canonical actions become search proposals.

This corrects the simplified fixed-width-256 diagnostic used in TG-4J. Those
counts motivated the hypothesis but were not exact traces of top-12 or top-256
production decoding. TG-4K records width-384 top-12 stages directly and checks
final top-256 separately.

## Prefix tracing

The trace covers 44 validation positives across frozen, MCTS-only, TG-4I set
coverage, TG-4J per-positive, and retained per-positive checkpoints. The first
five positions contain at most 243 prefixes, below beam capacity. No positive
was first pruned before position seven in this sample.

| First outcome | Frozen | MCTS-only | Set coverage | Per-positive | Retained per-positive |
|---|---:|---:|---:|---:|---:|
| Pruned at 7 | 10 | 14 | 3 | 2 | 2 |
| Pruned at 8 | 8 | 16 | 5 | 7 | 8 |
| Pruned at 9 | 5 | 9 | 10 | 14 | 13 |
| Pruned at 10 | 1 | 5 | 11 | 5 | 6 |
| Pruned at 11 | 3 | 0 | 2 | 3 | 2 |
| Pruned at 12 | 1 | 0 | 1 | 2 | 2 |
| Survived complete beam | 16 | 0 | 12 | 11 | 11 |

Median margin at the first failed stage was -0.931 nats frozen, -0.419
MCTS-only, -0.502 set coverage, -0.518 per-positive, and -0.553 with retention.
Some failures were close to the boundary: the largest failed margins ranged from
-0.003 to -0.045 nats across trained policies. Others were many nats below it.
A single uniform margin therefore covers heterogeneous failures.

Complete beam survival and final top-12 membership differ. Set coverage produced
12 complete survivors but only four top-12 actions. Invalid complete sequences,
canonical duplicates, and higher-ranked valid competitors affect the final
accepted list.

## Experimental loss

For each unique sign-equivalent positive prefix at every full beam stage through
its first failure, the loss applies:

`max(0, 0.1 + cutoff_score - best_positive_prefix_score)`

The cutoff is the last score selected into the exact width-384 beam. Candidate
membership and lexicographic tie selection use detached values. The chosen
positive and cutoff score tensors remain attached. The loss is therefore
piecewise differentiable while membership changes remain discrete.

Shared prefixes are deduplicated within each state and stage. Multiple positives
are supported. Empty sets and negative margins are rejected. Targeted tests
verified finite values, gradient flow, tie-compatible tracing, shared-prefix
handling, and agreement between final beam survival and production decoding.
The loss is a surrogate. It does not directly optimize final top-12 recall.

## Tiny training diagnostic

The experiment used the first eight training target IDs, chosen before outcomes.
Both models started from frozen TG-3B, trained only the policy head with AdamW at
0.001, seed 42, and ran 100 matched updates. One optimized TG-4I set coverage.
The other optimized prefix survival. There were 9 unique verified positives.

| Tiny objective | Initial loss | Final loss | Train top-12 | Train top-256 | Runtime |
|---|---:|---:|---:|---:|---:|
| Whole-action set | 20.1439 | 0.7397 | 8/9 | 8/9 | 0.37 s |
| Prefix survival | 0.2474 | 0.0000 | 0/9 | 9/9 | 37.71 s |

Prefix loss achieved its intended boundary condition and slightly exceeded set
loss at broad top-256 coverage. It failed at the operational top-12 cutoff. Once
prefixes safely survived, the hinge became zero and supplied no pressure on
their final canonical ranking.

The unoptimized Python beam reconstruction made prefix training about 101 times
slower in this tiny run. This is practical for diagnostics but not yet for a
full experiment. Vectorization or cached detached beam membership could reduce
the overhead, but neither was implemented.

## Validation-only results

| Configuration | Top-1 | Top-12 | Top-32 | Top-256 | MCTS cross-entropy |
|---|---:|---:|---:|---:|---:|
| Frozen | 2/44 | 7/44 | 9/44 | 15/44 | 8.2523 |
| MCTS-only | 0/44 | 0/44 | 0/44 | 0/44 | 2.6490 |
| TG-4I set coverage | 0/44 | 4/44 | 7/44 | 11/44 | 3.8763 |
| TG-4J per-positive | 0/44 | 2/44 | 3/44 | 11/44 | 3.1642 |
| TG-4J retained | 1/44 | 3/44 | 4/44 | 11/44 | 3.2096 |
| Tiny set overfit | 0/44 | 1/44 | 1/44 | 3/44 | 13.2011 |
| Tiny prefix overfit | 0/44 | 0/44 | 1/44 | 2/44 | 10.4288 |

Tiny-set overfitting is not generalization evidence. Both tiny models degraded
validation MCTS imitation substantially. Prefix training did not improve
validation top-12 or top-256 over the matched set-loss diagnostic.

## Answers

1. Pruning began at token seven and concentrated at positions seven through ten.
2. Median failed margins were roughly 0.42–0.93 nats below cutoff for historical
   models, with both near-boundary and severe failures.
3. The objective reproduces cumulative scoring, width, and tie behavior. Its
   discrete boundary limitation is explicit.
4. It improved its own training prefix-survival criterion to zero and moved 9/9
   training positives into top-256.
5. Better broad survival did not improve top-12: training recall stayed 0/9 and
   validation stayed 0/44.
6. Validation MCTS cross-entropy was 10.4288 after prefix overfit versus 13.2011
   for matched set overfit. Both were poor because the diagnostic optimized only
   coverage on eight roots.
7. The current implementation is too expensive for a full run without
   optimization and, more importantly, lacks top-12 benefit.
8. A full TG-4L training experiment with this loss is not justified.

## Recommended TG-4L milestone

Run a training-only final-ranking surrogate study. Combine prefix survival with
pressure against the score of the twelfth accepted valid canonical action, and
verify on the same eight roots that positives cross actual top-12 rather than
only the wider beam. First optimize or cache beam tracing to reduce its roughly
100× overhead. Do not open a held-out discovery set until both conditions are
met.
