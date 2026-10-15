# TG-4I coverage-aware policy training

## Result

Coverage-aware training improved actual decoded recall and recovered some exact
discoveries relative to a faithfully reproduced MCTS-only control. It did not
match the frozen policy and it degraded MCTS imitation cross-entropy. The tested
objective is useful but incomplete.

The experiment used commit `4a83b8d6677de426297226f2f72ab5c959bd3554`
on branch `research/tensor-game`. Training and evaluation ran on CPU with Python
3.13.11 and PyTorch 2.11.0+cu128. Existing working-tree changes were preserved.

## Protocol and correction

The initial preregistration hash was
`cb6ce26d247c76b4b06936a8643d9e02a10b7085ca12771099e88857da63b4f1`.
Before held-out access, a verification check found that it had mistakenly copied
dataclass defaults: batch 8 and learning rate 0.0005. The historical
`policy_only_natural` manifest specifies batch 128 and learning rate 0.001.

The incorrect runs are preserved under `nonconforming_runs/default_config_mismatch/`
and excluded. A preregistration amendment froze the corrected values before the
conforming rerun. Its SHA-256 is
`8a422b15a328eef69829cf7afa5a78ad45bec905e22b6aa320ff086c8933d11f`.
Coverage weights, selection, data, seed, and evaluation rules did not change.
The held-out manifest remained sealed.

The conforming MCTS-only checkpoint has SHA-256
`1c6f830d1c0915c89b0a2ebb00fdabc96ab4070c279ab70c67b27b5dbca70a79`,
an exact match to the original TG-4B.2B policy-only checkpoint.

## Labels and objective

Labels cover 140 training roots and 30 validation roots. They contain 260
canonical verified-positive actions. Every state includes its validated
construction action. Residual-teacher actions were added only when greedy
continuation reached exact zero within the construction-length budget. The new
TG-4I targets and TG-4G targets were excluded.

The policy represents an action as 12 independent ternary coefficients. For a
canonical action, sequence probability is the sum of the four sign-equivalent
factorizations under the globally normalized coefficient probabilities. For a
positive set `P`, the added loss is
`-log(sum(p(a) for a in P))`, computed with nested `logsumexp`. Logits remain
attached to the graph.

This objective rewards total positive-set probability. It does not require every
positive to enter top-12, and it can concentrate on one easy positive. It also
does not optimize the beam decoder or downstream search directly.

All trained configurations used the frozen TG-3B initialization, AdamW, learning
rate 0.001, batch 128, seed 42, strict torso/value-head freeze, natural decision
frequencies, and 30 fixed epochs. Only the coverage weight changed: 0, 0.1, 0.5,
or 1.0.

## Validation selection

The preregistered rule selected the highest verified-positive top-12 recall,
then lower MCTS cross-entropy, then smaller weight.

| Configuration | Positive top-12 | Reference top-12 | MCTS cross-entropy |
|---|---:|---:|---:|
| Frozen | 7/44 | 5/30 | 8.2523 historical baseline |
| MCTS-only | 0/44 | 0/30 | 2.6490 |
| Coverage 0.1 | 2/44 | 2/30 | 2.7376 |
| Coverage 0.5 | 3/44 | 3/30 | 3.3576 |
| Coverage 1.0 | **4/44** | **4/30** | 3.8763 |

Weight 1.0 was selected without inspecting final targets. It improved actual
bounded decoding over MCTS-only, but the tradeoff was worse MCTS imitation.
Frozen retained the strongest validation coverage.

At cutoff 256, selected coverage recalled 11/44 positives versus 0/44 for
MCTS-only and 15/44 frozen. Higher sequence probability and bounded inclusion
remain distinct from top-12 inclusion.

## Sealed held-out evaluation

The new manifest was generated and hashed before labels or training. It contains
four constructed-depth-one and four constructed-depth-two targets. Evaluation
used candidate budget 12, depths 1/2 simulation budgets 16/64, node limit 2,000,
timeout 10 seconds, PUCT 1.5, heuristic value, and seeds 3, 19, and 35.

| Policy | Target successes | Trial successes | Reference top-12 | Mean search time |
|---|---:|---:|---:|---:|
| Frozen | 2/8 | 6/24 | 4/8 | 0.1245 s |
| MCTS-only | 0/8 | 0/24 | 0/8 | 0.1250 s |
| Coverage 1.0 | 1/8 | 3/24 | 1/8 | 0.1256 s |

All successes were at constructed depth one. Frozen solved 2/4 depth-one targets,
coverage solved 1/4, and MCTS-only solved none. All policies solved 0/4
depth-two targets. Against MCTS-only, coverage produced three failure-to-success
trial transitions and 21 shared failures. Against frozen, coverage shared three
successes, regressed on three trials, and shared 18 failures.

The eight unique targets are too few for a useful target-cluster uncertainty
interval. Repeated seeds are paired search trials, not independent mathematical
samples.

## Critical-action regression

On the four historical TG-4C states, frozen retained 4/4 required actions in
top-12 and reproduced MCTS-only retained 0/4. Coverage 1.0 restored 2/4 to
top-12. A third action reached rank 65. The fourth remained outside the first
256. Coverage therefore partially prevented the rare-action regression, but did
not eliminate it.

## Cost and interpretation

Conforming training took 81.7 seconds for MCTS-only and 94.7, 92.0, and 93.6
seconds for weights 0.1, 0.5, and 1.0. The selected coverage run added about
11.9 seconds versus control in this CPU execution. Label generation used 170
root states and the TG-4H exhaustive residual teacher. Search runtimes were
similar within observed variability.

Measured answers:

1. Coverage training improved top-12 validation recall from 0/44 to 4/44.
2. It restored 2/4 historical critical actions to top-12, so protection was partial.
3. It did not preserve MCTS imitation: cross-entropy worsened from 2.6490 to 3.8763.
4. It improved sealed held-out discovery over MCTS-only from 0/8 to 1/8 targets,
   but remained below frozen at 2/8.
5. Supervision was computationally practical. Training overhead was modest here.
6. Weight 1.0 had the strongest preregistered validation recall. None offered a
   uniformly favorable recall/imitation tradeoff.

These results do not establish a universal ranking. Constructed depth is an
upper bound, labels cover roots rather than every visited state, and the
positive-set loss can ignore harder positives after accumulating mass on an
easier one.

## Recommended TG-4J milestone

Run one preregistered loss-mechanism study on validation data only. Compare the
selected positive-set mass loss with a multiple-positive top-k margin loss that
requires every verified positive to outrank decoded hard negatives. Keep the
data, initialization, optimizer budget, decoder, and search fixed. The goal is
to recover broader critical-action coverage without accepting the observed MCTS
cross-entropy degradation. Do not begin that experiment as part of TG-4I.
