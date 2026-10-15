# Offline TensorGame learning: held-out discovery case study

## Research question

Can offline policy and value learning from sampled-MCTS experience improve exact
tensor-decomposition discovery on unseen synthetic targets?

TG-3B trained a small policy/value network on synthetic demonstrations. TG-4A
then recorded actual sampled-MCTS decisions. TG-4B.2A expanded that replay buffer
to 200 independent originating targets with train, validation, and test partitions.
TG-4B.2B trained policy-only, value-only, and joint candidates from the frozen
TG-3B initialization. Its offline validation metrics selected checkpoints without
opening the test partition.

Those prediction metrics were not discovery results. In particular, lower policy
cross-entropy on recorded MCTS supports did not establish that a policy could find
an exact decomposition under bounded search. TG-4B.2C tested that question directly.

## Frozen protocol

The held-out experiment used all 30 persisted test target groups and six seeds per
target. Ten configurations therefore produced 1,800 trials. Every search used:

- 200 MCTS simulations per decision;
- at most 12 action proposals per expansion;
- PUCT constant 1.5;
- node limit 200;
- the target's known construction length as the episode move limit; and
- a five-second maximum for the complete multi-decision episode.

The frozen plan SHA-256 was
`addb8863ea5c9e78a3a298db6ca1a2cb4b6d83d81a91226df9883b73c124d0a6`.
The separately reviewed addendum had content digest
`6c55696e316cb1746fd1d736ad5bd0a9f798a238a0874040bd2109406c753615`.
It fixed the comparison matrix, checkpoint path resolution, target-cluster
bootstrap, and whole-episode timeout interpretation before test access.

The curriculum contains `4×4×4` integer tensors generated from one to four known
rank-one terms with coefficients in `{-1, 0, 1}`. Construction length is a verified
upper bound. It is not a proof of tensor rank. The 30 test groups contained seven
length-one, seven length-two, eight length-three, and eight length-four targets.

All five learned candidates started from the frozen TG-3B checkpoint. Its SHA-256
was `857cbf4aa5eec24033bb84180782e2675febec1edb8d4427d1fa9833db4cbbe5`.
Policy-only training changed the policy head while retaining completion-length
value semantics. Value-only candidates kept residual-informed action proposals
and used direct observed-return value heads. Joint candidates changed both policy
and value behavior. Natural and balanced variants differed in their offline
outcome weighting.

## Configurations

| Configuration | Action proposals | Leaf value |
|---|---|---|
| `residual_heuristic_control` | Residual-informed | Nonzero heuristic |
| `residual_constant_max_control` | Residual-informed | Constant optimistic `max_moves` |
| `frozen_policy_heuristic_value` | Frozen TG-3B policy | Nonzero heuristic |
| `policy_only_natural_policy_heuristic_value` | Policy-only natural | Nonzero heuristic |
| `residual_frozen_checkpoint_value` | Residual-informed | Frozen completion-length head |
| `residual_value_only_natural_checkpoint_value` | Residual-informed | Natural direct-return head |
| `residual_value_only_balanced_checkpoint_value` | Residual-informed | Balanced direct-return head |
| `frozen_policy_frozen_value` | Frozen TG-3B policy | Frozen completion-length head |
| `joint_natural_policy_value` | Joint natural policy | Matched direct-return head |
| `joint_balanced_policy_value` | Joint balanced policy | Matched direct-return head |

Completion-length predictions were clamped to `[0, remaining_moves]` and converted
to the MCTS return scale. Direct-return predictions were clipped to the terminal
return bounds. Exact terminal success and move-limit failure overrode both kinds
of estimate. The constant optimistic control tested whether value magnitude alone
could reproduce a learned-value result.

## Exact discovery results

| Configuration | Exact successes | Rate | Distinct targets solved |
|---|---:|---:|---:|
| `frozen_policy_frozen_value` | 24/180 | 13.3% | 4/30 |
| `frozen_policy_heuristic_value` | 24/180 | 13.3% | 4/30 |
| `joint_balanced_policy_value` | 0/180 | 0.0% | 0/30 |
| `joint_natural_policy_value` | 0/180 | 0.0% | 0/30 |
| `policy_only_natural_policy_heuristic_value` | 0/180 | 0.0% | 0/30 |
| `residual_constant_max_control` | 18/180 | 10.0% | 5/30 |
| `residual_frozen_checkpoint_value` | 18/180 | 10.0% | 5/30 |
| `residual_heuristic_control` | 18/180 | 10.0% | 5/30 |
| `residual_value_only_balanced_checkpoint_value` | 18/180 | 10.0% | 5/30 |
| `residual_value_only_natural_checkpoint_value` | 18/180 | 10.0% | 5/30 |

The frozen neural-policy configurations succeeded on 24 of 42 length-one trials.
They solved no longer targets. Each residual-informed configuration succeeded on
12 of 42 length-one trials and six of 42 length-two trials. None solved any of the
48 length-three or 48 length-four trials. Policy-only and joint candidates solved
no target at any construction length.

## Paired comparisons

Pairs used identical target groups and seeds. The uncertainty calculation first
formed a success-rate difference for each originating target, keeping its six
seeds together. A deterministic percentile bootstrap then resampled the 30 target
groups 10,000 times with seed 20261013. The intervals are exploratory because the
experiment contains only 30 independent mathematical targets.

Positive differences favor the candidate.

| Contrast | Candidate minus baseline | 95% interval | Trial wins/losses/ties | Target wins/losses/ties |
|---|---:|---:|---:|---:|
| Frozen vs policy-only natural | -13.3 points | [-26.7, -3.3] | 0/24/156 | 0/4/26 |
| Frozen vs value-only natural | 0.0 points | [0.0, 0.0] | 0/0/180 | 0/0/30 |
| Frozen vs joint natural | -13.3 points | [-26.7, -3.3] | 0/24/156 | 0/4/26 |
| Value natural vs balanced | 0.0 points | [0.0, 0.0] | 0/0/180 | 0/0/30 |
| Joint natural vs balanced | 0.0 points | [0.0, 0.0] | 0/0/180 | 0/0/30 |
| Constant optimistic vs learned natural value | 0.0 points | [0.0, 0.0] | 0/0/180 | 0/0/30 |

Policy-only training had improved offline policy cross-entropy during validation,
but its held-out exact-discovery rate fell from 13.3% to zero when the heuristic
value was held fixed. The offline prediction improvement therefore did not
translate into improved discovery. This result identifies a regression. It does
not identify its mechanism.

Value-only natural, value-only balanced, frozen completion-length, heuristic, and
constant-optimistic guidance produced identical success counts under residual
proposals. More strongly, the learned natural direct-return value and the constant
control tied on every target and seed. These outcomes provide no evidence that the
learned value ranking improved exact discovery.

## Search efficiency

| Configuration | Mean wall | Median wall | Completed simulations | Simulations / episode-wall second | Timeouts |
|---|---:|---:|---:|---:|---:|
| `frozen_policy_frozen_value` | 1.520 s | 1.714 s | 92,400 | 338 | 0 |
| `frozen_policy_heuristic_value` | 2.018 s | 2.319 s | 77,552 | 214 | 35 |
| `joint_balanced_policy_value` | 1.878 s | 1.745 s | 84,224 | 249 | 20 |
| `joint_natural_policy_value` | 1.991 s | 2.184 s | 77,384 | 216 | 36 |
| `policy_only_natural_policy_heuristic_value` | 1.934 s | 1.582 s | 78,298 | 225 | 32 |
| `residual_constant_max_control` | 0.360 s | 0.192 s | 92,400 | 1,425 | 0 |
| `residual_frozen_checkpoint_value` | 0.399 s | 0.221 s | 92,400 | 1,285 | 0 |
| `residual_heuristic_control` | 0.484 s | 0.229 s | 92,400 | 1,062 | 0 |
| `residual_value_only_balanced_checkpoint_value` | 0.501 s | 0.273 s | 92,400 | 1,025 | 0 |
| `residual_value_only_natural_checkpoint_value` | 0.504 s | 0.277 s | 92,400 | 1,018 | 0 |

The complete run recorded 871,858 simulations and 2,085.87 aggregate episode-wall
seconds. Equal simulation limits did not mean equal wall-clock compute. Neural
policy decoding was much slower than residual proposals and caused timeouts in
four configurations. The saved schema did not retain neural inference time or
proposal-generation time separately, so the table cannot assign the difference
to a specific component.

The sampled root-action Q ranges also differed, from approximately `[-4.85, 0.01]`
for the constant control to `[-7.49, 0]` for the natural direct-return value and
`[-13.75, 0]` for policy-only plus heuristic value. These are backed-up root Q
statistics. They are not raw neural predictions. Raw predictions and clipping
counts were not persisted, so the planned value-scale diagnostic is incomplete.

## Correctness and evidence audit

The independent audit reloaded every per-trial JSON record and rebuilt all totals.
It found 1,800 unique trial IDs and all expected configuration, target, and seed
combinations. Status counts were:

- 138 exact successes;
- 1,539 bounded-search failures;
- 123 episode timeouts;
- zero infrastructure failures; and
- zero interruptions.

Independent integer summation of each successful trial's selected rank-one factors
reconstructed its initial target exactly. No failed record contained an exact
selected-action reconstruction. Checkpoint hashes and recorded policy/value
provenance matched the approved matrix. No simulation or proposal budget was
exceeded.

Raw evidence remains local in `artifacts/tensor-game/tg4b2c-heldout-v1`. The
independent analysis remains local in
`artifacts/tensor-game/tg4b2c-heldout-v1-analysis`. Both directories are ignored
by Git and are not implied to be downloadable by repository readers. The tracked
[`tensor_game_offline_learning_results.json`](tensor_game_offline_learning_results.json)
records their expected hashes and concise summary.

## Limitations

- The sample contains only 30 independent targets.
- Construction length is not proven tensor rank.
- Six seeds measure search variability but are not six independent problems.
- Bounded failure does not establish mathematical nonexistence.
- No configuration solved the harder length-three or length-four targets.
- Simulations-to-first-success, raw value predictions, clipping counts, expanded
  nodes, unique states, transpositions, neural inference time, and proposal time
  were not retained in the final trial records.
- Episode time mixes search, inference, proposal generation, Python, and
  environment overhead.
- The evidence is local and ignored rather than distributed with the repository.

## Conclusions

For this workload, checkpoint set, search budget, and held-out sample, offline
MCTS-experience learning did not improve exact discovery. Policy-only and joint
training regressed to zero success. Value-only training left discovery outcomes
unchanged. Balancing did not affect success. Constant optimism reproduced the
learned direct-return value outcomes exactly under residual proposals.

The result does not show that offline search-experience learning cannot work. It
shows that the tested update did not improve the downstream search objective and
that better offline prediction metrics were insufficient evidence of progress.

## Proposed TG-4C investigation

TG-4C should investigate the policy regression before attempting another large
held-out run. It should remain a diagnostic milestone rather than immediate
retraining. Useful checks include proposal overlap and rank changes on fixed
non-test residual fixtures, probability mass assigned to known-valid actions,
action diversity, entropy, sign-equivalence aggregation, and how head-only policy
updates interact with the frozen torso.

Any future replication should be preregistered on a fresh held-out target set. It
should preserve the constant-optimistic control and persist raw leaf predictions,
clipping reasons, simulations to first success, expanded nodes, unique states,
transpositions, inference time, and proposal-generation time.
