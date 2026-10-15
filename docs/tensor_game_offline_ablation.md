# Controlled offline policy/value ablation

TG-4B.2B compares five updates from the same frozen TG-3B checkpoint with an
unchanged baseline. It uses the verified TG-4B.2A replay buffer. This is an
offline supervised experiment over MCTS experience. It is not online self-play
and offline losses do not establish improved decomposition discovery.

## Frozen evidence and leakage controls

The source buffer contains 200 originating targets, 800 episodes, and 2,000
decisions. Its target assignments were fixed before search: 140 train, 30
validation, and 30 held-out test targets. The runner verifies exact replay,
canonical sampled actions, normalized policy targets, return labels, source
digests, and zero target or residual overlap across partitions.

Training constructs only train and validation adapters. Validation selects the
best epoch. It does not construct a test adapter, calculate test metrics, or run
held-out MCTS. The frozen future evaluation plan is written only after candidate
selection and checkpoint hashing.

## Models

The comparison contains the frozen TG-3B model plus:

- `policy_only_natural`
- `value_only_natural`
- `value_only_balanced`
- `joint_natural`
- `joint_balanced`

Every candidate starts from the same state dictionary. Policy-only and value-only
runs strictly freeze the shared torso and the unused head. Joint runs update the
torso and both heads. This strict design isolates head updates but also limits the
capacity of single-head ablations.

Policy learning uses cross-entropy against the normalized MCTS visit distribution
conditional on each recorded sampled action support. Joint action probabilities
and sign-equivalent forms are aggregated in log space. These targets describe
bounded search behavior, not mathematically optimal actions.

Value learning uses uncensored observed episode returns. Updated value heads use
`direct_observed_mcts_return_not_completion_length`. The frozen and policy-only
value heads retain TG-3B completion-length semantics. Their return MAE and RMSE
are reported only as raw cross-semantics diagnostics.

## Outcome balancing

Natural candidates retain observed decision frequencies. Balanced candidates use
deterministic inverse decision-outcome-frequency weights on training examples.
The success-to-truncation weight ratio is capped at 10:1 and weights are normalized
to mean one. Labels are never changed. No episode is duplicated or discarded.
Validation is always unweighted.

Repeated residuals remain explicit correlated observations. Balancing does not
turn them into independent evidence. Reports include raw decision frequencies,
effective weighted frequencies, and unique residual counts.

## Training and model selection

The frozen settings are seed 42, at most 30 epochs, batch size 128, learning rate
0.001, AdamW, gradient clipping at 1.0, and patience three. Each epoch records
policy loss, value loss, total loss, gradient norm, and validation metrics.

Policy-only selection minimizes validation policy cross-entropy. Value-only
selection minimizes validation return MSE. Joint selection minimizes policy
cross-entropy plus return MSE. No held-out result affects epoch selection or
hyperparameters.

The CLI defaults to a dry run:

```sh
PYTHONPATH=src python examples/tensor_game_offline_ablation.py
```

Run one joint-natural smoke epoch in a new directory:

```sh
PYTHONPATH=src python examples/tensor_game_offline_ablation.py \
  --smoke --output artifacts/tensor-game/tg4b2b-offline-smoke-v1
```

Run the frozen five-candidate experiment explicitly:

```sh
PYTHONPATH=src python examples/tensor_game_offline_ablation.py \
  --train --output artifacts/tensor-game/tg4b2b-offline-ablation-v1
```

## Value scale and frozen search design

TG-3E showed that absolute optimism can dominate PUCT. Direct-return checkpoints
therefore require a distinct adapter that preserves exact terminal overrides and
clips nonterminal predictions only to the environment's terminal return bounds.
They must not use the completion-length conversion.

`evaluation_plan.json` preregisters held-out targets, paired seeds, search budgets,
policy/value ablations, the constant optimistic control, exact reconstruction,
failure handling, and target-cluster analysis. TG-4B.2B creates this plan but does
not execute it.

The main limitation is the source imbalance: most episodes truncate, especially
at construction lengths three and four. Balancing can change optimization emphasis
but cannot manufacture successful independent target groups. TG-4B.2C should run
the frozen held-out protocol without retuning from its results.
