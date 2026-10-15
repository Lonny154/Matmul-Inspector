# TensorGame offline policy/value learning

TG-4B.1 turns the validated TG-4A replay buffer into supervised targets for the
existing TG-3B policy/value MLP. It is the learner portion of a future search,
replay, update, and controlled-search loop. It does not perform self-play or
claim an improvement in matrix multiplication discovery.

## Targets and validation

`ExperienceDataset` loads the complete replay directory through the TG-4A exact
validator before exposing any record to PyTorch. Each item contains the residual,
the original sampled root actions, MCTS visit probabilities and counts, an
observed return plus validity mask, and episode, target, and model provenance.
Action supports remain variable length. Collation pads them and supplies a mask.
No target is invented outside the support that MCTS actually evaluated.

The policy target is conditional on that sampled support. For every canonical
action, the learner sums the four sign-equivalent coefficient factorizations in
log space. It then uses `logsumexp` to normalize joint action probability only
over the sampled actions. Cross-entropy is measured against the MCTS visit
distribution. This distribution is search-improved evidence. It is not a
ground-truth optimal policy or a probability over the full action space.

The updated value head predicts the exact observed episode return under the
TG-4A edge-reward convention. Censored or interrupted outcomes do not contribute
to value regression. Successful and fully truncated episodes remain distinct.
The manifest records the return range and episode outcomes. These targets are
not synthetic completion lengths.

TG-3E showed that optimistic value magnitude can dominate PUCT. For that reason,
TG-4B.1 gives the candidate checkpoint a new explicit value interpretation:
`direct_observed_mcts_return_not_completion_length`. The frozen TG-3B checkpoint
retains its completion-length semantics. A future MCTS evaluation must add an
explicit adapter for the new return scale. It must not silently reuse the TG-3B
completion-length conversion.

## Splits and leakage checks

Splits are deterministic and grouped by originating tensor target. All decisions
from an episode stay together. Identical target groups and residual digests may
not cross partitions. A collision is an error rather than a reason to weaken the
split. Repeated residuals within a partition are retained as replay evidence but
are reported separately and must not be treated as independent uncertainty
samples.

The TG-4A buffer has 12 episodes, 42 decisions, 20 unique residuals, and only four
independent targets. Its train, validation, and test results therefore validate
the pipeline only. They are not credible generalization or reinforcement-learning
results.

## Training modes

- `policy_only` updates the shared torso and policy head. The value head is frozen.
- `value_only` updates the shared torso and value head. The policy head is frozen.
- `joint` updates the torso and both heads.

With the first two modes, changing the shared torso can still change the nominally
frozen head's outputs. `--strict-torso-freeze` freezes the torso for strict head
isolation. The experiment manifest lists every trainable parameter and the
effective loss coefficients.

Training uses AdamW, deterministic seeded ordering, gradient clipping, validation
after each epoch, validation-only best-checkpoint selection, and early stopping.
Output directories must not already exist. Checkpoints use safe state-dictionary
loading and record source, replay, configuration, environment, history, and held-
out metrics.

## CPU smoke run

```sh
PYTHONPATH=src python examples/tensor_game_offline_train.py \
  --experience artifacts/tensor-game/tg4a-experience-v1 \
  --source-checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \
  --output artifacts/tensor-game/tg4b1-offline-smoke-v1 \
  --mode joint --device cpu --epochs 2 --batch-size 8
```

Reload and evaluate the candidate without running MCTS:

```sh
PYTHONPATH=src python examples/tensor_game_offline_evaluate.py \
  --experience artifacts/tensor-game/tg4a-experience-v1 \
  --checkpoint artifacts/tensor-game/tg4b1-offline-smoke-v1 \
  --device cpu
```

Evaluation reports sampled-support cross-entropy, KL divergence, top-1 agreement,
policy entropy, and uncensored-return MAE, RMSE, and prediction range beside the
frozen TG-3B baseline. These are replay-buffer metrics only.

TG-4B.2 should collect a larger replay buffer with more independent targets, then
run preregistered matched MCTS comparisons. Policy-only, value-only, joint, and
frozen-baseline checkpoints should use identical search budgets and seeds. Value
adapters must be explicit and scale-controlled.
