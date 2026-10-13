# Neural policy and value learning

TG-3B adds the first supervised neural baseline for TensorGame. It asks whether
a model trained on synthetic factorizations can propose useful actions for
unseen residual tensors. It does not use reinforcement learning or neural-guided
MCTS.

PyTorch is optional. Importing Matmul Inspector and generating TG-3A datasets do
not import it.

## Architecture and input

For a 4x4x4 residual, the model flattens 64 exact integers and converts them to
float32. Values are divided by the configurable input scale, initially `4.0`,
which matches the largest construction length in the first curriculum. TG-3A
rejects float32 conversion when an integer lies outside its exact consecutive
integer range.

The shared network is:

```text
64 -> Linear(256) -> GELU -> Linear(256) -> GELU
```

The policy head emits `(batch, 12, 3)` logits. The 12 positions are the four
coefficients of `u`, `v`, and `w`. Classes are ordered `-1, 0, 1`. The initial
baseline treats coefficient positions as conditionally independent:

```text
P(a | S) = product(position p) P(a_p | S).
```

This is a defined factorized joint distribution, not a claim that factor
coefficients are actually independent. It cannot directly represent correlations
such as “each factor vector must be nonzero.” Inference therefore beam-ranks
complete actions, rejects zero factors, canonicalizes sign-equivalent actions,
and removes duplicates.

The value head predicts one scalar: the length of a known synthetic completion.
That label is an upper bound on the minimum remaining terms. It is not tensor
rank and the prediction is not a mathematical certificate.

## Multi-action policy loss

One residual can have several known valid next factors because addition is
order-independent. For known actions `A`, the loss is

```text
-log sum(a in A) P(a | S).
```

Implementation uses coefficient log-softmax values, sums each complete action in
log space, deduplicates full actions, excludes padded rows with the action mask,
and uses `logsumexp` across valid targets. Rows without a target are rejected.

The joint objective is:

```text
policy_loss + value_coefficient * MSE(predicted_completion, known_completion).
```

Probability outside the recorded target set is not necessarily wrong. The data
contains known completions, not an exhaustive classification of every valid or
optimal action.

## Training

Training uses AdamW, fixed Python and PyTorch seeds, deterministic algorithms,
gradient clipping, validation after every epoch, patience-based stopping, and
best-checkpoint selection by validation loss. The test split is evaluated only
after selecting and reloading the best validation checkpoint.

The versioned experiment manifest records:

- model and training configuration;
- dataset identifier and aggregate content digest;
- split counts;
- optimizer and seed;
- per-epoch train and validation losses, gradient norm, duration, and throughput;
- best epoch and checkpoint;
- held-out metrics;
- Python, PyTorch, CUDA runtime, and device metadata.

Only an explicit model state dictionary is saved. Reloading reconstructs the
architecture from JSON and uses `torch.load(..., weights_only=True)`.

## Evaluation

Held-out policy metrics include top-1 match against any recorded next action,
valid nonzero proposal fraction, and average probability mass assigned to known
actions. Exact TensorGame subtraction measures the proposed one-step residual
change. Nonzero reduction is diagnostic only.

Value metrics include MAE, RMSE, error by known completion length, and the
prediction range. These are completion-length metrics, not rank-estimation
accuracy.

The search-oriented diagnostic beam-ranks a bounded proposal set and measures
known-action hits and diversity. The same held-out residuals and proposal budget
are compared with uniform, residual-informed, and hybrid TG-2B policies. Known
factors are used only for scoring. No policy receives them during inference, and
MCTS is not run.

## Commands

CPU smoke training generates a tiny in-memory dataset:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_train.py \
  --smoke --device cpu --epochs 2 --batch-size 8 --hidden-dim 32 \
  --output artifacts/tensor-game/tg3b-smoke
```

Training on a persisted 2,000-decomposition dataset:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_train.py \
  --dataset artifacts/tensor-game/tg3b-dataset-2000 \
  --device cuda --epochs 30 --batch-size 128 --hidden-dim 256 \
  --learning-rate 0.001 --seed 42 \
  --output artifacts/tensor-game/tg3b-baseline
```

Held-out evaluation and proposal comparison:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_evaluate.py \
  --dataset artifacts/tensor-game/tg3b-dataset-2000 \
  --experiment artifacts/tensor-game/tg3b-baseline \
  --proposal-count 16 --output artifacts/tensor-game/tg3b-evaluation.json
```

All destinations refuse silent overwrites.

## Leakage and limitations

TG-3A splits complete originating decompositions and rejects exact target and
cross-split residual collisions. Standard and Strassen trajectories are absent.
Exact deduplication still cannot remove every algebraic equivalence.

Independent coefficient logits are a simple baseline. Beam truncation can miss
joint actions that rank below many invalid combinations. Known-action match is
not proof of optimality, and failure to match is not proof that a proposal is
mathematically useless. Synthetic data may also differ materially from matrix
multiplication residuals.

TG-3C can wrap the trained model in a sampled proposal policy and compare
neural-guided MCTS under matched budgets. TG-3B does not make that integration.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022).
[doi:10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4).
