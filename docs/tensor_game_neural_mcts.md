# Neural-guided TensorGame MCTS

TG-3C connects the fixed TG-3B policy/value MLP to sampled PUCT search. It is a
planning experiment. It does not update weights, run self-play, or implement
reinforcement learning.

## Adapters and PUCT

`NeuralActionPolicy` receives only the current exact residual tensor. It runs the
policy head and reuses the bounded complete-action beam decoder from TG-3B. The
adapter rejects zero factors and coefficients outside `{-1, 0, 1}`. It then
canonicalizes sign-equivalent actions and removes duplicates.

For each returned canonical action, its prior is the sum of the independent
coefficient-model probabilities of all four sign-equivalent factorizations. This
aggregation is exact for that action and is computed in log space. The returned
priors are normalized across the bounded decoded set. They are not calibrated
probabilities over the full legal action space. PUCT consumes these normalized
priors without changing TensorGame transitions.

`NeuralValueEvaluator` receives the same residual and predicts a synthetic known
completion length. If the prediction is `L`, the leaf value is
`max_moves - clamp(L, 0, remaining_moves)`. MCTS backup subtracts one for every
already traversed edge. This keeps incurred action costs separate from the
estimated future completion length. Exact zero residuals return `max_moves`, and
an exhausted nonterminal budget returns `-(max_moves + 1)`. Nonfinite and
out-of-range predictions are clamped and counted. These estimates are not tensor
rank certificates.

Policy and value adapters are independent. The experiment names are:

| Configuration | Proposals | Leaf evaluation |
|---|---|---|
| `residual_baseline` | Residual-informed | Nonzero heuristic |
| `neural_policy_only` | Neural | Nonzero heuristic |
| `neural_value_only` | Residual-informed | Neural value |
| `neural_policy_value` | Neural | Neural value |
| `neural_residual_hybrid` | Fixed neural/residual mix | Nonzero heuristic |

The hybrid reserves a configurable share of the returned-action budget for each
source. It canonicalizes and deduplicates across sources, aggregates priors, and
records proposal provenance. Its mixture weights are fixed rather than learned.

## Correctness and leakage boundary

Held-out synthetic problems are selected only from dataset test decomposition
groups. Training and validation residuals are excluded. Stored next actions and
reference factors are never passed to proposal or value adapters. The 2×2 matrix
multiplication tensor is a separate out-of-distribution target. Every claimed
success is checked by exact integer reconstruction.

The current checkpoint has fixed `4×4×4` tensor input. This supports the held-out
synthetic curriculum and 2×2 matrix multiplication. Other rectangular matrix
multiplication tensor shapes require a separately trained compatible model, so
TG-3C does not silently reshape or pad them.

## Running the experiment

The default is a small CPU smoke run on one held-out target:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_neural_mcts.py
```

An explicitly larger, still bounded comparison can use:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_neural_mcts.py \
  --dataset artifacts/tensor-game/tg3b-dataset-2000 \
  --experiment artifacts/tensor-game/tg3b-mlp-256-seed42 \
  --device cpu --simulations 200 --proposals 12 --seeds 3 11 \
  --wall-time-limit 5 --output artifacts/tensor-game/tg3c-study
```

The output directory must not exist. `results.json` stores identities, digests,
limits, seeds, exact results, proposal counts, node statistics, clipping counts,
and policy/value inference timing. `report.md` gives matched comparison tables.
Failed and timed-out trials remain in the evidence. Equal simulation budgets do
not imply equal computational cost.

CPU and CUDA use the same search semantics. The model is loaded once and reused.
Inference runs without gradients. This initial implementation does not batch
nodes or cache values because either change needs dedicated semantic and timing
tests. GPU use should be measured rather than assumed faster for this small MLP.

The next research step is a larger held-out study with enough seeds to estimate
success uncertainty and separate policy, value, and hybrid effects. Self-play or
policy improvement should come only after that controlled ablation.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022),
<https://doi.org/10.1038/s41586-022-05172-4>.
