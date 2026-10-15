# TensorGame search experience and replay

TG-4A records actual TensorGame decisions for future policy/value learning. It
does not train a model, update weights, or implement self-play optimization.

At each environment state, a fresh sampled MCTS search produces statistics over
its finite sampled root support. The search target is
`visits^(1/temperature)`, normalized over those root actions. At temperature zero
the highest visit count wins, followed by prior and stable insertion order. If
all visits are zero at positive temperature, the original normalized priors are
used explicitly. This distribution is not an optimal policy over the complete
rank-one action universe.

The behavior policy is recorded separately. It may use another temperature to
select the action actually applied through `TensorGame.step`. Internal MCTS
branches are never labeled as played decisions. An episode contains only these
actual state transitions.

Decision records preserve the exact residual, its digest, depth, remaining move
budget, canonical sampled actions, original priors, visits, Q values, the
search-improved distribution, selected action, behavior probability, search
budget, seeds, termination information, and model provenance. Episode records
link decisions in order and preserve the target, shape, coefficient set, final
residual, outcome, action reward, and exact reconstruction result.

For successful or move-limit-truncated episodes, return targets use the existing
MCTS convention: terminal success has value `max_moves`, truncated failure has
value `-(max_moves+1)`, and each future edge subtracts one. Interrupted episodes
are censored and receive no return target. These observed returns are distinct
from known completion length, a learned value estimate, and minimum tensor rank.

Replay reconstructs the exact initial tensor and applies recorded actions. It
checks every state digest, transition, reward, final residual, policy
normalization, canonical action uniqueness, references, and successful exact
decomposition. This guarantees environment replay. It does not guarantee full
MCTS re-execution across code, RNG, or model-version changes.

`ReplayBuffer` rejects duplicate episode and experience identifiers and refuses
to save invalid records. Optional episode capacity evicts the oldest complete
episode and its decisions. Buffers do not silently combine incompatible reward
conventions; shape and coefficient compatibility remain explicit per episode.

The default example collects held-out synthetic targets and 2×2 matrix
multiplication using residual proposals with heuristic, optimistic constant, and
frozen trained neural values:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_experience_collection.py
```

TG-4B can consume verified decisions as sampled-support policy targets and
observed return targets. It must retain censored outcomes, provenance, repeated
state grouping, and the distinction between this small system and AlphaTensor's
full reinforcement-learning pipeline.
