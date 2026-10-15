# TG-4C policy regression investigation

TG-4C reproduced the held-out policy regression from the frozen TG-4B.2C trial
records and compared the frozen and policy-only checkpoints on identical states.
It did not retrain a model or change the search implementation.

The frozen policy solved 24 of 180 paired trials. The policy-only checkpoint
solved none. The transition table contained 24 regressions, 156 persistent
failures, and no retained successes or improvements. All 24 regressions were six
seeded repeats of four construction-length-one targets.

Every regressed pair diverged at its initial state. On each of the four distinct
states, the frozen policy's 12-action proposal set contained the exact one-step
action. The trained proposal set was disjoint and omitted it. The exact action
ranked 2, 2, 3, or 5 under the frozen decoder. It was outside a bounded
256-action audit under the trained checkpoint. Its same-state canonical log
probability decreased by 5.64 to 8.56 nats. With one move available, none of
these failures was recoverable. No regression timed out, looped, or entered an
invalid state.

Validation cross-entropy improved from 8.252298 to 2.649050 while top-1 agreement
fell from 27.7% to 11.3%. Cross-entropy is conditional on each recorded MCTS
support. It does not optimize complete discovery or protect a critical action
from hard top-12 truncation. Natural policy training also contained 37 decisions
from successful episodes and 1,363 from truncated episodes. Together, the
evidence most strongly supports rare critical-action degradation plus objective
mismatch. Exposure bias is not needed to explain these failures because
divergence happened before either policy changed the state. Conditional top-12
entropy rose slightly on all four states, so increased confidence is not the
supported mechanism.

The full local report, per-trial CSV, trajectories, supports, checkpoint hashes,
and JSON records are generated under
`artifacts/tensor-game/tg4c-policy-regression-v1`. The artifact directory is
ignored by Git.

Run the investigation with CPU PyTorch installed:

```sh
PYTHONPATH=src python examples/tensor_game_policy_regression.py \
  --trials artifacts/tensor-game/tg4b2c-heldout-v1/trials \
  --validation artifacts/tensor-game/tg4b2b-offline-ablation-v1/validation_comparison.json \
  --baseline-checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \
  --trained-checkpoint artifacts/tensor-game/tg4b2b-offline-ablation-v1/candidates/policy_only_natural \
  --output artifacts/tensor-game/tg4c-policy-regression-v1
```

The command refuses to overwrite an existing output directory. Choose a new
path when reproducing it.

The recommended follow-up is a preregistered proposal-limit sweep on validation
states and a fresh held-out target set. Keep checkpoints, MCTS, value guidance,
targets, and seeds fixed. Persist exact-action rank, inclusion, mass, and final
discovery. Do not tune the proposal limit on the existing held-out outcomes.

Limitations remain. The experiment has 30 independent mathematical targets and
six repeated search seeds. All baseline successes were one-step targets. It
therefore cannot diagnose later-step exposure bias. The trained rank is bounded
at 256 rather than exhaustive over the action space.
