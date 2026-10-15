# Diverse TensorGame MCTS experience

TG-4B.2A expands the small TG-4A integration buffer into a reproducible target
curriculum and a staged collection workflow. This milestone collects and audits
experience. It does not train a network or test discovery improvement.

## Target curriculum

The default curriculum generates 200 exact integer tensors of shape 4×4×4 with
coefficients in `{-1, 0, 1}`. It creates 50 targets at each construction length
from one through four. Every factor is stored in canonical sign form and exactly
reconstructs its target. Construction length is a verified upper bound. It is not
a proof of tensor rank.

Generation is deterministic from a recorded seed. Exact target digests must be
unique. Known construction-trajectory residuals may not collide across targets.
The generator records every zero, duplicate, reserved-target, factor-multiset,
and residual-collision rejection. It does not attempt an expensive uniqueness
proof across all algebraic equivalences.

The 2×2 matrix multiplication tensor is excluded and reserved for later independent
evaluation. Standard and Strassen reference trajectories are not inserted into
the curriculum.

## Splits before search

Targets receive deterministic 70/15/15 train, validation, and held-out test
assignments before MCTS runs. The default counts are 140, 30, and 30. Lengths are
balanced within rounding limits. Every episode inherits its target's assignment.
Exact targets and residuals may not cross partitions.

Persisting the split before collection matters because search outcome cannot
influence partition membership. Difficult and unsuccessful targets remain in the
planned schedule. The TG-4B.1 dataset adapter reads these persisted assignments
instead of making a new split.

## Search configurations and returns

Both configurations use the existing residual-informed proposal policy. One uses
the handcrafted nonzero-count heuristic. The other uses TG-3E's constant optimistic
`max_moves` leaf value. Seeds, proposal limits, simulation limits, state limits,
move limits, and targets are matched. Neither configuration depends on a trained
model. PUCT and environment reward semantics are unchanged.

The move limit is the known construction-length upper bound by default. A success
must reach exact zero and independently reconstruct the target. Move-limit
truncation is preserved as bounded negative experience. It does not prove that a
decomposition is impossible. Infrastructure interruption is censored and receives
no manufactured value target.

## Staged execution

The command defaults to a dry run:

```sh
PYTHONPATH=src python examples/tensor_game_diverse_experience.py
```

An eight-episode smoke test uses one target per length, one seed, and both search
configurations:

```sh
PYTHONPATH=src python examples/tensor_game_diverse_experience.py \
  --smoke --output artifacts/tensor-game/tg4b2a-diverse-smoke-v1
```

The pilot uses ten targets, both seeds, and both configurations for 40 episodes:

```sh
PYTHONPATH=src python examples/tensor_game_diverse_experience.py \
  --pilot --output artifacts/tensor-game/tg4b2a-diverse-pilot-v1
```

The full default plan contains 800 episodes and requires `--execute-full`. Never
infer that flag from an output name. The collector refuses to overwrite a directory.

## Audit and readiness

Every saved episode passes exact replay. The audit reports target, state, action,
and outcome diversity. It includes repeated-state frequencies, cross-target and
cross-split overlap, sampled-support entropy and concentration, search-policy to
prior KL, return distributions, simulation counts, throughput, storage, and
approximate maximum resident memory.

Repeated decisions from one target are correlated. They remain useful replay
evidence but are not independent samples for uncertainty estimates. The audit
separately reports independent target coverage and effective unique training
states.

A buffer is ready for offline pipeline work only when replay is valid, sampled
supports are normalized, action identities are canonical, value masks match
censoring, and target partitions have no residual leakage. Outcome imbalance also
matters. Heavy truncation can teach behavior specific to a tight search budget and
must not be described as mathematical negative proof.

Each experiment writes `manifest.json`, `episodes.json`, `decisions.json`,
`replay_validation.json`, `dataset_audit.json`, and `report.md`. TG-4B.2B should
train preregistered policy-only, value-only, and joint ablations only after the
pilot establishes reasonable cost and data quality. Held-out test targets must
remain untouched until final evaluation.
