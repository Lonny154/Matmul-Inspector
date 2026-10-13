# TG-4B.2C held-out protocol addendum

TG-4B.2C will evaluate frozen checkpoints on the persisted test partition. This
document covers pre-execution protocol review only. No held-out tensor was opened
and no discovery search was run while preparing the addendum.

## Why an addendum was required

The immutable TG-4B.2B plan had four execution blockers. Its TG-3B relative path
did not exist. It named six checkpoints and five search-ablation families without
an exact matrix. It required a target-cluster bootstrap without its seed, sample
count, or confidence level. Finally, its wall-time field did not say whether the
deadline covered one search or the complete episode.

The original `evaluation_plan.json` remains unchanged. The separately versioned
`evaluation_protocol_addendum_v1.json` records each clarification as a new
pre-execution decision and remains `pending_review`. Its content digest excludes
only the `content_digest` field itself and uses canonical sorted compact JSON.

## Resolutions

The parent TG-4B.2B experiment records
`artifacts/tensor-game/tg3b-mlp-256-seed42` as its source checkpoint. The file at
that location matches the preregistered SHA-256 digest. Execution must reject any
future mismatch rather than copy or alter the checkpoint.

The 10-configuration matrix contains two residual controls, three residual-policy
checkpoint-value runs, two checkpoint-policy heuristic-value runs, and three
matched checkpoint-policy/checkpoint-value runs. This preserves all requested
policy, value, joint, balancing, and constant-optimism contrasts without a full
redundant cross product. Thirty target groups, six seeds, and ten configurations
produce exactly 1,800 planned trials.

The exploratory 95% percentile bootstrap uses 10,000 resamples and seed 20261013.
It resamples originating target groups, keeps all six search seeds for a target
together, and calculates paired target-level success-rate differences. Timeouts
remain failures. Infrastructure failures and missing pairs are reported and
excluded from the primary paired denominator, with an all-fail sensitivity
analysis. Thirty target groups are too few for broad superiority claims.

The established TG-4A implementation applies `episode_wall_time_seconds` to the
entire multi-decision episode. Each inner MCTS invocation receives at most the
remaining episode time. The addendum preserves this behavior and does not change
MCTS semantics.

## Pre-execution validation

Run the validator without an output directory collision:

```sh
PYTHONPATH=src python examples/tensor_game_protocol_addendum.py \
  --plan artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_plan.json \
  --addendum artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_protocol_addendum_v1.json \
  --output artifacts/tensor-game/tg4b2c-heldout-v1
```

The command verifies plan and addendum integrity, checkpoint bytes, checkpoint
loading and value semantics, matrix coverage, trial accounting, wall-time
interpretation, and output-path availability. It does not load the test tensors.
Execution must still reject the addendum until a reviewer changes its status to
`approved` through a separately reviewed change.

## Configuration wiring and durable execution

The first audit found that structural validation did not construct the ten matrix
rows. `tensor_game_heldout_execution.py` now parses each structured row without
using its identifier to infer behavior. It produces a typed configuration with
explicit policy and value sources, checkpoint references, coefficients, frozen
MCTS budgets, the episode deadline, and model provenance.

Checkpoint models are loaded only after their digests and declared semantics
match the plan. A factory-local cache shares an inference-only model when policy
and value use the same checkpoint. Neural policy rows construct
`NeuralActionPolicy`; controls construct `ResidualInformedPolicy`. Heuristic and
constant rows select their exact evaluators. The frozen plan's
`tg3b_predicted_completion_length` label is treated as the recorded name for the
known-completion-length target and dispatches only to `NeuralValueEvaluator`.
Direct observed-return checkpoints dispatch only to `DirectReturnValueEvaluator`.

The synthetic readiness check sends all ten configurations through the real
multi-decision episode runner with a generated one-term `4×4×4` fixture. A valid
bounded failure is acceptable; the check establishes construction and execution,
not discovery quality.

The dry-run schedule uses 30 opaque ordinal slots in deterministic target-digest
order. This avoids reading held-out tensors before approval. Combining those
slots with six frozen seeds and ten sorted configuration IDs produces 1,800
stable trial identifiers. Approval-time execution must resolve each ordinal slot
against the persisted digest ordering without changing its trial identity.

Results use one atomically replaced JSON file per trial plus a resume manifest
bound to protocol and checkpoint digests. Identical repeated writes are accepted;
conflicting duplicates are rejected. Completed, bounded-failure, timeout,
infrastructure-failure, and interrupted records remain distinct. Scheduled IDs
without a record remain explicitly countable as never started. There is no
automatic retry.

Technical validity and reviewer approval are separate. The validator reports
protocol, wiring, fixture, schedule, persistence, and review checks independently.
`execution_ready` remains false while the addendum is `pending_review` even when
every technical check passes.

Reviewer approval is stored separately in
`evaluation_protocol_approval_v1.json`. The validator checks its own content
digest, the frozen plan hash, addendum content digest, approved configuration and
trial counts, episode deadline, and explicit approval statement. Neither frozen
document is edited. With that artifact present, technical readiness and approval
can both pass even though the immutable addendum continues to say
`pending_review`.

Launch from the repository root only after readiness validation:

```sh
PYTHONPATH=src .venv/bin/python examples/tensor_game_heldout_evaluation.py \
  --plan artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_plan.json \
  --addendum artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_protocol_addendum_v1.json \
  --approval artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_protocol_approval_v1.json \
  --experience artifacts/tensor-game/tg4b2a-diverse-full-v1 \
  --output artifacts/tensor-game/tg4b2c-heldout-v1 \
  --progress-every 10
```

After an interruption, use the identical command with `--resume`. Resume scans
atomic trial records before rebuilding progress, so a crash after a result write
but before `progress.json` cannot cause that trial to execute twice.
