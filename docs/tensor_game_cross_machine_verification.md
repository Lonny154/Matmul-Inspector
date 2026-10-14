# TG-4X cross-machine verification preparation

TG-4X is prepared but has not been run on AWS. The workflow reuses the exact
TG-4W target manifest and evaluation specification. It does not generate new
targets or alter search settings.

> Follow-up: the full AWS search replay was later completed successfully. Its
> installed CLI audit exposed one missing transferred dependency,
> `tensor_game_out_of_sample.py`. The transfer lists below now include it. This
> correction does not change the frozen workload or recorded search results.
>
> TG-4Y subsequently changed the public default to `bounded_128`. This TG-4X
> harness intentionally records the pre-rollout exhaustive default and is now a
> historical replay tool, not a current-default compatibility test.

## Frozen inputs

| File | SHA-256 |
|---|---|
| `target_manifest.json` | `c55792e3866d2064eae411c5427fdb7b2513b910f5670bbb37d439ea03776e75` |
| `evaluation_spec.json` | `1f51262e5b1737812d091c8183fd1eb09c26a30b21c89d876d1a873d325ebdab` |

The full schedule is 150 targets, seed 3, and two providers, for 300 trials.
Candidate budget, simulation budgets, node limit, timeout, `c_puct`, and
alternating provider order come directly from the frozen specification. The
runner refuses any hash or semantic mismatch.

## What the runner records

`tensor_game_cross_machine_verification.py` records the CPU model, architecture,
logical CPU count, OS, Python and dependency versions, Git revision and dirty
state, runner hash, frozen artifact hashes, provider identity, discovery outcome,
proposal latency, non-proposal search time, total search latency, simulations,
expansions, termination reason, and execution order.

Timing is summarized independently for each host. The comparison command does
not pool heterogeneous-host timing. Discovery outcomes are compared by exact
`(target_id, seed, provider)` keys.

Results are atomically rewritten after every trial. Restarting the same `run`
command validates that existing rows are a unique exact prefix of the frozen
schedule, then continues with the next trial. Reordered, duplicated, incomplete,
or schema-incompatible rows are rejected.

## Files to transfer

The current Git revision predates the uncommitted provider integration. Transfer
these files into the isolated EC2 clone, preserving their relative paths:

```text
examples/tensor_game_cross_machine_verification.py
src/matmul_inspector/cli.py
src/matmul_inspector/tensor_game_bounded_residual.py
src/matmul_inspector/tensor_game_cli.py
src/matmul_inspector/tensor_game_out_of_sample.py
src/matmul_inspector/tensor_game_residual_supervision.py
src/matmul_inspector/tensor_game_search_calibration.py
artifacts/tensor-game/tg4w-rollout-decision-v1/target_manifest.json
artifacts/tensor-game/tg4w-rollout-decision-v1/target_manifest.sha256
artifacts/tensor-game/tg4w-rollout-decision-v1/evaluation_spec.json
artifacts/tensor-game/tg4w-rollout-decision-v1/evaluation_spec.sha256
```

The core Tensor Game and MCTS modules are already tracked at revision
`4a83b8d6677de426297226f2f72ab5c959bd3554`. No CUDA, NKI, Neuron, model-weight,
or Trainium artifact is required.

## EC2 setup

Run this in a normal shell on the existing instance. Do not activate the Neuron
virtual environment. The commands clone the existing repository locally, so the
Trainium/NKI working tree remains untouched.

```sh
export TG4X_SOURCE_REPO="$HOME/Matmul-Inspector"
export TG4X_REPO="$HOME/matmul-inspector-tg4x"

git clone --no-local "$TG4X_SOURCE_REPO" "$TG4X_REPO"
git -C "$TG4X_REPO" checkout --detach 4a83b8d6677de426297226f2f72ab5c959bd3554
mkdir -p "$TG4X_REPO/artifacts/tensor-game/tg4w-rollout-decision-v1"
```

If the existing repository uses a different case or location, change only
`TG4X_SOURCE_REPO`.

From the local workstation, transfer the exact files. Replace the host value
with the existing instance's SSH hostname or address.

```sh
export TG4X_AWS_HOST="ubuntu@YOUR_EXISTING_EC2_HOST"
export TG4X_REMOTE_REPO="/home/ubuntu/matmul-inspector-tg4x"

rsync -avR \
  examples/tensor_game_cross_machine_verification.py \
  src/matmul_inspector/cli.py \
  src/matmul_inspector/tensor_game_bounded_residual.py \
  src/matmul_inspector/tensor_game_cli.py \
  src/matmul_inspector/tensor_game_out_of_sample.py \
  src/matmul_inspector/tensor_game_residual_supervision.py \
  src/matmul_inspector/tensor_game_search_calibration.py \
  artifacts/tensor-game/tg4w-rollout-decision-v1/target_manifest.json \
  artifacts/tensor-game/tg4w-rollout-decision-v1/target_manifest.sha256 \
  artifacts/tensor-game/tg4w-rollout-decision-v1/evaluation_spec.json \
  artifacts/tensor-game/tg4w-rollout-decision-v1/evaluation_spec.sha256 \
  "$TG4X_AWS_HOST:$TG4X_REMOTE_REPO/"
```

Back on EC2, create an isolated CPU-only environment and install the copied
working tree. This project has no required runtime dependencies.

```sh
cd "$HOME/matmul-inspector-tg4x"
python3 -m venv .venv-tg4x
. .venv-tg4x/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
```

## Integrity and smoke commands

```sh
cd "$HOME/matmul-inspector-tg4x"
. .venv-tg4x/bin/activate

python examples/tensor_game_cross_machine_verification.py inspect \
  --inputs artifacts/tensor-game/tg4w-rollout-decision-v1

python examples/tensor_game_cross_machine_verification.py smoke \
  --inputs artifacts/tensor-game/tg4w-rollout-decision-v1 \
  --output artifacts/tensor-game/tg4x-aws-smoke \
  --cli matmul-inspector \
  --smoke-targets 4
```

Inspect `host_metadata.json`, `compatibility.json`, `results.csv`, and
`summary.json` before starting the full run. The smoke uses the first four frozen
targets and their unchanged search budgets. Smoke timings are not part of the
full comparison.

## Full execution and deterministic resume

Do not run this command until the AWS benchmark is intentionally authorized.

```sh
cd "$HOME/matmul-inspector-tg4x"
. .venv-tg4x/bin/activate

python examples/tensor_game_cross_machine_verification.py run \
  --inputs artifacts/tensor-game/tg4w-rollout-decision-v1 \
  --output artifacts/tensor-game/tg4x-aws-full \
  --cli matmul-inspector
```

If interrupted, run the identical command again. Do not edit `results.csv` and
do not use the smoke directory as the full output directory.

## Return and comparison

From the local workstation:

```sh
export TG4X_AWS_HOST="ubuntu@YOUR_EXISTING_EC2_HOST"

rsync -av \
  "$TG4X_AWS_HOST:/home/ubuntu/matmul-inspector-tg4x/artifacts/tensor-game/tg4x-aws-full/" \
  artifacts/tensor-game/tg4x-cross-machine-preparation-v1/aws-full/

PYTHONPATH=src:. python examples/tensor_game_cross_machine_verification.py compare \
  --left artifacts/tensor-game/tg4w-rollout-decision-v1/results.csv \
  --right artifacts/tensor-game/tg4x-cross-machine-preparation-v1/aws-full/results.csv \
  --output artifacts/tensor-game/tg4x-cross-machine-preparation-v1/cross-machine-comparison.json
```

The TG-4W local results are the left-hand discovery baseline. Timing remains
stratified as `left_timing` and `right_timing`. A final rollout decision should
also inspect compatibility failures, missing trial keys, discovery mismatches,
timeouts, and host provenance before applying the TG-4W release gate.

## Local preparation result

The four-target local smoke completed eight trials. Both providers succeeded on
all four targets and bounded was faster in all four pairs. This validates the
harness path only. It is not additional independent-machine evidence and does
not resolve the TG-4W cross-machine gate.
