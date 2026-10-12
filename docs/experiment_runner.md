# CUDA/Triton experiment runner

The experiment runner executes one predefined FC1 candidate against a
PyTorch/cuBLAS baseline. It is deliberately a small submission boundary for a
future optimization agent: the agent supplies structured candidate parameters,
not source code or a shell command. The only candidate currently in the registry
is the existing `scripts/triton_fc1.py` kernel.

## Run the default FC1 experiment

From a source checkout with CUDA-enabled PyTorch and Triton installed:

```sh
PYTHONPATH=src python -m matmul_inspector.cli experiment
```

The default workload is `(512x768) @ (768x3072)` in FP16. It uses seed 0,
`atol=0.01`, `rtol=0.01`, 100 warmups, 500 timed iterations per measurement,
and 10 alternating-order rounds. Print the complete default configuration with:

```sh
PYTHONPATH=src python -m matmul_inspector.cli experiment --print-default-config
```

To submit a configuration saved as `experiment.json`:

```sh
PYTHONPATH=src python -m matmul_inspector.cli experiment --config experiment.json
```

The candidate section has this form:

```json
{
  "implementation": "triton_fc1",
  "parameters": {
    "block_m": 64,
    "block_n": 128,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3,
    "group_size_m": 1
  }
}
```

All top-level fields printed by `--print-default-config` are required. Unknown
fields and implementation names are rejected; arbitrary commands and generated
source are never executed.

## Output and measurement

Each invocation creates a unique ignored directory under
`artifacts/experiments/`:

```text
<run-id>/
  submitted_config.json
  resolved_config.json
  result.json
  timings.json
  timings.csv
  stage.log
```

`result.json` records the schema version, run ID, UTC timestamps, Git commit and
dirty state, resolved configuration, implementation identities and parameters,
dependency versions, CUDA device, setup/first-invocation wall time, correctness,
and timing summaries. `timings.json` and `timings.csv` retain every round and its
baseline/candidate order. The reported steady-state value is mean device elapsed
time per GEMM in milliseconds, measured with CUDA events over the configured
iterations. Speedup is baseline median latency divided by candidate median
latency. A speedup close to one is only a measurement, not evidence of a useful
optimization or an end-to-end Transformer improvement.

The candidate is checked for shape, finite values, and explicit tolerances
before any timed rounds. Correctness failures are never benchmarked. FP32 Triton
accumulation and FP16 output are recorded; PyTorch/cuBLAS accumulation is labeled
implementation-selected rather than assumed.

## Failures and future agents

Configuration, prerequisite, compilation, execution, correctness, benchmark,
timeout, and interruption failures produce a nonzero exit code and a structured
record when practical. The CUDA worker runs in a separate process. On timeout it
is terminated (and killed if necessary), so a hung kernel cannot indefinitely
block the controlling process. Missing CUDA, PyTorch, Triton, or the repository
kernel produces an actionable prerequisite failure.

Python callers can use `matmul_inspector.experiment.run_experiment(config)` and
receive the same structured result written to disk. A future AI agent should
choose a supported implementation and submit a JSON configuration through this
function or CLI. Expanding the candidate registry should remain an explicit code
change with review; it should not become dynamic source execution.

## Sequential candidate batches

The native batch operation accepts shared single-experiment settings plus an
explicit candidate list, scheduling seed, and repetition count. Run the checked-in
four-candidate FC1 example with:

```sh
PYTHONPATH=src python -m matmul_inspector.cli experiment-batch \
  --config examples/fc1_experiment_batch.json \
  --output-root artifacts/experiment-batches
```

The [example configuration](../examples/fc1_experiment_batch.json) compares
128x128, 128x64, 64x128, and 64x64 Triton tiles for the same
`M=512, K=768, N=3072` FP16 workload. Each uses `BLOCK_K=32`, four warps, three
stages, and `GROUP_SIZE_M=1`. It requests 100 warmups, 500 iterations per CUDA
event measurement, 10 within-experiment rounds, and three batch repetitions.

Batch JSON has this top-level shape:

```json
{
  "shared": {
    "workload": {"m": 512, "k": 768, "n": 3072, "dtype": "float16"},
    "seed": 0,
    "correctness": {"atol": 0.01, "rtol": 0.01},
    "benchmark": {"warmups": 100, "iterations": 500, "rounds": 10},
    "timeout_seconds": 300.0,
    "baseline": {"implementation": "pytorch_matmul"}
  },
  "candidates": [
    {"id": "unique-id", "implementation": "triton_fc1", "parameters": {}}
  ],
  "scheduling_seed": 20261012,
  "repetitions": 3
}
```

The abbreviated empty `parameters` object above illustrates structure only; use
all six parameters shown in the checked-in example. Candidate IDs must be unique.
Malformed batch structure or shared settings reject the batch. A candidate with
invalid implementation or parameters receives its own configuration-failure
entry, while other candidates continue.

For every repetition, candidate order is shuffled deterministically from the
scheduling seed. Execution is strictly sequential to avoid competing GPU work.
Every scheduled item calls the single-experiment runner, so it creates fresh
seeded inputs and a freshly measured matched PyTorch/cuBLAS baseline. The
single-run alternating baseline/candidate measurement order remains unchanged.

The batch directory contains:

```text
<batch-id>/
  submitted_batch_config.json
  resolved_batch_config.json
  planned_schedule.json
  batch_results.json
  summary.json
  summary.csv
  experiments/
    <experiment-run-id>/
      result.json
      resolved_config.json
      timings.json
      timings.csv
      stage.log
```

The planned schedule is saved before execution. `batch_results.json`,
`summary.json`, and `summary.csv` are replaced atomically after every attempted
experiment, so a failure or interruption leaves valid partial records. Entries
contain candidate ID and parameters, repetition, correctness, baseline and
candidate median milliseconds, paired baseline/candidate speedup, failure stage,
child run ID, and result path. Unavailable values are JSON null or blank CSV
fields. Compilation, execution, correctness, benchmark, and timeout failures do
not stop later candidates. An interruption stops new scheduling after the active
single runner has cleaned up its worker.

Successful runs are aggregated per candidate. `median_run_speedup` is the median
of their run-level paired speedups. Its reported spread is the median absolute
deviation of those same values. The descriptive ranking includes only candidates
whose configuration is valid and whose every repetition completes correctness
and benchmarking successfully. Slower, failed, invalid, and incomplete candidates
remain in the artifacts; the ranking does not claim statistical significance.

Batch CLI exit codes are:

- `0`: every scheduled experiment succeeded and every candidate was valid.
- `1`: execution completed, but at least one candidate was invalid or one
  experiment failed.
- `2`: the batch file could not be read, or batch/shared structure was invalid.
- `130`: execution was interrupted; the partial summary was persisted.

Python agents can call
`matmul_inspector.experiment_batch.run_batch(config, output_root=...)` and consume
the returned batch record or its `summary.json`. Agents submit only registered
implementation names and structured parameters; batches do not execute arbitrary
commands or generated source.

## Persistent search sessions

A session fixes the objective, FC1 workload, baseline, tolerances, benchmark
settings, allowed Triton parameter values, scheduling seed, and total experiment
budget. One budget unit is one invocation of the single-experiment runner,
whether it succeeds, fails correctness, fails compilation, or times out. Thus
four candidates repeated three times cost 12 units.

Create the checked-in 24-unit example session:

```sh
PYTHONPATH=src python -m matmul_inspector.cli session create \
  --config examples/fc1_experiment_session.json \
  --output-root artifacts/experiment-sessions
```

The command prints the new session directory. Save it as `SESSION`, then inspect
the concise state or export complete agent-readable context:

```sh
PYTHONPATH=src python -m matmul_inspector.cli session status --session "$SESSION"
PYTHONPATH=src python -m matmul_inspector.cli session context \
  --session "$SESSION" --output /tmp/fc1-session-context.json
```

Submit exactly one proposal and inspect the updated context:

```sh
PYTHONPATH=src python -m matmul_inspector.cli session submit \
  --session "$SESSION" --proposal examples/fc1_session_proposal.json
PYTHONPATH=src python -m matmul_inspector.cli session context --session "$SESSION"
```

The proposal contains a unique proposal ID, candidate IDs, registered
implementation, parameter overrides, a hypothesis for each candidate, references
to prior proposal IDs or result paths, and a repetition count. Omitted candidate
parameters resolve to the established defaults before validation and
fingerprinting. Candidate fingerprints include the implementation, all resolved
parameters, and the fixed session context; changing key order or candidate name
does not create a new configuration. Repeating an accepted configuration requires
a non-empty `repeat_reason` such as `"Verify a noisy prior result."`.

Proposal files cannot change shared settings. Unsupported implementations,
out-of-range parameters, duplicate IDs, unknown prior-result references, and
over-budget proposals are rejected and recorded without consuming budget. The
entire proposal cost is reserved atomically before execution. A nonblocking
session lock prevents concurrent submit or recovery operations from spending the
same budget.

Session state uses these proposal states:

- `accepted`: validated and fully reserved, before batch launch.
- `running`: batch execution began; a stale value after process death requires
  recovery rather than automatic rerun.
- `completed`: every child experiment completed successfully.
- `failed`: execution finished with a failure, or ambiguity was explicitly
  reconciled without rerunning.
- `interrupted`: the batch stopped on interruption and every launch was resolved.
- `needs_reconciliation`: one or more invocations may have launched but lack
  conclusive result artifacts; their units remain reserved.

After a normal interruption, child artifacts identify consumed launches and the
unstarted tail is released. Following an unexpected process failure, inspect a
stale `running` or `needs_reconciliation` proposal:

```sh
PYTHONPATH=src python -m matmul_inspector.cli session recover \
  --session "$SESSION" --proposal-id PROPOSAL_ID --resolution inspect
```

Inspection incorporates any conclusive child results but never reruns them. If
ambiguity remains, explicitly choose `--resolution consume` to conservatively
charge all reserved units, or `--resolution release` only after establishing
externally that they never launched. No recovery action starts GPU work.
An `accepted` proposal with no batch artifacts is known not to have launched, so
`--resolution inspect` releases its reservation safely.

`session.json` is atomically replaced and contains provenance, budget totals,
proposal history, fingerprints, batch links, and individual result paths.
Accepted and recovered proposal records are also kept below `proposals/`; rejected
submissions are stored below `rejected/`. Context export separates candidate
hypotheses from measured observations, retains failures and slower results, and
reuses the batch runner's descriptive rankings and variability. Historical
artifacts are not imported in this milestone: the first session starts fresh.

An example instruction for Codex is:

> Read this session context, propose at most four new configurations with
> hypotheses, submit exactly one batch within the remaining budget, and report
> measured findings.
