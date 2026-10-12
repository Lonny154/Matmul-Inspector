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
