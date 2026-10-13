# Hardware adaptive-search comparison

Milestone 3C connects the paired experiment design to the existing real Triton
FC1 evaluator. The research question is whether feedback-guided sampling finds
competitive configurations more efficiently than uniform sampling under equal
logical evaluation budgets. The search is exploratory. It is not confirmation
of a performance improvement.

## Dry run is the default

The command defaults to a CPU-only plan:

```sh
python examples/compare_triton_search.py --dry-run
```

It validates the search space and budget, selects the shared initial candidates,
prints every derived seed, shows adaptive rebuild boundaries, marks future
adaptive selections as pending, checks whether the optional `torch` and `triton`
modules can be found without importing them, and reports expected artifacts.
It does not query CUDA or execute a kernel. Omitting both mode flags has the same
dry-run behavior.

The proposed full plan uses three seeds, 16 logical submissions per arm, five
shared initial candidates, adaptive batches of two, epsilon 0.25, and prior
strength 2.0. The 96 logical arm evaluations require at most 81 physical
candidate evaluations because each shared result is referenced by both arms.
Future cross-arm coincidences are not silently cached or shared.

## Explicit hardware execution

Hardware execution requires `--execute` and a new output directory:

```sh
python examples/compare_triton_search.py \
  --execute \
  --output artifacts/hardware-search/fc1-full-001
```

Prerequisites are an NVIDIA RTX 4060 Ti or another SM 8.9 device, a compatible
driver, CUDA-enabled PyTorch, and Triton. The existing `TritonGemmEvaluator`
checks CUDA availability and compute capability. It compiles, validates against
PyTorch with FP16 tolerances, warms up, and records repeated CUDA-event samples.
Compilation and setup stay outside measured kernel latency. Host submission and
synchronized wall clocks remain separate in evaluation extensions.

Environment evidence includes Python, PyTorch, Triton, CUDA runtime, GPU name,
compute capability, memory size, and the NVIDIA driver version reported by
`nvidia-smi`. If that management interface is unavailable, driver version is
recorded as unavailable without preventing evaluation.

Do not launch the full plan as a first test. After reviewing the dry run, the
recommended hardware smoke test is:

```sh
MATMUL_INSPECTOR_RUN_CUDA_TESTS=1 \
python examples/compare_triton_search.py \
  --execute \
  --trial-seed 7 \
  --budget 6 \
  --initial-size 5 \
  --batch-size 1 \
  --warmups 5 \
  --iterations-per-sample 10 \
  --samples 3 \
  --output artifacts/hardware-search/fc1-smoke-001
```

This smoke plan has 12 logical evaluations and at most seven physical candidate
evaluations. The environment variable gates the optional pytest hardware test;
the CLI still requires the separate `--execute` opt-in.

## Budget and feedback fairness

A logical submission consumes one unit when a candidate is sent to the evaluator.
Compilation, correctness, and benchmark failures consume budget and receive no
latency reward. A GPU infrastructure failure stops the current trial. It may
leave unequal arm counts if the failure occurs between the two physical
executions of a paired step. That exception is explicit in the trial status and
must not be presented as an equal-budget comparison.

Both arms receive the same initial candidate IDs and the same physical evaluation
records. Those records count once toward physical work and once toward each arm's
logical budget. Later candidates cannot repeat a candidate already submitted by
the same arm.

Before each later adaptive batch, a new Milestone 3A sampler is constructed from
the complete deduplicated adaptive history. Only completed, correct, nonsynthetic
CUDA-event measurements are eligible. Uniform-only results never enter adaptive
feedback. Uniform sampling never reads adaptive results. Epsilon and prior
strength are frozen in the experiment manifest.

The scheduling seed chooses which arm runs first in the first paired step. The
runner then alternates the first arm across steps. GPU work is sequential. This
counterbalances a fixed arm-order bias but does not eliminate drift from thermals,
clocks, background processes, or compilation caching.

## Outputs and metrics

Every unique output directory contains:

- `experiment.json`: versioned manifest, plans, seeds, logical histories,
  physical order, rebuild diagnostics, environment, accounting, and wall time;
- `evaluations.jsonl`: unique physical evaluation records with correctness and
  raw CUDA-event distributions;
- `progress.json`: plotting data for best valid latency versus equal budget;
- `report.md`: concise human-readable status and results.

Per-arm records include submitted and successful counts, failures by category,
unique coverage, threshold crossing, best latency, and best-so-far progress.
Paired progress reports adaptive-minus-uniform best latency at every equal
logical budget. Search wall time is recorded separately from kernel latency.

## Separate confirmation

Search winners are hypotheses. Inspect the search artifacts first. Planning a
deduplicated confirmation does not touch CUDA:

```sh
python examples/confirm_triton_search.py \
  artifacts/hardware-search/fc1-pilot-seed101/experiment.json \
  artifacts/hardware-search/fc1-pilot-seed202/experiment.json \
  artifacts/hardware-search/fc1-pilot-seed303/experiment.json
```

Explicit confirmation reuses the existing controlled benchmark. It benchmarks
all unique arm winners against the same preallocated PyTorch/cuBLAS baseline in
randomized interleaved rounds:

```sh
python examples/confirm_triton_search.py \
  artifacts/hardware-search/fc1-pilot-seed101/experiment.json \
  artifacts/hardware-search/fc1-pilot-seed202/experiment.json \
  artifacts/hardware-search/fc1-pilot-seed303/experiment.json \
  --execute \
  --output artifacts/hardware-search/fc1-winner-confirmation
```

Confirmation is never launched automatically after search.

## Cost and limitations

The default evaluator performs 100 warmup launches and 10 samples of 500 timed
launches per successful physical candidate, plus compilation and correctness
work. The full maximum of 81 physical evaluations therefore represents roughly
413,100 warmup and timed Triton launches, before setup and any failures. Actual
wall time is hardware and compilation-cache dependent and is not predicted by
the dry run.

The search covers one bounded FC1 space, workload, precision, device class, and
adaptive model. Shared initial observations introduce correlation by design.
Later physical measurements are not simultaneous. Failed candidates can give
arms different amounts of eligible feedback despite equal attempted budgets.
The parameter-wise model cannot represent arbitrary interactions. Search curves
show discovery progress in this experiment; they do not prove causal superiority
or generalize to other kernels.

The AlphaTensor connection is limited to sampled actions and runtime feedback.
This implementation does not learn matrix-multiplication algorithms and does not
use reinforcement learning, tree search, or a neural policy.
