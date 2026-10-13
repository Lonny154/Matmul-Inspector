# Controlled adaptive-search experiments

Milestone 3B adds a CPU-only paired experiment. It asks a narrow question: under
a fixed simulated landscape and a fixed attempted-evaluation budget, does the
Milestone 3A adaptive policy find lower-latency candidates than uniform sampling?
The simulation is experiment-design evidence. It is not a GPU benchmark and
does not establish an adaptive-search advantage on real kernels.

## Trial design

Each replicate has uniform and adaptive arms. They share an immutable bounded
search space, simulated objective, first batch, evaluation budget, and batch
schedule. Arm sampler state and RNG state remain separate. Seeds are derived by
hashing the master seed, replicate index, arm or round, and purpose. The result
records the derived seeds.

The primary budget is the number of candidates submitted to the evaluator.
Compilation, correctness, benchmark, and other candidate failures consume one
unit. They are not replaced. Proposal attempts, duplicate proposals, invalid
proposals, successful and failed evaluations, unique candidates, eligible
feedback, and completed rounds are separate ledger entries. Normal trials end
with equal submitted budgets. An infrastructure failure stops both arms after
the same paired step and records `paired_infrastructure_stop`.

Equal attempted-evaluation budgets do **not** guarantee equal successful or
feedback-eligible counts. Different proposals can encounter different failures.
That imbalance is recorded as an outcome rather than corrected with extra work.

## Cumulative adaptive rebuilding

The initial batch is identical in both arms. Before each later adaptive batch,
the runner creates a fresh `AdaptiveSampler` and gives it the arm's complete,
deduplicated evaluation history. It never incrementally updates a prior adaptive
state. Failed candidates remain in the trial's seen set even though they do not
influence parameter scores.

Each rebuild records the round, source evaluation and candidate IDs, eligible
feedback count, SHA-256 digest of the feedback identity list, sampler settings,
proposal seed, and pre-proposal RNG state. Rebuilding the same round from the
same history and seed reproduces its proposals.

Synthetic observations remain rejected by `AdaptiveSampler` by default. The
runner uses the explicit `allow_synthetic_feedback=True` option. Simulation
records still set `BenchmarkResult.synthetic=True` and use the measurement type
`simulated objective latency`, so they cannot be confused with CUDA events.

## Simulated evaluator and metrics

The evaluator implements existing evaluation records without CUDA or Triton.
Objective noise and injected failures are deterministic functions of the
landscape seed and candidate. Evaluation order and arm identity cannot change a
candidate's latent outcome.

Available landscapes are:

- `structured`: additive parameter effects that the parameter-wise model can learn;
- `interaction`: an XOR-like tile interaction that challenges independent scores;
- `flat`: no informative objective structure apart from optional fixed noise.

Failure injection can exercise compilation, correctness, benchmark, invalid, or
infrastructure outcomes. The complete finite landscape supplies a known optimum
and ranking. Per-step evidence includes best latency, simple regret, best rank,
evaluations to the known optimum, unique coverage, successful evaluations, and
eligible feedback. Aggregate output retains raw final metrics for every paired
replicate. It reports descriptive means, medians, paired adaptive-minus-uniform
deltas, and win/tie counts. It does not perform or claim a confirmatory
statistical test.

## Run the example

From the repository root:

```sh
python examples/compare_adaptive_sampling.py \
  --landscape structured \
  --replicates 8 \
  --budget 16 \
  --batch-size 4 \
  --seed 314159 \
  --output adaptive_trial.json
```

The command refuses to overwrite its output. `adaptive_trial.json` uses strict
schema version 1. Unknown top-level fields are rejected. Future optional data
may go in `extensions`; new required or top-level fields require a new schema
version. Embedded evaluations retain the existing evaluation schema.

## Threats to validity and GPU gate

The additive adaptive model cannot represent general parameter interactions.
Results depend on the chosen search space, landscape family, seed set, batch
size, prior, epsilon, noise, and failure mechanism. Sharing the initial batch
reduces one source of variation but does not make later candidate sets identical.
Simulation cost also omits compilation caching, thermal effects, timing drift,
and hardware-specific invalid configurations.

Before a GPU experiment, review the budget ledger, failure policy, seed plan,
batch schedule, cumulative rebuild evidence, candidate validity rules, and the
planned paired analysis. Then pre-register those choices. Do not tune them after
seeing GPU outcomes. A GPU pass should use fresh matched measurements and must
not claim an advantage from one seed, one workload, or unmatched baselines.
