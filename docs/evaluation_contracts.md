# Evaluation contracts

The evaluation layer connects a sampled candidate to compilation, correctness,
and benchmark evidence without depending on a sampling strategy. Milestone 2A
is CPU-only. Its mock evaluator never compiles or runs a kernel and labels every
measurement as synthetic.

Each candidate has a stable candidate ID derived from its parameters and search
context. Each evaluation has a separate unique evaluation ID. Repeating an
evaluation therefore preserves the candidate ID but creates a new evaluation
identity and timestamp. Default mock IDs and timestamps are deliberately unique,
so deterministic mock behavior refers to its configured outcome and synthetic
measurement values. Tests that need byte-for-byte records can inject a fixed ID
factory and clock.

## Lifecycle and results

The successful lifecycle is:

```text
pending -> compilation -> correctness_validation -> benchmarking -> completed
```

Any active stage can transition to the terminal `failed` state. Lifecycle state
does not stand in for correctness: correctness is a separate typed result. A
completed record requires successful compilation, passing correctness, and a
benchmark result. A failed record requires failure information and cannot carry
a successful latency result.

Failure categories are:

- `invalid_or_unsupported_candidate`: candidate identity, context, parameters,
  or requested target is not supported by the evaluator.
- `compilation_failure`: compilation did not produce an executable candidate.
- `correctness_failure`: the compiled candidate failed numerical validation.
- `benchmark_failure`: timing could not be completed after correctness passed.
- `evaluation_infrastructure_error`: orchestration or infrastructure failed.

Optional compiler versions and artifact references remain absent when unknown.
Backend-specific details belong in artifacts or `extensions`, not in the shared
contract.

## Evaluators and persistence

An evaluator accepts an immutable sampling record and returns an immutable
evaluation record. It neither selects candidates nor changes sampler state.
Versioned JSONL persistence validates records on read and writes through a
temporary file before atomic replacement. Duplicate evaluation IDs are rejected
before the destination changes. Version 1 rejects unknown top-level fields while
reserving `extensions` for optional forward data.

`MockEvaluator` supports controlled success and failure outcomes. Its latency,
measurement type, and units are explicitly synthetic. They are test data and
must not be interpreted as GPU, CUDA, Triton, or Trainium performance.

Milestone 2B can implement the same evaluator protocol for Triton, CUDA, or NKI.
Those implementations will populate real compiler versions, artifacts,
correctness evidence, and carefully defined measurements without changing the
sampler or the shared lifecycle contract.

Run `examples/mock_evaluate_candidates.py` to sample five configurations, create
synthetic evaluations, persist them, reload them, and print a short summary.
