# Matmul Inspector 0.3.0 release notes

Matmul Inspector 0.3.0 adds supported CPU-only Tensor Game search while keeping
the policy-training and evaluation programs explicitly experimental. It also
includes the GitHub Actions maintenance validated during release preparation.

## Highlights

- The installed `tensor-game-search` command runs sampled MCTS on JSON target
  records, lists, and manifests.
- `bounded_128` is the default residual candidate provider. It uses bounded
  coarse-to-fine residual ranking and exact integer scoring for returned
  candidates.
- `exhaustive` remains available as an explicit compatibility and diagnostic
  fallback.
- Streaming heap selection replaces the earlier full-list bounded selector while
  preserving its outputs on the permanent differential corpus.
- Reproducibility tools cover frozen target manifests, paired provider runs,
  cross-machine comparison, resumable execution, and release-gate auditing.

The repository also contains research utilities for synthetic data, policy and
value learning, neural-guided MCTS, coverage studies, and controlled ablations.
Those tools support experiments. They are not promoted as production training
or model-serving interfaces.

## Usage and compatibility

Install the release and inspect the complete command options:

```sh
python -m pip install matmul-inspector==0.3.0
matmul-inspector tensor-game-search --help
```

The command requires a JSON target and a positive `construction_length`. When a
file contains multiple targets, `--target-id` selects one record. Omitting the
provider selects `bounded_128`:

```sh
matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example
```

Select the exhaustive fallback explicitly with the same required target
arguments:

```sh
matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example \
  --candidate-provider exhaustive
```

Python callers can use
`matmul_inspector.tensor_game_residual.create_residual_candidate_provider()`.
Its default is `bounded_128`; passing `"exhaustive"` selects the fallback. Both
providers currently require a `(4, 4, 4)` target shape.

## Performance evidence

The release decision used a frozen 150-target workload with identical targets,
search settings, and seed for both providers. Timings were analyzed separately
on each host and were not pooled.

| Host | Exhaustive mean | Bounded mean | Mean speedup | Discovery outcomes |
| --- | ---: | ---: | ---: | ---: |
| Local AMD Ryzen 7 7700X under WSL2 | 1.0933 s | 0.5148 s | 2.12× | 134/150 for each provider |
| AWS Intel CPU host | 2.0874 s | 1.0015 s | 2.08× | 134/150 for each provider |

No discovery-outcome mismatch was observed on this corpus. This does not prove
universal discovery equivalence. Bounded ranking is approximate and may choose
a different candidate set or ordering. Latency varies with workload, search
settings, Python runtime, and host hardware.

Earlier controlled component studies found that `bounded_128` avoided expensive
exact discrete processing for about 98% of `(u, v)` pairs while still computing
all contractions. The streaming heap implementation used about 29× less traced
Python peak memory than the original full-list bounded implementation. These are
measurements for the recorded workloads, not general complexity or performance
guarantees.

## Correctness and validation

- Returned candidates remain legal rank-one actions and use exact integer
  residual scoring after bounded prefix selection.
- Differential tests compare streaming selection against the previous bounded
  selector and cover stable ordering, ties, duplicate handling, and budgets.
- Search regressions cover exact transitions, reference decompositions, provider
  selection, JSON target lists, target-shape errors, result metadata, and
  deterministic small-target execution.
- Wheel and source-distribution installations are tested outside the checkout.
  Installed CLI and Python imports do not require ignored benchmark artifacts.
- GitHub CI covers Python 3.10 through 3.13, GCC and Clang CPU CMake/CTest jobs,
  and CUDA compilation with controlled FP inspection. CUDA compilation is not
  CUDA runtime validation.
- The release artifact workflow verifies wheel and source-distribution SHA-256
  integrity across the upload/download handoff before publication can run.

## Limitations

- Search wall-time limits are cooperative. A proposal already in progress is not
  forcibly interrupted at the exact deadline.
- Bounded residual ranking is an approximation. Equal outcomes on the frozen
  benchmark do not establish universal equivalence to exhaustive ranking.
- Policy-training, neural, coverage, and ablation utilities remain experimental.
- Learning workflows require optional PyTorch dependencies, installed with the
  `learning` extra or a compatible environment. GPU experiments separately use
  the `gpu` extra and require a compatible CUDA-enabled PyTorch installation.
- Frozen raw benchmark bundles are ignored local artifacts and are not included
  in the wheel or source distribution. Tracked reports preserve commands,
  hashes, aggregate results, and limitations. A separately reviewed scrubbed
  evidence publication would be needed for public raw-data distribution.
- Construction lengths in generated studies are verified upper bounds, not
  proofs of tensor rank.

## CI and release infrastructure

The release candidate includes validated updates to
`actions/checkout@v7.0.1`, `actions/setup-python@v7.0.0`,
`actions/upload-artifact@v7.0.2`, and `actions/download-artifact@v8.0.2`.
The release workflow builds once, audits both archives, installs both archives,
passes artifacts by exact artifact ID, and publishes only for an approved
version-matching tag. Manual dispatch remains build-only.
