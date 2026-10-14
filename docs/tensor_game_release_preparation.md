# Tensor Game release preparation

## Capabilities

Matmul Inspector now includes CPU-compatible Tensor Game search with two
residual candidate providers. `bounded_128` is the public CLI and Python factory
default. `exhaustive` remains available through explicit selection.

The repository also includes the TG-4C through TG-4Y research tools and reports
covering learned-policy regression, candidate coverage, residual supervision,
bounded ranking, streaming memory optimization, held-out evaluation, and the
cross-machine release gate.

## Evidence

The frozen 150-target study produced identical discovery outcomes for exhaustive
and bounded ranking on the local AMD host and an AWS Intel host. Both providers
solved 134/150 targets on each host. Mean end-to-end speedup was 2.12× locally
and 2.08× on AWS. Timing was analyzed per host and was not pooled.

The one-sided 95% upper bound for an unobserved exhaustive-only target regression
remains 1.977%, using 150 independent targets. Replaying those targets on a
second host does not create another 150 independent samples.

The streaming heap preserved the validated bounded candidate output while
reducing mean traced peak memory by about 29× in the TG-4T study.

## Usage

```sh
matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example

matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example \
  --candidate-provider exhaustive
```

Python callers use `create_residual_candidate_provider()` for the bounded
default or pass `"exhaustive"` explicitly.

## Limitations

Bounded ranking is an approximation. It can return a different candidate set or
ordering from exhaustive ranking. Equal discovery on the tested corpora is not
proof of universal equivalence. Performance depends on workload and hardware.
Construction lengths in the generated studies are verified upper bounds rather
than proven tensor ranks.

Eight exhaustive AWS trials reached the wall-time limit. All were shared-failure
targets, and no timeout was reclassified as a success.

## Evidence storage

Generated evidence remains under the ignored local `artifacts/tensor-game/`
tree. It includes TG-4W raw results, the TG-4X AWS bundle and formal comparison,
and TG-4Y release-gate records. These generated files are intentionally not
version-controlled under the repository's current artifact policy. The tracked
reports record the commands, hashes, aggregate results, and limitations.

## Before publishing

- Review the three local commits and ignored evidence bundle.
- Decide how raw benchmark evidence should be archived outside Git.
- Run the repository's argument-driven native/GPU harnesses with their required
  executables and fixtures in the intended CI workflow.
- Review package contents in a clean build environment.
- Choose a package-version change separately.
- Do not tag or publish until a dedicated release review approves those steps.
