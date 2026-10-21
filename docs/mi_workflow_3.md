# MI-Workflow-3 — trusted local Triton kernels

## Result

**Workflow decision A: a standalone local Triton kernel ran through the paired
comparison without a registry edit. Performance decision: INCONCLUSIVE.** The
RTX 4060 Ti demonstration compared the registered FC1 kernel (A) with a
standalone file containing the same JIT kernel body and launch configuration
(B). Both passed numerical checks on three seeded inputs. Eight paired timing
rounds completed. The CUDA-event and synchronized host-wall clocks disagreed,
and the paired directions split four to four. There is no supported speedup.

This feature addresses the local-source friction in [MI-Dogfood-1](mi_dogfood_1.md)
and [MI-Workflow-2](mi_workflow_2.md). It adds no kernel discovery, model
integration, or new timing engine. The historical MI-Workflow-1 and
MI-Workflow-2 evidence remains unchanged.

## Try a local kernel

Start with the [standalone example](../examples/kernels/mi_workflow_3_fc1.py).
It imports Triton, defines a JIT kernel, and exposes one function:

```python
def matmul_into(a, b, out, parameters):
    # Write FP16 (M,K) @ (K,N) into the preallocated FP16 `out` tensor.
    # Return None, or the Triton compiled launch result for PTX capture.
```

The function receives CUDA tensors `a` and `b`, a preallocated output of
shape `(M,N)`, and the six validated configuration integers `block_m`,
`block_n`, `block_k`, `num_warps`, `num_stages`, and `group_size_m`. It must
overwrite the full output and return `None` or the Triton compiled launch
result. Returning an output tensor or another object violates the contract.
The tool fills outputs with NaN before correctness checks, rejects unfilled
elements, compares shape and dtype, checks finiteness, and uses the unchanged
PyTorch/cuBLAS reference and `atol=rtol=0.01` example tolerances. It checks
seed 0 before timing and seeds 1 and 2 as additional preallocated-output
checks. Correctness failures prevent benchmarking and performance claims.

Copy the [v2 paired spec](../examples/mi_workflow_3_local_pair.json), then
replace B's `source.path` with your Python file. Keep `source.entry_point` as
`matmul_into` and `source.interface_version` as `matmul_into_v1`. A and B may
each be registered FC1 or trusted local source. The initial v2 workflow uses
eager **preallocated** output on both sides so registered and local wrappers
have the same output-allocation boundary. The [v2 schema](../schemas/mi_workflow_3_pair_v2.schema.json)
and Python validator enforce the source fields, configuration, and existing
bounded workload limits. The previous registered-only v1 spec and command
remain valid.

Run from the repository root with a new output path:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_workflow_3_local_pair.json \
  --allow-trusted-local-source \
  --output results/my-local-fc1-pair
```

Read `report.md` first. It combines correctness, per-round paired timing,
uncertainty, timer discrepancies, source identities, compiler metadata, and
PTX differences. Verify the bundle later, without CUDA or source execution:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --verify results/my-local-fc1-pair
```

## Trust boundary and evidence

`--allow-trusted-local-source` is required for any local A or B kernel. It is
**not a sandbox**. Import-time and call-time Python can use the process's
permissions, environment variables, network access, and filesystem. Run only
source you trust. Parsing a spec and verifying a saved bundle read bytes but
do not import the local module. The command imports only the selected file
after opt-in, validation, and a hash-checked copy into the new bundle. It
does not search for kernels or register them internally.

The source copy is made **before** the GPU worker starts. The worker executes
that copy. `experiment.json` stores a relative `sources/B.py` path, the
original path, entrypoint, and SHA-256. `raw_result.json` records the worker's
snapshot path. The manifest covers both the local snapshot and packaged FC1
source. The verifier rehashes the snapshots and recomputes the decision from
raw rounds without importing either implementation. Later edits to the
original file cannot change what this run measured. For a local-source
failure, artifact records retain the stage but omit exception details and
tracebacks that might contain environment secrets. The tool does not save
environment-variable values. A developer should keep credentials out of
source files and artifact paths.

The paired runner retains its eight-round AB/BA ordering, CUDA-event blocks,
host clocks, unranked timer controls, uncertainty summary, and conservative
decision rule. Preallocated mode is a different benchmark mode from the
allocating MI-Workflow-2 run. Their latency numbers are not pooled.

## Retained RTX 4060 Ti demonstration

The exact command used the checked-in v2 example and the versioned
[`local-fc1-v1` bundle](../artifacts/mi-workflow-3/local-fc1-v1/). M=512,
K=768, N=3072; FP16; 30 warmups per kernel; 100 operations per block; eight
paired rounds. Both implementations used `64×128×32` tiles, four warps,
three stages, and `group_size_m=1`. A used the registered source SHA-256
`a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B used the standalone snapshot SHA-256
`9f6b6ecab30fcc7c75e683685896cc2e2d3b56ea6c5ac71408017da2b935aa73`.

Both kernels matched the independent PyTorch/cuBLAS output's shape and FP16
dtype. Both had zero nonfinite values and 0.0 maximum absolute error on the
benchmark seed and both additional seeds. These checks cover three inputs,
not every possible input.

| Measured timing | A: registered | B: local |
|---|---:|---:|
| CUDA-event median ms/operation | 0.063646 | 0.064020 |
| CUDA-event range ms/operation | 0.063334–0.068525 | 0.063459–0.068523 |
| CUDA-event CV | 2.80% | 2.79% |
| Host-wall median ms/operation | 0.062231 | 0.062644 |

The paired A-minus-B CUDA-event median was **−0.000155 ms/operation**, with
range **−0.005188 to +0.004504** and median absolute deviation **0.001945**.
Four rounds favored A and four favored B. The unranked event-minus-wall
controls were **+0.135485 ms/block for A** and **+0.129513 ms/block for B**.
All ranked blocks also had positive discrepancies. The bundle therefore
classifies timing as `INCONCLUSIVE` and withholds a speedup. The difference
between wrappers, compilation identities, or code location is not isolated
as a causal factor in any timing shift.

Triton reported 24,576 shared-memory bytes/program, 121 registers/thread,
four warps, three stages, and zero spills for both kernels. Each captured
PTX has 408 static instructions with the same instruction-mnemonic and
PTX-register counts. Normalized PTX text differs by 133 lines. The saved diff
places those changes in debug-info sections, including source paths. That
does not prove identical final machine code. Decoded SASS was unavailable.

The bundle retains 16 raw timing rows, original GPU result, all three-seed
correctness records, source snapshots, PTX files, normalized diff, compiler
metadata, provenance, and 17 hashed files. Its manifest SHA-256 is
`84336609b4bfd57ab755c3c45785b0eb22ddd7ae06cd24923ed27adc39c8a8cc`.
The offline verifier returns `PASS`, eight rounds, and `INCONCLUSIVE`. It also
checks local PTX against the compiler capture in the raw result.

The first command returned a bundle-replay error **after** the GPU worker and
compiler capture succeeded. The report had rendered a Python dictionary in
insertion order, while JSON replay sorted its keys. The v2 report renderer
now sorts software-version keys. The saved report and manifest were repaired
offline. An incidental Python bytecode cache was removed from the new bundle.
Raw GPU output, timing rows, correctness records, PTX, and source snapshots
stayed byte-for-byte unchanged. No second GPU benchmark was run.

## Validation and remaining friction

CPU tests cover spec bounds, source existence and hashes, import and callable
failures, explicit opt-in, output checks, two local sources, artifact
generation, secret-safe failure records, registered compatibility, and
offline replay. The old MI-Workflow-2 bundle still verifies.

Developers must still supply the narrow `matmul_into` wrapper and trust the
local Python code. Returning `None` is valid, but it leaves local compiler
metadata and PTX unavailable unless the entrypoint exposes a Triton compiled
launch result. The recurring CUDA-event versus host-wall discrepancy remains
unexplained. Decoded SASS requires separate tooling. These limits do not
prevent a correct local kernel from being benchmarked, but they constrain
performance conclusions.
