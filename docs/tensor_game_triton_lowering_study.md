# Tensor Game to Triton lowering study

## Research question

Can the existing algebraic cost descriptors predict meaningful GPU execution
differences after valid Tensor Game decompositions are lowered to Triton?

This milestone establishes the lowering and measurement path. The current
execution environment did not expose PyTorch, Triton, or a usable GPU, so no GPU
latency, numerical-error, PTX, SASS, or resource result is claimed here.

## Lowering architecture

`create_lowering()` accepts an algorithm name and a sequence of
`RankOneAction(u, v, w)` values. It validates every action and requires their
exact integer sum to equal the 2×2 matrix-multiplication tensor. Invalid
decompositions are rejected before the optional runtime is imported.

The shared runtime performs this schedule:

1. Split contiguous square A and B tensors into four row-major block views.
2. Materialize each `u` and `v` linear combination in FP32.
3. Run one Triton block GEMM for every rank term.
4. Materialize four FP32 output-block accumulators from `w`.
5. Copy the blocks into one contiguous output tensor.

The same code handles the conventional rank-8 and Strassen rank-7 references.
There is no algorithm-specific arithmetic in the runtime.

## Decomposition convention

The repository stores one rank term as vectors `(u, v, w)`. For a 2×2 block
grid, vector indices are row-major: `0=(0,0)`, `1=(0,1)`, `2=(1,0)`, and
`3=(1,1)`. The runtime forms one A combination from `u`, one B combination from
`v`, multiplies them, then adds that product to every output block selected by
`w`. This is the existing `RankOneAction` orientation, not a transposed `U/V/W`
matrix convention.

Inputs must be equally sized, square, contiguous CUDA FP32 tensors with an even
dimension. Output and accumulation are FP32. The Triton dot explicitly requests
`input_precision="ieee"`; PyTorch TF32 is disabled by the benchmark.

## Static lowering characteristics

| Algorithm | Rank / block GEMMs | Preparation additions | Reconstruction additions | Recorded materializations | Peak live block temporaries |
|---|---:|---:|---:|---:|---:|
| Conventional | 8 | 0 | 4 | 29 | 13 |
| Strassen | 7 | 10 | 8 | 26 | 12 |

The materialization model counts two inputs and one product per rank term over
the execution, four output accumulators, and the final output. Sequential input
preparation means input combinations are not all live together. These counts do
not describe allocator internals, memory transactions, registers, or spills.

The executable schedule initializes combination and reconstruction buffers to
zero, then applies one in-place update per nonzero coefficient. Conventional
therefore emits 16 preparation updates and 8 reconstruction updates. Strassen
emits 24 and 12. These update counts differ intentionally from the algebraic
addition estimates above, which do not count assigning the first term.

## Correctness methodology

GPU tests cover conventional and Strassen at sizes 2, 32, and 128 with seeded
random inputs. The benchmark additionally checks zero, identity, seeded random,
and negative-valued inputs at every requested size. Results are compared with
PyTorch FP32 matrix multiplication using `rtol=5e-3` and `atol=5e-3`. Maximum
absolute, mean absolute, and maximum elementwise relative errors are saved.
Bitwise agreement is not required because Strassen changes operation order.

CPU validation currently passes. GPU correctness is pending because this
environment lacks the optional dependencies and CUDA access.

## Benchmark configuration

The benchmark defaults to square sizes 128, 256, 512, and 1024. It compares:

- PyTorch/cuBLAS FP32 GEMM.
- Conventional rank-8 lowering.
- Strassen rank-7 lowering.

Each implementation receives identical tensors. Ten warmups precede five
independent runs of 30 measurements. Implementation order is shuffled per run
with seed `20261016`. CUDA events time the complete queued device operation,
including preparation, GEMMs, reconstruction, and device-side copies. Python
host-call time and host allocation overhead are not represented by CUDA-event
latency. Each invocation allocates fresh output and intermediate tensors, so
device allocation effects that fall between the recorded events may not be
fully represented.

Run on the target RTX 4060 Ti environment with:

```bash
PYTHONPATH=src python scripts/benchmark_tensor_game_triton_lowering.py \
  --output artifacts/tensor-game/triton-lowering-v1/results.json \
  --sizes 128 256 512 1024 --warmups 10 --iterations 30 --runs 5
```

The output path must not already exist. The JSON preserves versions, device
identity, compute capability, coefficient manifests, correctness metrics, raw
timings, distributions, and speedups. The script uses a dedicated Triton cache
and copies available TTIR, TTGIR, PTX, CUBIN, SASS, and metadata files into a
compiler-artifacts directory with SHA-256 digests.

## Measured results

No GPU results were produced in this environment. PyTorch and Triton imports
both failed, and the GPU-management probe reported that device access was
blocked. Therefore Experiments A–C are implemented but not executed here.

The CPU-side evidence is:

- Both reference decompositions pass exact reconstruction through the shared
  validation interface.
- Deterministic manifests preserve the expected rank and coefficient vectors.
- Invalid decompositions fail before GPU runtime loading.
- Existing Tensor Game search and hardware-cost code is unchanged by execution.

## Static cost versus measured runtime

The static model predicts that Strassen trades one fewer block GEMM and one fewer
peak block temporary for six more total matrix additions. The runtime also
introduces separate launches or framework operations for every combination,
GEMM, reconstruction update, and copy. The algebraic descriptor does not model
launch overhead, allocation behavior, bytes moved, cache reuse, or the Triton
kernel's tiling. Whether rank reduction wins is therefore an open measurement,
not an optimization claim.

## Compiler observations

There are no compiled artifacts from this environment. When run on CUDA, the
script preserves available Triton cache intermediates. PTX must be reported as
virtual ISA, not final machine code. Register allocation, shared memory, and
final SASS require compiler metadata or disassembly evidence. They must not be
inferred from Python or Triton source.

## Limitations and next experiment

This is a nonrecursive 2×2 block lowering. It materializes every combination and
product, uses a fixed untuned 32×32×16 FP32 GEMM tile, and does not fuse additions
or reuse output storage. The benchmark does not yet separate preparation, GEMM,
and reconstruction timings. GPU correctness and all performance claims remain
pending.

The next milestone is to run the frozen command on the RTX 4060 Ti, audit all
correctness rows, and inspect the captured compiler artifacts. If correctness
passes, add event-timed phase instrumentation without changing the complete-path
benchmark. Use those measurements to test whether launch and materialization
cost explain any gap between the rank-7 and rank-8 implementations before trying
fusion or a hardware-aware search score.
