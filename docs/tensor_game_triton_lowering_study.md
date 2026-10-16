# Tensor Game to Triton lowering study

## Research question

Can the existing algebraic cost descriptors predict meaningful GPU execution
differences after valid Tensor Game decompositions are lowered to Triton?

This milestone establishes and validates the lowering and measurement path on a
local NVIDIA GeForce RTX 4060 Ti (compute capability 8.9).

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

All 32 benchmark correctness rows passed. Across the 16 cases per algorithm,
conventional reached maximum absolute error `1.373291e-4`; its largest per-case
mean absolute error was `1.039860e-5`. Strassen reached maximum absolute error
`2.403259e-4`; its largest per-case mean absolute error was `1.980958e-5`.
Maximum elementwise relative error was large near reference values close to
zero, so absolute error and the configured elementwise tolerance determine the
pass result. The dedicated GPU test file also passed all 15 tests.

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

The additional phase and compiler summaries are reproducible with:

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/profile_tensor_game_triton_phases.py \
  --output artifacts/tensor-game/triton-lowering-v1/phases.json \
  --sizes 128 256 512 1024 --warmups 10 --iterations 30 --runs 5

python scripts/analyze_tensor_game_triton_compiler.py \
  --artifact-root artifacts/tensor-game/triton-lowering-v1/compiler-artifacts \
  --sass-dir artifacts/tensor-game/triton-lowering-v1/sass \
  --output artifacts/tensor-game/triton-lowering-v1/compiler_summary_v2.json
```

## Measured results

The environment used Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
Triton 3.6.0, NVIDIA driver 616.92, and an RTX 4060 Ti with 16,380 MiB. PyTorch
TF32 was disabled and the Triton dot used IEEE FP32 inputs. Values below are the
median of five independent run medians. The range is the minimum and maximum of
those five medians.

| Size | PyTorch/cuBLAS | Conventional | Strassen | Conventional / PyTorch | Strassen / conventional |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0408 ms (0.0391–0.0658) | 0.6119 ms (0.5924–1.3982) | 0.6928 ms (0.6768–1.5952) | 14.99× slower | 13.2% slower |
| 256 | 0.0330 ms (0.0246–0.0428) | 0.6154 ms (0.5756–0.6532) | 0.6736 ms (0.6369–0.7178) | 18.63× slower | 9.5% slower |
| 512 | 0.0638 ms (0.0438–0.0709) | 0.6282 ms (0.5978–0.9508) | 0.7218 ms (0.6650–0.9871) | 9.85× slower | 14.9% slower |
| 1024 | 0.2257 ms (0.2222–0.2626) | 1.2691 ms (0.6713–1.4816) | 1.4047 ms (0.7138–1.7419) | 5.62× slower | 10.7% slower |

These are complete-operation CUDA-event latencies. They include device-side
preparation, seven or eight Triton GEMMs, reconstruction, and output copies.
They exclude Python host-call latency, JIT compilation, and one-time setup.
Allocator work outside the recorded event interval may not be fully represented.
The nonmonotonic small-size PyTorch timings and wide ranges in several lowering
runs show that this experiment should not support fine-grained percentage
claims beyond the large observed gaps.

A separate diagnostic measured phases independently:

| Size | Algorithm | Preparation | GEMM-only rank batch | Reconstruction |
|---:|---|---:|---:|---:|
| 128 | Conventional | 0.3267 ms | 0.2421 ms | 0.1730 ms |
| 128 | Strassen | 0.2774 ms | 0.1300 ms | 0.1631 ms |
| 256 | Conventional | 0.2354 ms | 0.1464 ms | 0.1419 ms |
| 256 | Strassen | 0.2755 ms | 0.1343 ms | 0.1693 ms |
| 512 | Conventional | 0.2334 ms | 0.1459 ms | 0.1365 ms |
| 512 | Strassen | 0.4444 ms | 0.1705 ms | 0.2324 ms |
| 1024 | Conventional | 0.3579 ms | 0.3593 ms | 0.1769 ms |
| 1024 | Strassen | 0.4540 ms | 0.3145 ms | 0.2344 ms |

The phase medians come from separate invocations with precomputed dependencies.
They must not be summed to reconstruct the complete latency. Raw run medians also
show high variation for several phase/size combinations. At 1024, the expected
seven-versus-eight GEMM advantage is visible in isolation, but higher preparation
and reconstruction medians accompany a slower complete Strassen operation.

## Static cost versus measured runtime

The static model predicts that Strassen trades one fewer block GEMM and one fewer
peak block temporary for six more total matrix additions. The measured 1024
phase medians agree with that direction: lower isolated GEMM time and higher
preparation and reconstruction time. At smaller sizes, launch, allocation, and
measurement variability are large enough that the phase ordering is not stable.
The runtime also
introduces separate launches or framework operations for every combination,
GEMM, reconstruction update, and copy. The algebraic descriptor does not model
launch overhead, allocation behavior, bytes moved, cache reuse, or the Triton
kernel's tiling. Rank reduction did not win in the complete measured path.

## Compiler observations

Four matrix-size specializations produced TTIR, TTGIR, PTX 8.7 targeting `sm_89`,
CUBIN, metadata, and saved SASS. Their hashes are recorded in `results.json` and
`compiler_summary_v2.json`. Triton metadata reports four warps, three stages,
8,192 bytes of dynamic shared memory, and no global scratch. `cuobjdump` reports
48 registers per thread, zero local memory, zero stack, and zero static shared
memory for every specialization. The dynamic and static shared-memory fields are
different evidence and are not contradictory.

Each saved PTX contains 128 static `fma.rn.f32` occurrences and no MMA/WGMMA
text. Each final SASS disassembly contains 128 static `FFMA` occurrences and no
HMMA/IMMA/BMMA text. These are static occurrences, not executed instruction
counts. The PTX is virtual ISA; the separately saved `nvdisasm` output is the
final-machine-code evidence used for the SASS statement.

## Limitations and next experiment

This is a nonrecursive 2×2 block lowering. It materializes every combination and
product, uses a fixed untuned 32×32×16 FP32 GEMM tile, and does not fuse additions
or reuse output storage. The fixed IEEE FP32 kernel lowers to scalar FFMA rather
than Tensor Core MMA and was intentionally not tuned. Phase measurements are
isolated diagnostics and do not establish an additive causal breakdown. CUDA
events exclude host overhead, and no Nsight trace was collected to count every
PyTorch preparation/reconstruction launch or memory transaction.

The next experiment should capture one Nsight Systems trace at 1024 to verify
launch counts, allocation activity, and the device timeline. Then test a
preallocated implementation with identical arithmetic and fixed kernel geometry.
That isolates allocation and materialization overhead before considering fusion,
tile tuning, TF32, or a hardware-aware search score.
