# HA-Schedule-3: selected GPU schedule test

HA-Schedule-2 found static execution-DAG schedules under an abstract ON_CHIP
capacity. This study asks whether a few of those choices survive a real Triton
lowering and change latency on an RTX 4060 Ti. It is not a rank search or a
Triton autotuning campaign. The retained results are in
`artifacts/tensor-game/hardware-aware-schedule-gpu-v1`.

## Selected frontier points

Selection is fixed by algorithm, capacity in input-block units, recomputation
budget, and the `LEX_TEMP_RECOMPUTE_LAUNCH` objective. The runner verifies the
retained HA-Schedule-2 hash manifest. It rebuilds each schedule, validates it,
and independently evaluates its metrics before touching CUDA. The exact run IDs
and per-size schedule hashes are saved in `selected_schedules.json`.

| Candidate | HA2 run | Capacity | Budget | N=128 modeled temporary HBM | Classification |
|---|---|---:|---:|---:|---|
| Conventional combined-fusion reference | R0013 | 4 | 0 | 131,072 B | Approximately implementable |
| Strassen combined-fusion reference | R0037 | 4 | 0 | 262,144 B | Approximately implementable |
| Frozen rank-8 stored form | R0061 | 4 | 0 | 393,216 B | Approximately implementable |
| Frozen rank-8 recomputed form | R0063 | 4 | 1 | 368,640 B | Approximately implementable |
| Frozen rank-8 lower-capacity point | R0053 | 2 | 0 | 524,288 B | Not implementable in current lowering |

The last point has a different capacity/traffic hypothesis. Its selective
intermediate placement needs kernels the current fused path does not have, so
the runner retains its static mapping but excludes it from timing. No selected
point is claimed to be directly implementable.

## Schedule-to-Triton mapping

Conventional and Strassen reuse the existing fused-input GEMMs and fused-output
reconstruction. The project candidate uses the exact frozen rank-8 factors,
the same six generic fused-input GEMMs for its other products, and the same
fused reconstruction. All paths write rank products to FP32 global-memory
buffers and reconstruct into FP16. The existing fused path therefore does not
implement the abstract within-group output accumulation or local lifetimes.
Some singleton forms are computed in a GEMM load path even when the static
schedule materializes them. The mapping is an executable approximation, not a
claim that the CP-SAT schedule has been compiled exactly.

The matched rank-8 pair changes only the treatment of the shared `A_FORM00`
input form (`-A00 - A10`) in its first two products. The stored path launches one preparation
kernel that forms the FP16 matrix, stores it to global memory, and has two
GEMMs load it. The recompute path has no preparation buffer or kernel. Each of
the two GEMMs forms the same input tile from the original FP16 matrix and
rounds it to FP16 before the FP32 dot product. This is real recomputation in
the GEMM kernel, not duplicate arithmetic followed by the same store. The
first-two-term factors are checked before execution. Tile reuse and grid
geometry mean the actual number of form evaluations is not the abstract
`+4,096` operations at N=128. The compiler and caches may further change
actual traffic. The complete operation includes preparation where present,
all products, and reconstruction.

`lowering_map.json` records classification, static schedule hashes and metrics,
source kernel sequences, and deviations. `launch_accounting.json` records the
actual profiled kernel sequence and count. The abstract launch groups are not
the GPU launch count.

## Correctness and timing

All timed candidates use FP16 inputs, FP32 product accumulation, and FP16
output. Five fixed input seeds per size are checked against PyTorch matmul
with `rtol=0.005`, `atol=0.05`. Max and mean absolute error and every pass/fail
are retained. The two rank-8 outputs also have a direct agreement check.
Failures remain in the artifact and exclude affected candidates from timing.
The tolerance is fixed before the retained run; it is not widened to rescue a
candidate.

The runner benchmarks full matrix dimensions 128, 256, 512, and 1024. It uses
20 warmups, 100 measured complete operations per round, and 15 rounds by
default. Order is deterministic and randomized by round. The rank-8 pair
remains adjacent and reverses order every other round. CUDA events bracket one
complete invocation on the default stream, with a synchronized end event per
sample. This is elapsed GPU timeline time, not Python wall time. It can include
device idle time between launches if host dispatch cannot keep the stream
busy. The summary uses round medians as replicates and retains mean, standard
deviation, min, max, coefficient of variation, and paired ratios. The paired
bootstrap interval resamples round ratios, not individual operations. cuBLAS
is a separate external anchor. The conventional Triton path is the internal
baseline. Per-launch durations in `launch_accounting.json` come from the
PyTorch profiler, not the CUDA-event benchmark; they must not be summed as
replacement complete-operation times.

Use the optional CUDA-enabled PyTorch/Triton environment:

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_gpu_study.py \
  --output /tmp/ha-schedule3-smoke --smoke

PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_gpu_study.py \
  --output artifacts/tensor-game/hardware-aware-schedule-gpu-v1
```

Each command refuses an existing output directory. `--smoke` is a short
functionality check, not retained performance evidence. CPU-only selection
and artifact replay tests do not import PyTorch or Triton.

## Evidence interpretation

The artifact includes raw CUDA-event samples, round telemetry, launch
accounting, static/measurement joins, hardware and software provenance, and
Triton TTIR/TTGIR/PTX plus bounded SASS text for the rank-8 pair. The cubin is
hashed but not retained as a large binary. PTX is not SASS. Static instruction
counts are not dynamic execution counts. HA-Schedule-1/2 HBM numbers are
abstract modeled traffic. Actual DRAM traffic can differ because of caches,
fusion, register allocation, shared memory, and execution order. A schedule
can reduce modeled bytes while becoming slower. Failing to beat cuBLAS does
not invalidate this schedule-model test.

## Retained results and decision

To be added after the retained run from the implementation commit. A smoke
run or short validation pass is not sufficient for the decision gate.
