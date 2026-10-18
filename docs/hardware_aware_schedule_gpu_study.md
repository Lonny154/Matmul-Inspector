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

The retained run used implementation commit `c469232` on an NVIDIA GeForce
RTX 4060 Ti (SM 8.9, 16 GiB), driver 616.92, Python 3.12.13, PyTorch
2.11.0+cu128, CUDA runtime 12.8, and Triton 3.6.0. It recorded 30,000 raw
CUDA-event samples: 15 rounds × 100 operations × five implementations × four
sizes. The retained CSV had only its line endings normalized to LF after the
run. No timing values changed. All 80 candidate-versus-PyTorch checks and 20
direct pair checks passed. The largest absolute error across those checks was
0.125 at N=1024.
The two rank-8 outputs were bitwise equal for every seeded case.

The table shows complete-operation latency in milliseconds. Each entry is the
median of 15 round medians; parentheses give the coefficient of variation
across those round medians. All four Triton paths use preallocated output and
product buffers. PyTorch matmul returns an allocated result, so its anchor is
not an allocator-matched kernel-only comparison.

| N | cuBLAS/PyTorch | Conventional Triton | Strassen Triton | Rank-8 stored | Rank-8 recomputed |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0333 (11.4%) | 0.1514 (5.2%) | 0.1367 (8.4%) | 0.1557 (9.9%) | 0.1479 (8.1%) |
| 256 | 0.0547 (24.9%) | 0.2171 (42.6%) | 0.1960 (38.1%) | 0.2300 (42.4%) | 0.2106 (43.2%) |
| 512 | 0.0558 (22.1%) | 0.2356 (43.0%) | 0.2277 (44.8%) | 0.2757 (42.8%) | 0.2342 (39.4%) |
| 1024 | 0.1218 (11.9%) | 0.5643 (11.9%) | 0.5332 (8.9%) | 0.6386 (14.3%) | 0.5757 (9.4%) |

The matched ratio is **stored/recomputed**, so greater than one favors
recomputation. Its round-level median is 1.101 at 128, 1.097 at 256, 1.100 at
512, and 1.082 at 1024. The deterministic round-bootstrap 95% intervals are
1.056–1.132, 1.047–1.207, 1.040–1.164, and 0.991–1.104, respectively. The
first three exclude parity in this one session; 1024 does not. Individual
rounds still reverse the direction, and round CV reaches about 40–45% at
256/512. Clocks and power also varied in the retained round snapshots.
These intervals describe within-session round variability, not uncertainty
over other devices or sessions. A second-session replication is still needed
before treating the size dependence as stable.

The conventional FP16/preallocated control is in the same broad sub-millisecond
range as earlier project GPU studies. It is not directly comparable to their
FP32 or allocating paths. Strassen is correct and has a lower median than
conventional Triton in this run, despite twice the modeled temporary HBM at
N=128 (262,144 versus 131,072 B). Given the 256/512 dispersion and prior
timing diagnostics, this is not evidence that its larger modeled footprint is
beneficial or that Strassen has a stable hardware advantage. cuBLAS is faster
than every Triton path here; that does not invalidate the model test.

## Compiler, launch, and profiler comparison

Actual complete-operation launches are constant over size: conventional 9,
Strassen 8, rank-8 stored 10, and rank-8 recomputed 9. The PyTorch anchor has
one launch at 128/256/512 and two at 1024; the latter includes a device memset.
The static model says 8, 7, 8, and 8 logical launch groups, respectively. A
logical group is not a physical launch, and the modeled rank-8 pair has the
same group count despite the one-launch GPU difference. The extra preparation
kernel is part of the measured stored operation. All profiled Triton kernel
names matched the expected source sequence.

The representative first shared-form GEMM at N=1024 has different TTIR,
TTGIR, PTX, cubin, and SASS hashes across the pair. The stored variant uses
40 registers/thread and 8,192 B static shared memory. The recomputed variant
also uses 40 registers/thread but 10,240 B static shared memory. The stored
path additionally has a 13-register preparation kernel. In its selected
first-GEMM SASS, the stored variant has 4 static `FADD` and 6 `LDGSTS`
mnemonics; the recomputed variant has 12 and 9. This supports a real machine
code difference, not a compiler-collapsed source change. These are static
mnemonic counts for one specialization, not dynamic operations or DRAM bytes.
The cubin hashes are retained but binary cubins are not committed.

Nsight Compute 2026.2.1.0 selectively captured the preparation kernel and
two shared-form GEMMs at N=512 in two separate-process runs per variant. The
stored subset had 839,296 and 875,392 reported DRAM-read bytes; recomputed
had 803,840 and 822,272. In contrast, L2-sector counts were 164,129 and
165,228 for stored versus 215,699 and 216,284 for recomputed. Thus the
recomputed subset showed slightly fewer reported DRAM reads but materially
more L2 transactions. The profile excludes six unchanged generic GEMMs and
reconstruction. These are first-execution captures in separate processes, with
no independent profiler warmup. Replay and cache state can affect counters. Reported DRAM
writes were zero or near zero, so they do not reliably quantify the removed
form store. The profile cannot establish that DRAM reduction caused the
CUDA-event latency gap. Exact counters, metric names, and capture limitations
are in `profiler_summary.json`.

The static rank-8 pair trades 393,216 for 368,640 modeled temporary HBM
bytes at N=128, with +4,096 abstract operations and +16,384 source-read
bytes. Its latency direction agrees with the static traffic direction at the
first three sizes. The absolute byte figures are not measured traffic. The
actual lowering also removes a launch and changes tiled reads and shared
memory, so the schedule model is only partially descriptive. Static ON_CHIP
live bytes are full-block abstract lifetimes (32,768 B for these N=128
points), not Triton shared-memory allocations; they should not be equated
with 8–10 KiB compiler metadata. Only four implementable schedules were
timed, and three use different algebraic decompositions. A rank correlation
across them would imply more independent schedule evidence than exists, so it
is intentionally omitted.

**Primary decision: `SCHEDULE_SIGNAL_CONFIRMED`, with an implementation-limit
qualification.** The matched tradeoff changes compiled machine code and
shows a favorable paired timing signal at 128/256/512 in this session. The
static temporary-traffic direction agrees there, but extra launch removal
and L2 behavior prevent a traffic-causality claim. The lower-capacity frontier
point remains unimplemented, and 1024 plus cross-session repeatability remain
inconclusive.
