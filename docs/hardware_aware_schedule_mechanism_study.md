# HA-Schedule-4: recompute-vs-store mechanism isolation

HA-Schedule-3 found a rank-8 recomputation timing signal on the RTX 4060 Ti.
It did not identify the cause. The stored path has 10 launches and the
recomputed path has 9. Their consumer machine code and memory behavior also
differ. This milestone keeps the decomposition and production Triton kernels
fixed, then measures these pieces separately. It does not extend CP-SAT search
or implement HA-Schedule-5.

## Exact production topology

Both paths use the exact frozen rank-8 decomposition and FP16 inputs and
output with FP32 product accumulation. The common code writes eight FP32
rank-product buffers and reconstructs the FP16 output.

| Order | Stored path | Recomputed path |
|---:|---|---|
| 0 | `_materialize_u_form` for `A_FORM00 = -A00 - A10` | — |
| 1–2 / 0–1 | two `_shared_form_gemm` consumers loading prepared FP16 form | two `_shared_form_gemm` consumers forming FP16-rounded tiles internally |
| next six | six `_fused_input_gemm_kernel` launches | same six, with identical factor coefficients |
| last | one `_fused_reconstruction_kernel` | same reconstruction |

Only the preparation launch is present in the stored path. The first two
consumer GEMMs use the same source grid, tile shape, warps, stages, factors,
FP32 dot, and FP32 product layout. The `use_prepared` compile-time branch
changes the input-form computation and memory loads. Compiler scheduling and
shared-memory allocation can therefore differ. The remaining seven launches
are checked by name and cubin hash for both paths at every tested size.

The HA-Schedule-3 artifact retains TTIR, TTGIR, PTX, SASS, cubin hashes,
registers, shared-memory metadata, and two first-execution Nsight captures at
N=512. The earlier selective metrics were `dram__bytes_read.sum`,
`dram__bytes_write.sum`, and `lts__t_sectors.sum`. They did not measure a
complete-operation mechanism. CUDA Graph support already exists in the
repository's FC1 timing diagnostic; this study checks capture again for these
specific rank-8 paths.

## Stage and session protocol

The standalone command launches two fresh Python subprocesses. Session A
collects complete-operation timings, stage probes, graph replay, and launch
diagnostics. Session B independently repeats the complete-operation pair.
Each session has its own process ID and UTC start/end timestamps. The runner
does not pool sessions before comparing their paired round ratios.

For every size, five fixed input seeds gate both production paths against
PyTorch matmul with `rtol=0.005`, `atol=0.05`. The staged path and each of its
first two product buffers must match the unchanged production path. The
stored/recomputed outputs are compared directly. Graph replay and the dummy
diagnostic are separately correctness-gated. Failed diagnostics cannot enter
the mechanism conclusion.

Stage probes call the same compiled Triton kernels with the same arguments as
the production path. They measure preparation, each consumer GEMM, the two
consumers together, the six common GEMMs together, and reconstruction. The
stored form is prepared before the consumer-only paired timing block. It is
not prepared inside that block. All buffers are preallocated. CUDA events
bracket the actual stages. The stage pair alternates order by round and uses
the same original matrices. Raw operations and round medians are retained.
The complete-operation pair is measured separately with the original
`execute_selected` entry point. Stage medians are not naively added to infer
complete latency. Event intervals can contain host-enqueue gaps between
kernels, and device state can vary across blocks.

The default retained protocol uses 20 warmups, 100 operations per round, and
15 rounds for each complete and stage benchmark. The graph and dummy controls
use the same counts. The stage summary reports median, mean, standard
deviation, min, max, CV, round count, and operations per round. Paired
bootstrap intervals resample round medians, not individual operations.

The CUDA Graph control captures each complete production path separately
into preallocated buffers and checks replay output. Within each graph
diagnostic round, eager and graph pair blocks are interleaved. Graph replay
still executes the device kernels; it does not eliminate device work or all
launch overhead. It is a secondary diagnostic, not a replacement primary
latency number.

The optional equal-launch control uses a separate one-word scratch kernel
before the recomputed production path. It never touches matrix inputs,
products, or output. This `DIAGNOSTIC_ONLY` path is checked numerically and
must show 10 launches. Its tiny standalone duration gives launch-scale
context. It is not a production candidate or a constant to subtract from
the complete-operation result. Kernel launch overhead depends on context.

## Compiler and profiler evidence

The study captures both consumer specializations, the preparation kernel,
and one copy of each common kernel at the largest size. It retains bounded
TTIR, TTGIR, PTX, and SASS text, compiler register and static shared-memory
metadata, and cubin hashes. It does not commit cubin binaries. The comparison
normalizes only SASS instruction addresses; register and operand differences
remain visible. It records a bounded diff and selected static mnemonic
counts. PTX is not SASS. Static counts are not dynamic execution counts.

When Nsight Compute is available, the runner queries metrics, then captures
only the preparation and two consumer kernels at N=512 in two separate
processes per variant. It requests kernel duration, DRAM reads/writes, L2
sectors, and active-warp percentage if available. Nsight replays kernels and
its durations are not CUDA-event benchmark times. DRAM and L2 counters are
different memory hierarchy observations. First-execution and cache effects
limit causal interpretation. The profiler is not run over the candidate set.

HA-Schedule-2 temporary HBM bytes remain an abstract model. They are not
measured DRAM bytes. Caches, compiler fusion, tiled source reads, extra
launches, register allocation, shared memory, and execution order can all
change hardware behavior. A specific mechanism is claimed only if the
controls isolate it. One GPU and two process sessions do not establish a
universal architecture result.

## Reproduce

Use an RTX 4060 Ti environment with CUDA-enabled PyTorch and Triton. The
command refuses an existing output directory. A smoke run is a functionality
check, not retained performance evidence.

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_mechanism_study.py \
  --output /tmp/ha-schedule4-smoke --smoke

PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_mechanism_study.py \
  --output artifacts/tensor-game/hardware-aware-schedule-mechanism-v1
```

The retained directory contains config, provenance, hardware/software,
topology, correctness, raw session A/B and stage timings, graph and launch
diagnostics, compiler and profiler comparisons, static/measurement joins,
aggregate results, and a SHA-256 manifest. The GPU runtime remains optional
for CPU-only package tests.

## Results and decision

To be completed only after the two retained fresh-process sessions and
diagnostics have run from a committed implementation.
