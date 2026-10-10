# Tridiagonal CUDA Interview Review

Rapid review for GPU kernel, performance, ML systems, compiler/kernel, and
accelerator interviews. Measurements are hardware-specific results from the
project's RTX 4060 Ti experiments, not universal claims. For complete derivations,
glossary, exercises, and profiler detail, see
[`tridiagonal_study_guide.md`](tridiagonal_study_guide.md).

## 1. The project in 60 seconds

I compared two ways to solve batches of tridiagonal systems. CPU Thomas has
`O(N)` work but a sequential dependency chain. Parallel Cyclic Reduction (PCR)
does more arithmetic, yet exposes each equation within `O(log N)` stages, which
fits GPU execution. The initial CUDA path launched a complete solve separately
for every system. Flattening `B*N` equations into true batches reduced stage
launches from roughly `B*log2(N)` to `log2(N)`. I then separated allocation,
transfer, kernel-only, and device-resident costs with a reusable RAII workspace.

Nsight Compute showed large early PCR stages near 90% DRAM utilization with
long-scoreboard/L1TEX waits. A shared-memory-per-stage experiment was correct but
did not consistently improve performance. Fusing offsets 1, 2, and 4 removed two
launches and intermediate global-memory round trips, improving the documented
`4096x512` solve by `1.104x`. Because fusion was workload-dependent, I measured a
100-point grid and added a transparent adaptive rule. It improved aggregate
measured runtime about **5.5% over always-global** with zero false-fusion cases
among decisive grid points.

*See full guide: Sections 1, 20, and 30.*

## 2. The performance story

| Step | Problem or bottleneck | Change | Result or lesson |
|---|---|---|---|
| CPU Thomas | Serial `O(N)` dependency chain | Establish low-work baseline | Excellent CPU reference; little intra-system parallelism |
| CPU PCR | Need a parallel algorithm/reference | Offsets `1,2,4,...` | More work, `O(log N)` stage span |
| CUDA PCR | Map independent equations | One thread/equation | Correct GPU baseline, but launch structure mattered |
| Serial host loop | About `B*log2(N)` stage launches | Preserve as explicit baseline | Host scheduling dominated batched use |
| True batching | Systems launched separately | Flatten `B*N`; one launch/stage | Exposed system and equation parallelism |
| Reusable workspace | Repeated `cudaMalloc/cudaFree` | RAII device buffers reused | Allocation removed from steady-state path |
| Device resident | H2D/D2H obscured kernel behavior | Immutable device snapshots; separate D2D reset | Isolated PCR solve for accelerator pipelines |
| Nsight profiling | Need source of device cost | NVTX names each offset | Early stages were DRAM/long-scoreboard heavy |
| Shared-memory hybrid | Hypothesis: cache neighbors | Shared stages through offset 256 | No consistent win; target was ~2.1% slower |
| Fused early stages | Inter-stage traffic and launches remained | Fuse offsets 1,2,4 | Large workloads improved; small ones were noisy/underfilled |
| Adaptive dispatch | No universal kernel winner | Measured workload threshold | ~5.5% aggregate gain versus always-global |

*See full guide: Sections 6–11 and 16–23.*

## 3. Concepts I should know cold

| Concept | Definition and project relevance |
|---|---|
| Kernel | GPU function launched over many threads; each PCR stage is a kernel except fused 1/2/4. |
| Grid | All blocks in one launch; grid size exposed underfilled versus saturated workloads. |
| Block / CTA | Cooperating thread group scheduled on one SM; PCR uses 256 threads/block. |
| Thread | One logical CUDA worker; maps to one `(system,equation)`. |
| Warp | 32 threads scheduled together on NVIDIA GPUs. |
| SM | Streaming Multiprocessor executing resident blocks/warps. |
| SIMT | Scalar-looking threads execute in hardware warps under lane masks. |
| Global memory | Large off-chip device DRAM storing coefficient and result arrays. |
| Shared memory | Explicit on-chip block storage; used for halos and fused intermediates. |
| L1/L2 cache | Hardware-managed on-chip caches; existing reuse can reduce the value of manual staging. |
| Memory bandwidth | Bytes transferred per second; early large global stages approached DRAM limits. |
| Memory-bound | Data movement limits progress; it suggests investigation, not an automatic fix. |
| Compute-bound | Arithmetic pipelines limit progress; large fusion shifted toward compute. |
| Occupancy | Active resident warps divided by maximum; a latency-hiding indicator, not speed. |
| Waves/SM | Grid scheduling depth relative to SM capacity; low values indicate underfill. |
| Launch overhead | Fixed cost to submit/schedule a kernel; batching and fusion amortize it. |
| Synchronization | Coordination before dependent work consumes data. |
| `__syncthreads()` | Block-wide barrier and shared-memory visibility point. |
| Scoreboard stall | Warp waits for an operand dependency; long scoreboard often implicates L1TEX/global loads. |
| MIO stall | Wait associated with memory-I/O paths, often shared-memory dependencies here. |
| CTA-barrier stall | Warp waits for sibling warps at a block barrier. |
| Batching | Processing independent systems together to increase work per launch. |
| Device resident | Inputs already on GPU and output remains there across the measured solve. |
| Kernel fusion | Combine operations in one launch so intermediates remain local. |
| Work | Total operations: Thomas `O(N)`, PCR `O(N log N)`. |
| Span / critical path | Longest dependency chain: Thomas `O(N)`, PCR `O(log N)`. |
| Adaptive dispatch | Runtime selection among preserved kernels using workload shape. |
| Amortization | Spread fixed launch/allocation cost across more useful work. |
| Bottleneck migration | Optimization shifts the limiting resource, as DRAM waits became MIO/barrier/compute pressure. |

*See full guide: Section 25, Vocabulary Glossary.*

## 4. Thomas versus PCR

| Property | Thomas | PCR |
|---|---|---|
| Total work | `O(N)` | `O(N log N)` |
| Critical path | `O(N)` | `O(log N)` stages |
| Dependencies | Forward elimination + back substitution chains | Equations independent within each stage |
| Best fit | CPU latency, small/host-resident work | Accelerator throughput with sufficient parallel work |
| Cost | Low arithmetic and storage traffic | More arithmetic, stage traffic, and launches |

**Interview lesson:** More arithmetic can still be faster when it exposes enough
parallelism. Big-O work alone does not describe parallel hardware performance.

*See full guide: Sections 3–5.*

## 5. CUDA mapping

```cpp
global_index = blockIdx.x * blockDim.x + threadIdx.x;
system       = global_index / N;
equation     = global_index % N;
```

All `B*N` equations occupy contiguous arrays. Neighbor addresses are relative to
`system*N`, and within-system boundary checks prevent cross-system reads.

- Serial host loop: about `B*ceil(log2(N))` stage launches plus `B` solves.
- True batch: `ceil(log2(N))` stage launches plus one solve.

Flattening matters because it exposes system-level parallelism and gives each
launch enough work to use the GPU.

*See full guide: Section 6.*

## 6. Profiling cheat sheet

| Metric | Meaning | High/low may imply |
|---|---|---|
| DRAM throughput | Fraction of off-chip bandwidth utilized | High: memory bandwidth pressure; low does not prove memory is irrelevant |
| Compute (SM) throughput | Compute-pipeline utilization | High: compute may limit; compare duration and instruction mix |
| L1/TEX throughput | Activity through L1/texture/load-store path | Includes normal global traffic, not only textures |
| L1 hit rate | Requests served near the SM | High can indicate locality; interpret with request volume |
| L2 hit rate | Requests served by device-wide L2 | High may reduce DRAM traffic |
| Achieved occupancy | Measured active warps / maximum | Low may limit latency hiding or simply reflect a short/small grid |
| Active warps | Resident unfinished warps per SM | More can hide latency until another limit dominates |
| Waves/SM | Total grid depth relative to capacity | `<1` strongly suggests underfill |
| Registers/thread | Thread-private resource usage | High values can reduce resident blocks or spill |
| Dynamic shared memory | Explicit shared bytes/block | Can improve locality but limit residency |
| Duration | Kernel elapsed time | Final metric to improve for like-for-like work |

**Observed patterns:**

- **Global PCR, large grid:** early stages often reached ~90–92% DRAM
  utilization with strong L1TEX/long-scoreboard waiting.
- **Fused `4096x512`:** 1.06 ms, 89.01% SM, 41.77% DRAM; MIO and CTA-barrier
  stalls replaced much of the global-memory waiting.
- **Fused `1024x32`:** only 128 blocks and 0.75 waves/SM; the GPU was underfilled.

*See full guide: Sections 12–15 and 18–19.*

## 7. Three important experiments

### Shared-memory experiment

- **Hypothesis:** Early memory-bound stages might benefit from shared staging.
- **Design:** Shared/global hybrid through offset 256; global from 512 onward.
- **Measured result:** No consistent win. At `4096x512`, 6.559312 ms global
  versus 6.698496 ms hybrid: `0.979x`, about **2.1% slower**.
- **Lesson:** Profiler recommendations generate hypotheses; shared memory adds
  loads, halos, barriers, redundant work, and resource pressure.

### Fusion experiment

- Fused offsets: **1, 2, 4**.
- Dependency radius: **7**.
- One block: 256 outputs + two seven-value halos = 270 staged equations.
- Shared footprint: `2 * 4 * 270 * 8 =` **17,280 bytes/block**.
- `4096x512`: `1.104x` speedup, commonly described as ~10.4% faster by the
  speedup ratio (actual latency reduction ~9.4%).
- `2048x512`: `1.114x` (~11.4% speedup ratio).
- **Lesson:** Eliminating two launches and intermediate global round trips
  mattered more than caching one stage, once the grid was large enough.

### Adaptive dispatch

```text
fused if N * B >= 786,432
global otherwise
```

Identifier: `adaptive_work_v1:n_times_batch_gte_786432`

- Derived from a 100-point grid; ±2% treated as tied.
- Boundary region rerun with at least 50 iterations.
- ~**5.5% aggregate improvement** versus always-global.
- **Zero false-fusion cases** among decisive dense-grid points.

*See full guide: Sections 16–20.*

## 8. Common interview traps

| Do not say | Say instead |
|---|---|
| Higher occupancy is always faster. | Occupancy helps hide latency; resource-heavy kernels can be faster with lower occupancy. |
| Shared memory is always faster. | Shared access is fast, but staging, synchronization, halos, and residency have costs. |
| The GPU is always faster than the CPU. | The winner depends on algorithm, shape, residency, transfers, and timing scope. |
| `O(N)` always beats `O(N log N)`. | Lower work can lose when the higher-work algorithm has much shorter parallel span. |
| More parallelism always wins. | Parallelism must amortize launch, synchronization, and communication overhead. |
| Memory-bound means “use shared memory.” | It identifies a limit; shared memory is one hypothesis to measure. |
| Kernel-only time is application performance. | End-to-end time may be dominated by allocation and transfers. |
| Profiler advice is automatically correct. | Profile guidance must become a controlled experiment with a baseline. |

*See full guide: Section 26.*

## 9. Questions I should be able to answer

1. **Why PCR instead of Thomas on a GPU?** More work, but logarithmic span and
   independent equations within each stage expose SIMT parallelism.
2. **What is work versus span?** Work is total operations; span is the longest
   dependency chain and lower bound on ideal parallel time.
3. **Why did true batching help?** It removed the factor of `B` from stage-launch
   count and exposed system-level parallelism.
4. **Why did device-resident timing matter?** It isolated algorithm/kernel cost
   from allocation and PCIe transfers in accelerator-pipeline scenarios.
5. **What does memory-bound mean?** Data movement is the limiting resource for
   the measured kernel; it does not prescribe one optimization.
6. **What is occupancy?** Active resident warps relative to maximum, used to
   reason about latency hiding.
7. **Why can high occupancy still be slow?** Warps may all wait on the same
   bandwidth/dependency, or the algorithm may perform unnecessary traffic.
8. **What is a scoreboard stall?** A warp cannot issue because an input producer,
   often a memory operation, has not completed.
9. **Why did shared memory not help?** Its setup, halos, barriers, and resource
   costs did not offset saved loads, and inter-stage global traffic remained.
10. **Why did fusion help?** It removed launches plus full intermediate global
    reads/writes while retaining intermediate states on chip.
11. **Why could fusion hurt small workloads?** Too few blocks to amortize shared
    setup, barriers, halo work, and the larger kernel.
12. **What is waves/SM?** A grid-depth indicator; low waves mean insufficient
    blocks to sustain the GPU.
13. **Why adaptive dispatch?** Global and fused kernels win in different regimes;
    a simple measured rule preserves stable wins and avoids regressions.
14. **How would you validate another GPU?** Repeat the controlled grid and
    boundary runs, record provenance, and publish a new versioned threshold.
15. **What next?** First cross-GPU/application validation; then profile evidence
    for a fourth fused stage, CUDA Graphs, warp methods, or another backend.

*See full guide: Section 27 for expanded Q&A.*

## 10. Project story for interviews

### 30-second version

I compared CPU Thomas with CUDA PCR to study work versus parallel span. The main
architectural gain was flattening all systems into one true-batched launch per
PCR stage. I then separated allocation, transfer, and device-resident costs and
used Nsight to find memory-heavy early stages. A shared-memory attempt failed,
but fusing offsets 1,2,4 helped large workloads. A measured adaptive rule captured
those wins and improved aggregate grid runtime about 5.5% over always-global,
with CPU/GPU reference tests and sanitizer validation throughout.

### Two-minute version

Thomas is `O(N)` but has an `O(N)` dependency chain. PCR performs `O(N log N)`
work in `O(log N)` parallel stages, so it can better fit a GPU when enough systems
are batched. My initial CUDA baseline launched each system from a host loop,
which produced roughly `B log N` stage launches. Flattening `B*N` equations cut
that to `log N` and exposed both dimensions of parallelism.

Next I decomposed overhead. A reusable RAII workspace removed per-solve
allocations, and device-resident timing separated D2D reset from the PCR solve.
NVTX and Nsight then showed early large stages near 90% DRAM utilization with
long-scoreboard waits. I tested shared-memory stages through offset 256; the path
was correct but about 2.1% slower at the target, so I preserved the failed
hypothesis. I then fused offsets 1,2,4. A seven-value halo around 256 outputs and
two FP64 coefficient states required 17,280 bytes/block. On `4096x512`, the fused
early kernel shifted to 89% SM and 42% DRAM utilization, and the full solve
improved `1.104x`.

Fusion was noisy or slower on some small workloads, so I measured 100 shapes,
used a ±2% tie band, repeated the boundary, and implemented the versioned rule
`N*B >= 786,432`. The key lesson was to preserve baselines, separate timing
scopes, treat profiler guidance as a hypothesis, and dispatch when no kernel is
universally best.

*See full guide: Sections 28 and 30.*

## 11. Debugging story

A transient PCR bug wrote the right-side `beta*c[right]` coefficient to
`next_a` instead of `next_c`. Final mismatches did not reveal whether the fault
was initialization, stage math, pointer swapping, or solve logic. I compared
intermediate CPU/GPU coefficient state after every stage, found the first failing
stage, and traced the misplaced right-side coupling to its destination array.
I corrected the write, reran stage and final comparisons, and retained boundary
and non-power-of-two tests. The interview lesson is systematic root-cause work:
own the bug, instrument the earliest invariant, and fix the source rather than
patching downstream symptoms.

*See full guide: Section 29.*

## 12. Final cheat sheet

```text
CUDA hierarchy:   Grid -> Block/CTA -> Warp (32 threads) -> Thread
Memory hierarchy: Registers -> Shared/L1 -> L2 -> Global DRAM -> Host

Performance questions:
1. Is there enough parallel work?
2. Is the kernel memory- or compute-bound?
3. Is occupancy actually limiting?
4. Are launches or transfers dominating?
5. Can work be batched?
6. Can intermediate global traffic be eliminated?
7. Is the optimization workload-dependent?

PCR numbers:
- fused offsets: 1,2,4
- dependency radius: 7
- fused shared memory: 17,280 bytes/block
- large profile: 1.06 ms, 89.01% SM, 41.77% DRAM
- adaptive threshold: N*B >= 786,432
- rule ID: adaptive_work_v1:n_times_batch_gte_786432
- adaptive result: ~5.5% aggregate improvement over always-global
- target result: 1.104x at 4096x512 (~9.4% latency reduction)

Core lesson:
Correct baseline -> define timing scope -> measure -> profile -> form one
hypothesis -> validate -> remeasure -> preserve failures -> dispatch if needed.
```

For deeper review, use the full guide's **Vocabulary Glossary**, **Interview
questions and answers**, and **Final cheat sheet** sections.
