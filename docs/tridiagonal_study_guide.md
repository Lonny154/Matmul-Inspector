# Tridiagonal CUDA / PCR Study Guide

This guide describes the implementation on the `tridiagonal-solvers` branch at
commit `09390c3` (`Add adaptive PCR dispatch policy`). It uses the current source,
commit history, benchmark documentation, tests, and the local Nsight Compute
captures as evidence. Hardware measurements are from one NVIDIA GeForce RTX 4060
Ti (compute capability 8.9) with the CUDA 13.3 toolchain. They are examples, not
portable performance promises.

The most useful source map is:

| Concern | Source of truth |
|---|---|
| CPU Thomas and PCR | `src/tridiagonal.cpp`, `include/tridiagonal.hpp` |
| CUDA kernels, workspace, timing | `src/cuda_tridiagonal.cu` |
| Reusable workspace API | `include/cuda_tridiagonal_workspace.hpp` |
| Benchmark row API | `include/cuda_tridiagonal_benchmark.hpp`, `include/tridiagonal_benchmark.hpp` |
| Benchmark CLI and dispatch | `src/tridiagonal_benchmark_cli.cpp`, `src/tridiagonal_benchmark_main.cpp` |
| Policy analysis | `scripts/analyze_pcr_dispatch.py` |
| CPU and CUDA validation | `tests/tridiagonal_test.cpp`, `tests/cuda_tridiagonal_test.cpp` |
| Experimental record | `docs/tridiagonal_benchmarking.md` |

## 1. Big-picture overview

The project asks a practical systems question: **when can a parallel algorithm
with more arithmetic outperform a lower-work sequential algorithm because the
hardware rewards parallelism?** The concrete problem is solving many independent
tridiagonal linear systems. A CPU-friendly baseline, the Thomas algorithm, does
linear work but has a long dependency chain. Parallel Cyclic Reduction (PCR)
does more total work, yet reduces that chain to logarithmically many stages whose
equations can run concurrently.

Tridiagonal systems occur in one-dimensional discretizations, spline fitting,
implicit finite-difference methods, alternating-direction solvers, filtering,
and subproblems inside larger scientific and machine-learning pipelines. A
single small system may favor a CPU. Thousands of systems already on an
accelerator change the cost model.

Here, **algorithm archaeology** means revisiting an established algorithm under
modern hardware constraints. It is not a claim that PCR is new. It asks whether
older conclusions based primarily on arithmetic count still hold once SIMT
parallelism, memory hierarchy, launch overhead, and data residency are included.

The branch history records this progression:

1. **CPU Thomas** (`da783bc`): correct, low-work sequential baseline.
2. **CPU PCR** (`6f7344e`): reference implementation of the parallel algorithm.
3. **CUDA PCR** (`f96a1fa`): one thread per equation, initially one system.
4. **Benchmark harness** (`475f6a9`): CPU, kernel-only, and end-to-end scopes.
5. **True batching** (`5048895`): flatten `B*N` equations into one launch per stage.
6. **Reusable workspace** (`9e6d588`): remove repeated `cudaMalloc`/`cudaFree`.
7. **Device residency** (`b6f1e5c`): separate D2D reset from solve timing.
8. **Stage-aware profiling** (`709e29c`): NVTX names expose each PCR offset.
9. **Shared/global hybrid** (`e11955a`): cache early-stage neighborhoods.
10. **Fused early stages** (`40392b0`): combine offsets 1, 2, and 4.
11. **Winner/loser profiling**: explain why fusion depends on workload size.
12. **Adaptive dispatch** (`09390c3`): select global or fused with a measured rule.

The bottleneck moved at each step. The initial GPU version paid too many host
launches. True batching removed the factor of `B` from launch count. One-shot
execution then exposed allocation and transfer overhead. Persistent buffers
removed allocation, making H2D/D2H more visible. Device residency isolated the
algorithm. Profiling then showed early global stages limited by DRAM and long
scoreboard waits. Shared memory moved some waits to shared-memory/MIO and barrier
cost without consistent benefit. Fusion finally removed two launch boundaries
and two full intermediate global-memory round trips, but only large enough grids
amortized its extra work and resources. Adaptive dispatch encoded that regime.

## 2. Tridiagonal systems

A tridiagonal system has nonzero coefficients only on the main diagonal and the
two neighboring diagonals:

\[
a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i.
\]

- `a` is the **lower diagonal**; it couples row `i` to `x[i-1]`.
- `b` is the **main diagonal**.
- `c` is the **upper diagonal**; it couples row `i` to `x[i+1]`.
- `d` is the right-hand-side (RHS) vector.
- `x` is the unknown solution vector.
- The first row has no left neighbor; the last has no right neighbor. These are
  the boundary rows.

A 4-by-4 example is:

```text
[ b0  c0   0   0 ] [x0]   [d0]
[ a1  b1  c1   0 ] [x1] = [d1]
[  0  a2  b2  c2 ] [x2]   [d2]
[  0   0  a3  b3 ] [x3]   [d3]
```

`make_benchmark_systems()` creates deterministic systems with `a=c=-1` and
`b=4`, then constructs `d=A*x_expected`. These matrices are strictly diagonally
dominant: `|b_i|` exceeds the sum of off-diagonal magnitudes in its row. This
makes the fixtures well behaved and supplies a known solution for validation.

The public compact inputs contain `N-1` lower and upper entries. CPU and CUDA
PCR expand them into length-`N` arrays with `a[0]=0` and `c[N-1]=0`, which makes
per-equation indexing uniform.

## 3. Thomas algorithm

Thomas is specialized Gaussian elimination for tridiagonal matrices. It first
eliminates the lower diagonal and then back-substitutes.

```text
b = copy(main_diagonal)
d = copy(rhs)

for i = 1 .. N-1:                  # forward elimination
    m = lower[i-1] / b[i-1]
    b[i] = b[i] - m * upper[i-1]
    d[i] = d[i] - m * d[i-1]

x[N-1] = d[N-1] / b[N-1]
for i = N-2 .. 0:                  # back substitution
    x[i] = (d[i] - upper[i] * x[i+1]) / b[i]
```

The implementation is `solve_thomas()` in `src/tridiagonal.cpp`. It checks
dimensions and zero pivots. Its work is `O(N)` and extra working storage is
`O(N)` in this implementation. Its critical path, or **span**, is also `O(N)`:
iteration `i` needs the transformed values from `i-1`, and backward iteration
`i` needs `x[i+1]`.

This regular sequential loop runs well on a CPU: little arithmetic, contiguous
memory, few branches, and no kernel-launch cost. A literal GPU port cannot expose
many equations within one system because of the dependency chain. Independent
systems can still run in parallel, but the within-system parallelism remains
limited.

## 4. Parallel Cyclic Reduction (PCR)

PCR eliminates increasingly distant neighbors. At a stage with offset `s`, row
`i` combines its current equation with rows `i-s` and `i+s`. Every output row
reads only the previous stage, so all rows in one stage are independent.

For the left neighbor, define:

\[
\alpha_i = -a_i / b_{i-s}.
\]

For the right neighbor, define:

\[
\beta_i = -c_i / b_{i+s}.
\]

The repository update is conceptually:

```text
next_a[i] = 0
next_b[i] = b[i]
next_c[i] = 0
next_d[i] = d[i]

if i >= offset:
    left = i - offset
    alpha = -a[i] / b[left]
    next_a[i] = alpha * a[left]
    next_b[i] += alpha * c[left]
    next_d[i] += alpha * d[left]

if i + offset < N:
    right = i + offset
    beta = -c[i] / b[right]
    next_b[i] += beta * a[right]
    next_c[i] = beta * c[right]
    next_d[i] += beta * d[right]
```

Offsets double: `1, 2, 4, 8, ...`, stopping when `offset >= N`. Thus there are
`ceil(log2(N))` stages, including non-power-of-two `N`. After the final stage,
off-diagonal couplings have been eliminated and each row is independent:

\[
x_i = d_i / b_i.
\]

The old and new coefficient arrays are **ping-pong buffers**. A stage must not
overwrite values that another equation still needs from the same logical stage.
After a launch, pointer swaps make `next_*` the current state.

PCR performs `O(N log N)` work versus Thomas's `O(N)`, but its ideal parallel
span is `O(log N)`. In the CPU reference the inner equations are executed by a
normal loop, so that implementation does not realize the parallel span. The
CUDA mapping does.

## 5. Work versus span

**Work** counts all operations. **Span** is the longest dependency chain, or the
time on an ideal machine with unlimited processors. Available parallelism is
often described by `work/span`.

| Algorithm | Work | Span | Main property |
|---|---:|---:|---|
| Thomas | `O(N)` | `O(N)` | Low work, serial chain |
| PCR | `O(N log N)` | `O(log N)` | More work, many independent equations per stage |

A GPU has many execution lanes. It can finish more arithmetic sooner if that
arithmetic is independent and the grid is large enough. This does not violate
complexity theory: PCR spends more resources, but shortens the critical path.

Latency asks how long one solve takes. Throughput asks how many systems or
equations complete per unit time. Thomas can have excellent single-system CPU
latency, while batched PCR can have better accelerator throughput. Amdahl's Law
warns that unavoidable serial work and launch/transfer overhead cap speedup.
Gustafson's Law explains why increasing the batch can make accelerator
parallelism worthwhile: the parallel portion grows while fixed overhead is
amortized.

## 6. Mapping PCR to CUDA

The true-batched kernel treats all systems as a flat array of `B*N` equations:

```cpp
global   = blockIdx.x * blockDim.x + threadIdx.x;
system   = global / n;
equation = global % n;
base     = system * n;
```

One CUDA thread updates one `(system,equation)` pair. Neighbor indices use
`base + equation ± offset`, and conditions use the within-system `equation`.
Those checks are essential: flat neighbors must never cross a system boundary.

CUDA hierarchy in this implementation:

```text
grid
  +-- block 0: 256 threads
  +-- block 1: 256 threads
  +-- ...

thread global index -> one equation in one system
```

A **thread** is a logical CUDA worker. Threads are grouped into **blocks**; this
code uses 256 threads per block. Hardware schedules threads in 32-thread
**warps** on a Streaming Multiprocessor (**SM**). The **grid** is all blocks in a
kernel launch.

The serial-host-loop baseline invokes a complete single-system PCR sequence for
each of `B` systems. It therefore issues approximately:

```text
B * ceil(log2(N)) stage launches + B final-solve launches
```

True batching launches each stage once across all `B*N` equations:

```text
ceil(log2(N)) stage launches + one final-solve launch
```

This exposes both equation-level and system-level parallelism and removes the
factor of `B` from host launch count. It changes scheduling, not PCR mathematics.

## 7. Kernel-launch overhead

A kernel launch is a host request that schedules a grid on the GPU. Submission,
driver/runtime bookkeeping, and device scheduling have a fixed cost even when
the kernel performs almost no work. For tiny kernels, launch latency may be
comparable to or greater than useful execution.

Batching helps by putting more useful work behind one launch. Fusion helps by
combining multiple logical stages in one launch. Both are forms of amortization:
pay a fixed cost fewer times or spread it over more work.

Fusion is not free. It can require more registers, shared memory, barriers, halo
work, and instructions. The `1024x32` profile demonstrates that a more complex
fused kernel can be unattractive when only 128 blocks are launched on 34 SMs.
The `4096x512` case launches 8,192 blocks and can amortize the added machinery.

## 8. Persistent/reusable workspace

`DeviceDoubleBuffer` owns one `double*` allocated with `cudaMalloc` and frees it
in its destructor. This is Resource Acquisition Is Initialization (**RAII**):
ownership follows object lifetime, including exception paths.

The original one-shot path allocated nine buffers every solve: four current
coefficient arrays, four next arrays, and the result. `cudaMalloc` and
`cudaFree` involve runtime/device bookkeeping and are expensive in a repeated
solve loop.

`CudaPcrBatchedWorkspace` allocates once and exposes phases:

```text
upload()              host coefficients -> mutable device buffers
make_device_resident() mutable buffers -> immutable device snapshots
reset_from_device()   immutable snapshots -> mutable working buffers
execute*()            PCR stages and final solve
download()            device result -> caller-provided host buffer
```

The timing scopes answer different questions:

| Scope | Includes | Excludes |
|---|---|---|
| one-shot `end_to_end` | allocation, H2D, kernels, sync, D2H, cleanup | correctness checks |
| `reusable_end_to_end` | H2D reset, kernels, sync, D2H | workspace allocation/cleanup |
| `kernel_only` | all stage kernels and final solve | allocation, reset copies, H2D, D2H |
| `device_resident` | all stage kernels and final solve | allocation, H2D, D2H, D2D reset |

A reusable workspace is not a persistent kernel. Buffers persist; kernels are
still launched normally.

## 9. Host/device transfers

- **H2D**: host-to-device transfer, normally across PCIe on a discrete GPU.
- **D2H**: device-to-host transfer.
- **D2D**: device-to-device copy.

PCIe latency and bandwidth are much worse than on-device accesses. Once repeated
allocation was removed, input and output movement became a larger fraction of
end-to-end time. A fast kernel can therefore lose to CPU Thomas if every solve
must upload coefficients and immediately download results.

That is why the benchmark never calls a kernel-only result an application
speedup. End-to-end and kernel-only measurements answer different questions.
Pinned host memory could change transfer behavior, but this milestone uses the
repository's existing pageable host vectors and does not claim pinned-transfer
performance.

## 10. Device-resident execution

Device-resident means inputs are already on the GPU before the timed solve and
the output remains there when timing ends. This models a larger ML/HPC pipeline
in which neighboring operations also execute on the accelerator.

PCR mutates its coefficient state, so repeatable sampling requires reset. The
workspace stores immutable device copies of `a,b,c,d` and D2D-copies them into
mutable working arrays. `d2d_reset` measures those four copies separately.
`device_resident` places its start event after reset and its stop event after the
final solve. The correctness download happens later.

This scope isolates the algorithm and kernel architecture. It does not predict
an application whose data originates on the CPU; it predicts an accelerator
pipeline where data residency is realistic.

## 11. Benchmark methodology

The harness validates results before collecting timing data. It runs warmups,
then measured iterations, stores `benchmark::Sample` objects, and reports median,
mean, minimum, and maximum milliseconds.

Warmups reduce first-use effects such as context initialization, code loading,
and clock-state transitions. CUDA-event timing measures elapsed work on the GPU
timeline. The code records a start event, launches the complete solve, records a
stop event, synchronizes the stop, and asks CUDA for elapsed milliseconds.
Host `steady_clock` is used when host-visible allocation, API, transfer, or
cleanup overhead belongs in the scope.

The median is robust to occasional scheduling or system outliers. The mean still
shows the average cost, while min/max expose spread. A median is not permission
to ignore noise: the adaptive experiment classified speedups between 0.98 and
1.02 as ties and reran boundary cases with at least 50 iterations.

Like-for-like comparison requires identical data, dimensions, warmups,
iterations, and timing boundaries. A kernel-only CUDA time cannot fairly be
compared to a CPU end-to-end time without saying what is omitted. One result on
one GPU is evidence for that environment, not a universal claim.

Useful current commands are:

```bash
cmake --build build -j

# One CPU-only configuration
./build/tridiagonal-benchmark --system-size 33 --batch-size 3 \
  --mode cpu --warmups 3 --iterations 20 --output results/pcr-cpu

# Global, fused, and adaptive device-resident rows
./build/tridiagonal-benchmark --system-size 4096 --batch-size 512 \
  --mode true_batched_adaptive --warmups 3 --iterations 20 \
  --output results/pcr-adaptive-4096x512

# Analyze a previously captured dense global/fused run
python3 scripts/analyze_pcr_dispatch.py results/pcr-dispatch-dense --sm-count 34
```

The output directory must be new/empty; this prevents silent overwriting.

## 12. Nsight Compute profiling

NVTX ranges name logical stages, for example
`pcr_global_stage_5_offset_32`,
`pcr_fused_stages_0_2_offsets_1_2_4`, and `pcr_final_solve`.
`MATMUL_INSPECTOR_NVTX=1` enables them. `PcrProfilingRange` becomes a no-op when
NVTX headers are unavailable and adds no synchronization.

Important Nsight Compute metrics:

| Metric | Meaning |
|---|---|
| DRAM Throughput | Fraction of peak off-chip DRAM throughput reached by the busiest relevant path |
| Memory Throughput | Overall memory-system utilization summary |
| Compute (SM) Throughput | Utilization of SM compute pipelines relative to peak |
| L1/TEX Throughput | Activity through the combined L1/texture path |
| L1/TEX Hit Rate | Fraction of requests served by that level rather than lower levels |
| L2 Hit Rate | Fraction served by shared device L2 |
| Registers Per Thread | Registers allocated to each thread; affects resident blocks/warps |
| Dynamic Shared Memory | Per-block shared bytes supplied at launch |
| Theoretical Occupancy | Resource-model maximum resident warps / hardware maximum |
| Achieved Occupancy | Time-averaged measured active warps / hardware maximum |
| Active Warps per SM | Resident warps that have not completed |
| Waves per SM | Grid blocks divided by aggregate simultaneous block capacity; a grid-size indicator |
| Elapsed Cycles / Duration | Kernel lifetime in device cycles / time |

`L1TEX` does not mean this project created CUDA texture objects. NVIDIA combines
several load/store/cache paths in the L1/TEX unit, including ordinary global and
local memory traffic.

For `4096x512`, each early global stage launched 8,192 blocks, used 40 registers
per thread, achieved roughly 86--88% occupancy, and reached about 90--92% DRAM
throughput. These metrics motivated an experiment; they did not prescribe its
outcome.

## 13. Warp stalls

A warp scheduler can issue an instruction only when its inputs and execution
resource are ready. A **scoreboard** tracks outstanding dependencies. A stall
reason describes why a sampled warp could not issue; it is a symptom to
investigate, not automatically a defect.

- **Long scoreboard** commonly means a warp awaits a high-latency L1TEX-related
  operation, often a global/local load that missed nearby caches.
- **Short scoreboard** often means a dependency on lower-latency memory-input/
  output (**MIO**) work such as shared-memory operations or special-function
  instructions.
- **MIO throttle/dependency** means relevant memory/special instruction paths or
  their dependencies limit issue.
- **CTA barrier** means a warp waits at a block-wide barrier for sibling warps.
- A general dependency stall means a consumer cannot execute until a producer
  completes.

The early global PCR profiles spent large fractions of issue intervals waiting
on L1TEX scoreboard dependencies. Offset-1 at `4096x512`, for example, reported
about 126.5 stalled cycles out of 172.2 cycles between issued instructions. The
fused kernel moved the pattern: about 44.2 of 107.1 cycles were attributed to
MIO dependencies and about 33.9 to CTA barriers. That is consistent with
replacing global intermediate traffic with shared-memory exchanges and
`__syncthreads()`.

Latency hiding occurs when the scheduler issues a different ready warp while
one warp waits. It requires enough active and eligible warps. A high stall
percentage can coexist with good performance if other warps keep hardware busy,
or it can identify the current limiting dependency. Always relate stalls to
duration, throughput, source locations, and grid/resource information.

## 14. Occupancy

Occupancy is the ratio of active warps to the architectural maximum per SM.
Theoretical occupancy comes from static resource constraints; achieved occupancy
is measured over time.

Resident blocks may be limited by:

- threads or warps per SM;
- registers per SM divided by registers per block;
- shared memory per SM divided by shared memory per block;
- the architectural block-count limit.

The global stage used 40 registers/thread, no dynamic shared memory, and had 100%
theoretical occupancy in the capture. The fused kernel used 42 registers/thread
and 17.28 KB dynamic shared memory, limiting theoretical occupancy to 83.33%; it
still achieved 82.03% on the large workload and ran faster than three separate
early stages.

This is why “maximize occupancy” is a bad optimization objective. Occupancy is a
means of hiding latency. Once enough warps exist, more occupancy may not help;
using registers or shared memory can reduce occupancy while reducing total work
or memory traffic. Conversely, `1024x32` had only 128 blocks. Its low utilization
was caused largely by total grid size, not merely per-SM occupancy capacity.

## 15. Memory hierarchy

From closest to broadest scope:

| Storage | Scope/control | Typical role in this project |
|---|---|---|
| Registers | Private to one thread; compiler allocated | Indices, `alpha`, `beta`, temporary values |
| Shared memory | Explicit, block-scoped, on chip | Staged coefficient windows and fused intermediate states |
| L1 cache / L1TEX | Per-SM hardware cache/path | Caches ordinary global loads among other traffic |
| L2 cache | Shared across GPU | Reuse and traffic filtering before DRAM |
| Global/device memory | All GPU threads; high capacity, high latency | Persistent coefficient, ping-pong, and result arrays |
| Host memory | CPU-visible; across interconnect from discrete GPU | Input vectors and downloaded results |

Registers have the lowest access latency but are private and scarce. Shared
memory is explicitly indexed and much lower latency than uncached DRAM, but
requires cooperative loads, barriers, and capacity budgeting. Caches are
hardware-managed. Global memory has high bandwidth only when many accesses are
well structured and concurrent; individual access latency is high.

**Spatial locality** means nearby addresses are used together. Flattened PCR
threads access adjacent equations, producing coalesced center-array accesses.
**Temporal locality** means recently used data is reused. Early neighbor values
may be reused by nearby threads and caches. **Coalescing** combines warp accesses
to contiguous addresses into efficient memory transactions.

The shared and fused kernels deliberately load a contiguous center-plus-halo
window. Halo loads are redundant across neighboring blocks. This can be a good
trade if on-chip reuse saves more off-chip traffic than loading and synchronizing
the halo costs. The experiment showed that this balance differs by design and
workload.

## 16. Shared-memory experiment

The hypothesis was:

> Early PCR stages are strongly DRAM-bound and wait on global-memory
> dependencies; staging each block's neighbor window in shared memory may reduce
> global loads.

`pcr_batched_shared_stage_kernel` allocates four dynamic arrays for `a,b,c,d`.
For 256 output threads and offset `s`, each array spans `256 + 2s` elements. The
halo supplies neighbors on both sides. Threads cooperatively load the span,
execute `__syncthreads()`, then compute their owned output while checking the
within-system equation boundaries.

Shared footprint is:

\[
4(256+2s)\times 8\ \text{bytes}.
\]

Examples are 8,256 bytes at offset 1, 10,240 at offset 32, 16,384 at offset 128,
and 24,576 at offset 256. The hybrid uses shared stages through offset 256 and
the unchanged global kernel from offset 512 onward.

The result was not a consistent win. At `4096x512`, global measured 6.559312 ms
and hybrid 6.698496 ms: speedup `0.979x`, or about 2.1% slower. Many larger cases
were tied or slightly slower. One tiny point showed a large nominal win, another
regressed; those short durations were sensitive to noise.

Why might a plausible shared-memory optimization fail?

- It adds cooperative-load instructions and a barrier every stage.
- Neighboring blocks redundantly load halo data.
- Existing L1/L2 caching may already capture useful reuse.
- Dynamic shared memory may reduce resident blocks.
- PCR still writes every full stage back to global memory.
- It can move the bottleneck from DRAM/long-scoreboard waits to MIO/barriers.

The result is valuable because it rejects a concrete hypothesis while preserving
the baseline. “Memory-bound” does not logically imply “shared memory wins.”

## 17. Kernel fusion

Kernel fusion combines multiple logical operations into one launch so
intermediate state can remain on chip. The fused PCR kernel combines offsets 1,
2, and 4, then the normal global loop resumes at offset 8.

Dependency radius grows as follows:

```text
after offset 1: radius 1
after offset 2: radius 1 + 2 = 3
after offset 4: radius 1 + 2 + 4 = 7
```

For a block that owns 256 final equations, the kernel loads seven extra equations
on each side: `256 + 14 = 270`. It computes 268 offset-1 intermediates, 264
offset-2 intermediates, and 256 offset-4 outputs. Two shared ping-pong states
each hold four FP64 arrays:

\[
2\times4\times270\times8 = 17{,}280\ \text{bytes/block}.
\]

Barriers separate initial loading, offset 1, and offset 2. Every thread reaches
the barriers; only after the last barrier may out-of-range output threads return.
The per-stage update still checks logical equation boundaries, so a halo never
borrows coefficients from another flattened system. Systems with `N<=4` use the
global path because they do not contain all three logical stages.

Benefits:

- two fewer kernel launches for the first three stages;
- offset-1 and offset-2 intermediates never round-trip through global memory;
- intermediate producer/consumer locality is explicit.

Costs:

- 17,280 bytes of dynamic shared memory per block;
- two block-wide barriers;
- redundant halo computation and loads;
- a larger kernel with 42 registers/thread in the measured build;
- lower theoretical occupancy than the global stage.

Fusion succeeded where single-stage shared caching did not because it removed
whole intermediate global writes and reads as well as launches. It changed more
than the location of one stage's input.

## 18. Fused-kernel profiling

For `N=4096`, `B=512`, there are 2,097,152 equations and 8,192 blocks. The
fused early-stage kernel measured:

| Metric | Measured value |
|---|---:|
| Duration | 1.06 ms |
| Compute (SM) throughput | 89.01% |
| DRAM throughput | 41.77% |
| L2 hit rate | 91.74% |
| Registers/thread | 42 |
| Dynamic shared memory/block | 17.28 KB |
| Waves/SM | 48.19 |
| Theoretical / achieved occupancy | 83.33% / 82.03% |

The separate global offsets 1 and 2 took about 466 µs each, and the neighboring
early stages were generally around 0.46–0.50 ms with roughly 90% DRAM
throughput. Three such launches total roughly 1.4 ms. Fusion traded DRAM traffic
for compute, shared-memory, and barrier work, producing a shorter combined stage.
The complete benchmark improved from 6.573568 to 5.956096 ms (`1.104x`) in the
documented run.

For `1024x32`, only 32,768 equations and 128 blocks were present. The fused
capture reported 23.07 µs, 0.75 waves/SM, 64.41% SM throughput, 25.95% DRAM
throughput, and 58.31% achieved occupancy. The original documented benchmark
showed a regression (`0.735x`), although later repeated runs sometimes favored
fusion. This instability is itself evidence that launch-scale timings need
conservative interpretation.

## 19. GPU underutilization

Occupancy asks how many warps can be resident within an SM. Grid utilization
asks whether the launch supplies enough blocks to all SMs for enough time. They
are distinct.

With only 128 blocks on 34 SMs, `1024x32` has little scheduling depth. The fused
profile's 0.75 waves/SM means there is less than one full modeled wave of blocks
per SM under the profiler's resource calculation. Fixed launch, shared setup,
halo, and barrier costs cannot be spread over long sustained execution.

At `4096x512`, 8,192 blocks produce 48.19 waves/SM. Work remains available as
blocks finish; latency can be hidden, pipelines stay busy, and removing global
round trips matters. Adding instructions to an underfilled kernel does not create
useful independent work automatically; it may simply make each scarce block
more expensive.

## 20. Adaptive dispatch

Production performance libraries commonly retain several kernels because no
single implementation dominates every shape and device. This branch adds a
transparent rule:

```text
if N * B >= 786,432: fused
else:                global
```

Its identifier is:

```text
adaptive_work_v1:n_times_batch_gte_786432
```

`N*B` is a proxy for total equations, grid size, and the opportunity to amortize
fusion overhead. `should_use_fused_pcr()` computes the rule without overflowing.
`execute_adaptive()` delegates to `execute_fused()` or `execute()`; it does not
duplicate either kernel.

The policy came from 100 points: ten `N` values by ten batch sizes, each with
three warmups and 20 measured iterations. Points within ±2% were ties. The
transition region was rerun with 50 iterations and `1024x32` with 100.

- **False fusion**: policy selects fused when a decisive point favored global.
- **False global**: policy selects global when a decisive point favored fused.
- **Oracle**: picks the faster measurement independently at every point.
- **Always-global/fused**: fixed comparison policies.
- **Aggregate runtime**: sum of point medians, useful only for comparing these
  policies on the measured grid.

The selected rule improved aggregate measured runtime by about 5.5% over
always-global and had zero false-fusion cases among decisive dense-grid points.
Its raw winner-classification accuracy was low because it intentionally rejected
many noisy tiny-workload fused wins. This is a conservative engineering choice:
avoid robust regressions while capturing stable 9–13% large-workload gains.

`scripts/analyze_pcr_dispatch.py` reconstructs paired medians, speedup, winner,
total equations, fused blocks, and optional estimated waves. Adaptive CSV rows
also record `selected_path` and `dispatch_rule`, making each decision auditable.

The threshold is hardware-specific. On another GPU, rerun the grid and issue a
new versioned rule rather than silently changing `adaptive_work_v1`.

## 21. Performance-portability lesson

The project demonstrates conditional winners:

- CPU Thomas is excellent when work is small or data is host-resident.
- PCR becomes attractive when enough independent work exposes GPU parallelism.
- True batching matters more than micro-optimizing a serial launch pattern.
- Reuse and residency can matter more than kernel arithmetic.
- Fused PCR helps saturated workloads and can lose on underfilled ones.
- A simple dispatcher can retain different winners in different regimes.

This pattern appears in cuBLAS and cuDNN kernel selection, Triton autotuning,
compiler-generated kernels, and accelerator runtime libraries. It also applies
conceptually to systems such as AWS Neuron/Trainium: workload shapes, on-chip
storage, launch/runtime overhead, and architecture-specific throughput determine
which schedule wins. No proprietary behavior is implied here.

Performance portability does not mean one binary has identical speed everywhere.
It means preserving correctness while selecting or generating a suitable
implementation from measured or modeled hardware characteristics.

## 22. Algorithm archaeology

An algorithm's asymptotic work is only one part of its cost model. Modern
machines add vector/SIMT width, memory tiers, synchronization, launch latency,
topology, and massive concurrency. An older “too much arithmetic” algorithm may
become useful when it shortens dependencies; an older cache optimization may
become redundant when hardware caches improve.

Good archaeology questions include:

- reductions: serial, tree, warp, pairwise, compensated;
- scans: work-efficient versus low-span variants;
- FFTs: radix and fusion choices by size;
- sorting: comparison, radix, network, and bucket methods;
- sparse algorithms: format and irregularity versus reuse;
- dynamic programming: wavefront parallelism and tiling;
- linear solvers: direct, cyclic-reduction, iterative, and batched regimes.

The method is empirical and disciplined: recover assumptions, construct fair
baselines, expose current bottlenecks, and retain negative results.

## 23. Experimental method

The branch followed a repeatable loop:

1. Establish a correct CPU and GPU baseline.
2. Benchmark explicit timing scopes.
3. Profile the dominant configuration.
4. Form a narrow hypothesis.
5. Change one architectural variable.
6. Validate difficult dimensions and state reuse.
7. Benchmark the same scope again.
8. Profile both winners and losers.
9. Accept, reject, or limit the hypothesis.
10. Record evidence and preserve the baseline.

The shared-memory result is exemplary. Profiling found DRAM pressure. The
hypothesis “cache neighbor windows” was reasonable. Correctness passed, but
performance did not consistently improve. Rather than hiding the failure or
stacking more changes, the project preserved it, explained likely costs, and
formed a new hypothesis: fuse stages to eliminate intermediate traffic. Fusion
then showed a workload-dependent win, which led to dispatch rather than a claim
of universal superiority.

## 24. Correctness and validation

Kernel engineering requires proving that performance changes preserve the
algorithm. The branch uses several layers:

- known solutions generated first, with RHS computed as `A*x_expected`;
- CPU PCR comparison for every benchmark path;
- global CUDA PCR comparison for hybrid, fused, and adaptive paths;
- sizes around powers of two and block boundaries: 255, 256, 257, among others;
- non-power-of-two sizes such as 31, 33, 100, 257, and 1000;
- awkward batches such as 3, 5, 7, and 8;
- repeated solves from one immutable resident snapshot;
- replacement of resident inputs in an existing workspace;
- adaptive tests below and exactly at the 786,432-equation threshold;
- CTest labels separating 22 CPU and 9 GPU tests in the validated tree;
- Compute Sanitizer memcheck; and
- `git diff --check` for patch hygiene.

Compute Sanitizer's memcheck detects illegal and out-of-bounds device memory
accesses and related CUDA memory errors. Other sanitizer tools can examine races,
initialization, or synchronization hazards; memcheck alone is not a complete
race proof. Algorithmic comparisons and boundary-focused tests remain necessary.

The transient development bug discussed later demonstrates why final-output
tests sometimes need intermediate-state instrumentation: many parallel updates
can obscure the first place state becomes wrong.

## 25. Vocabulary glossary

### Algorithms and complexity

- **Algorithm:** A finite procedure that transforms inputs into outputs.
- **Asymptotic complexity:** How resource use grows as input size grows,
  abstracting away constants.
- **Big-O notation:** An upper-bound growth class such as `O(N)` or `O(log N)`.
- **Work:** Total operations performed across all workers.
- **Span:** Longest dependency chain on an ideal parallel machine.
- **Critical path:** The sequence that determines the minimum possible latency;
  another name for span.
- **Sequential dependency:** A computation that cannot begin until an earlier
  result exists, as in Thomas forward elimination.
- **Parallelism:** Work that can execute at the same time.
- **Concurrency:** Multiple tasks making progress during overlapping time;
  concurrency describes structure, while parallelism requires simultaneous
  execution resources.
- **Throughput:** Completed work per unit time.
- **Latency:** Time to complete one requested operation or batch.
- **Arithmetic intensity:** Arithmetic operations per byte moved at a specified
  memory level.
- **Operational intensity:** Roofline term for operations per byte transferred
  from the modeled memory level.
- **Memory-bound:** Performance limited mainly by data movement.
- **Compute-bound:** Performance limited mainly by arithmetic execution capacity.
- **Roofline model:** Upper-bound model relating operational intensity, memory
  bandwidth, and peak compute.
- **Amdahl's Law:** Fixed serial fractions limit parallel speedup.
- **Gustafson's Law:** Larger parallel workloads can scale while fixed serial
  work becomes a smaller fraction.
- **Algorithmic tradeoff:** Exchanging one resource or property for another,
  such as PCR's extra work for shorter span.
- **Cost model:** A simplified prediction using work, traffic, launches,
  resources, or measured constants.
- **Algorithm archaeology:** Reassessing known algorithms under different
  hardware and workload assumptions.

### Hardware and CUDA execution

- **CPU:** Latency-oriented general-purpose processor with a few sophisticated
  cores and large caches.
- **GPU:** Throughput-oriented processor containing many execution lanes and SMs.
- **Accelerator:** Specialized processor used to accelerate a workload class.
- **CUDA:** NVIDIA's programming/runtime platform for GPU computation.
- **Kernel:** Function launched from the host to execute across many GPU threads.
- **Host:** CPU side of a CUDA program.
- **Device:** GPU side of a CUDA program.
- **Kernel launch:** Host submission of one kernel grid.
- **Grid:** All blocks in one CUDA kernel launch.
- **Block:** Cooperating thread group scheduled on one SM at a time; CUDA also
  calls it a Cooperative Thread Array (CTA).
- **Thread:** One logical kernel invocation identified by CUDA indices.
- **Warp:** Hardware scheduling group of 32 CUDA threads on current NVIDIA GPUs.
- **SM:** Streaming Multiprocessor; executes resident blocks and warps.
- **SIMD:** Single Instruction, Multiple Data; vector lanes execute one explicit
  vector instruction.
- **SIMT:** Single Instruction, Multiple Threads; CUDA presents scalar threads
  that hardware groups into warps.
- **Lane:** One execution position within a warp/vector.
- **Scheduler:** Hardware logic choosing ready work to issue.
- **Warp scheduler:** SM component that selects eligible warps and issues their
  instructions.
- **Active warp:** Resident warp that has not completed.
- **Eligible warp:** Active warp whose next instruction is ready to issue.
- **Waves per SM:** Profiler grid-depth metric based on blocks and simultaneous
  capacity; low values reveal underfilled launches.
- **CTA:** Cooperative Thread Array, CUDA's thread block.
- **Branch divergence:** Threads in one warp follow different control paths,
  causing paths to execute with different active masks.
- **Predication:** Executing an instruction with per-lane enable masks.
- **Active threads per warp:** Average lanes participating in executed warp
  instructions.
- **Predicated-off thread:** Lane present in a warp but disabled for an
  instruction.
- **Instruction-level parallelism:** Independent instructions from one thread or
  warp that hardware may overlap.
- **Thread-level parallelism:** Independent work exposed across threads/warps.
- **Memory-level parallelism:** Multiple outstanding memory requests that can
  overlap latency.
- **Underutilization:** Too little ready work to occupy available hardware.
- **Resource saturation:** A resource operates near its sustainable maximum.

### GPU resources and memory

- **Occupancy:** Resident active warps divided by architectural maximum warps.
- **Theoretical occupancy:** Static upper bound from registers, shared memory,
  threads, warps, and block limits.
- **Achieved occupancy:** Measured time-averaged active-warp ratio.
- **Register:** Fast thread-private hardware storage.
- **Register pressure:** Demand for registers; high demand can reduce residency
  or cause spills.
- **Spilling:** Compiler placement of values in local memory when registers are
  insufficient.
- **Local memory:** Per-thread address space commonly backed by device memory;
  despite its name it is not the same as low-latency shared memory.
- **Shared memory:** Explicit on-chip, block-scoped storage.
- **Dynamic shared memory:** Per-block bytes selected at launch using the third
  kernel launch parameter.
- **Global memory:** Large device DRAM address space accessible by all threads.
- **Device memory:** Memory attached to the accelerator; often used synonymously
  with global memory in CUDA discussion.
- **Host memory:** CPU-side system memory.
- **L1 cache:** Small per-SM cache close to execution units.
- **L1TEX:** NVIDIA combined L1/texture/load-store hardware path; ordinary global
  traffic can use it without texture objects.
- **L2 cache:** Larger device-wide cache shared by SMs.
- **Cache hit:** Requested data found at the cache level.
- **Cache miss:** Request continues to a lower/farther memory level.
- **Temporal locality:** Reusing the same data soon.
- **Spatial locality:** Accessing nearby addresses.
- **Memory coalescing:** Combining adjacent warp accesses into efficient memory
  transactions.
- **Memory bandwidth:** Bytes transferable per unit time.
- **DRAM:** Off-chip dynamic random-access memory backing CUDA global memory.
- **Memory throughput:** Achieved utilization or byte rate of memory resources.
- **Compute throughput:** Achieved utilization or operation rate of compute
  resources.
- **Global load/store:** Read/write to global device memory.
- **Shared load/store:** Read/write to block-local shared memory.
- **PCIe:** Common host-device interconnect for a discrete GPU.
- **H2D:** Host-to-device copy.
- **D2H:** Device-to-host copy.
- **D2D:** Device-to-device copy.
- **Device resident:** Inputs and outputs remain on accelerator across the timed
  operation.
- **Pinned memory:** Page-locked host memory that supports more efficient DMA;
  not introduced by this tridiagonal milestone.
- **Buffer:** Contiguous storage region.
- **Double buffer:** Two storage sets, one read and one write, swapped between
  phases.
- **Ping-pong buffer:** Synonym for double buffer; PCR uses current and next
  coefficient arrays.
- **Workspace:** Reusable collection of temporary and result buffers.
- **Persistent allocation:** Buffer lifetime spans repeated operations.
- **Persistent kernel:** Long-lived kernel that repeatedly consumes work; this
  project has persistent buffers, not a persistent PCR kernel.
- **RAII:** C++ lifetime pattern where constructors acquire resources and
  destructors release them.

### Synchronization and dependencies

- **Synchronization:** Coordination ensuring required work/data visibility before
  dependent work proceeds.
- **Barrier:** Point all participating workers must reach before any proceed.
- **`__syncthreads()`:** CUDA block-wide execution and shared-memory visibility
  barrier.
- **Grid-wide synchronization:** Coordination across blocks. Ordinary kernels do
  not provide it internally; separate launches form stage boundaries here.
- **Data dependency:** Relationship where an operation needs a prior result.
- **Scoreboard:** Hardware tracker for instruction operands awaiting producers.
- **Long scoreboard:** Stall commonly waiting for high-latency L1TEX/global/local
  memory dependencies.
- **Short scoreboard:** Stall commonly waiting for shared-memory/MIO or other
  shorter-latency dependencies.
- **MIO:** Memory input/output instruction pipeline category, including shared
  memory and some special operations.
- **CTA barrier:** Stall while sibling warps in a block reach a barrier.
- **Instruction latency:** Delay from issuing an instruction until its result is
  usable.
- **Latency hiding:** Running other ready warps while one waits.
- **Race condition:** Result depends on uncontrolled event ordering.
- **Data race:** Concurrent unsynchronized accesses to one location with at least
  one write.
- **Boundary condition:** Special rule at the start/end of a system or partial
  block.

### Tiling, fusion, and dispatch

- **Halo:** Extra neighboring values loaded around a block's owned output region.
- **Tile:** Block-local subset of a larger data domain.
- **Tiling:** Dividing data/work into tiles to improve locality or cooperation.
- **Fusion:** Combining operations so intermediates remain local and fixed costs
  are shared.
- **Kernel fusion:** Fusion specifically within one GPU kernel launch.
- **Fission:** Splitting one operation/kernel into multiple pieces.
- **Dispatch:** Selecting an implementation for a request.
- **Adaptive dispatch:** Runtime selection using shape, hardware, or measurements.
- **Heuristic:** Simple practical decision rule rather than an exact proof.
- **Threshold:** Boundary at which a heuristic changes choice.
- **Oracle policy:** Retrospective policy choosing the measured winner at every
  point; useful as an upper bound, unavailable prospectively without measurement.
- **False fusion:** Adaptive rule selects fused where decisive data favored global.
- **False global:** Rule selects global where decisive data favored fused.
- **Conservative heuristic:** Rule designed to avoid costly regressions even if it
  misses some wins.

### Measurement and tooling

- **Benchmark:** Controlled performance measurement of defined work.
- **Microbenchmark:** Narrow benchmark isolating a component or operation.
- **Warmup:** Untallied execution before samples to reduce cold-start effects.
- **Iteration:** One repeated execution/sample.
- **Median:** Middle sorted value; robust to isolated outliers.
- **Mean:** Arithmetic average; sensitive to all samples and outliers.
- **Variance:** Average squared deviation from the mean; measures spread.
- **Outlier:** Observation far from typical samples.
- **Noise:** Uncontrolled timing variation from clocks, scheduling, load, or tools.
- **Confidence interval:** Estimated range for a population parameter under a
  statistical procedure; this benchmark reports basic statistics, not CIs.
- **Reproducibility:** Ability to repeat a configured experiment with provenance
  and obtain interpretable results.
- **Profiler:** Tool collecting execution metrics and traces.
- **Instrumentation:** Markers or counters added to make behavior observable.
- **NVTX:** NVIDIA Tools Extension API used here to label PCR stages.
- **Nsight Compute:** NVIDIA per-kernel metric profiler.
- **Nsight Systems:** NVIDIA timeline profiler for CPU, CUDA, NVTX, and system
  interactions.
- **CUDA event:** Device-timeline marker used for GPU elapsed-time measurement.
- **Compute Sanitizer:** NVIDIA correctness-tool suite for memory, races, and
  synchronization depending on selected tool.
- **Baseline:** Preserved reference implementation or measurement.
- **Regression:** Correctness or performance becomes worse relative to baseline.
- **Speedup:** Baseline time divided by candidate time; greater than one is faster.
- **Crossover:** Change in which implementation is faster.
- **Crossover point:** Workload boundary near that change.
- **End-to-end latency:** Host-visible time for all operations in a defined request.
- **Kernel-only latency:** Time enclosing only selected device kernels.
- **Amortization:** Spreading fixed cost across more useful work.
- **Bottleneck:** Resource currently limiting performance.
- **Bottleneck migration:** Optimization shifts the limiting resource elsewhere.
- **Optimization:** Change intended to improve a defined metric while preserving
  requirements.
- **Premature optimization:** Optimizing without evidence or before correctness and
  relevant baselines.
- **Performance portability:** Maintaining useful performance across environments,
  often through variants and dispatch.

### Numerical and tridiagonal terms

- **Flattened indexing:** Mapping multidimensional logical indices into one linear
  index; here `system=global/N`, `equation=global%N`.
- **Batch:** Collection of independent systems processed as one request.
- **Batch size (`B`):** Number of independent tridiagonal systems.
- **System size (`N`):** Equations/unknowns per system.
- **Tridiagonal matrix:** Matrix with nonzeros only on main, lower, and upper
  diagonals.
- **Thomas algorithm:** Sequential `O(N)` specialized tridiagonal solver.
- **Parallel Cyclic Reduction (PCR):** Log-stage elimination algorithm exposing
  per-equation parallelism.
- **PCR stage:** One update of all equations using a fixed neighbor offset.
- **Offset:** Neighbor distance for a PCR stage: 1, 2, 4, and so on.
- **Dependency radius:** Furthest original input that can influence an output
  after fused stages; it is seven after offsets 1,2,4.
- **Elimination:** Algebraically removing coupling to an unknown.
- **Forward elimination:** Thomas pass that removes the lower diagonal.
- **Back substitution:** Reverse Thomas pass recovering unknowns.
- **Diagonal system:** Each equation contains one unknown, so `x_i=d_i/b_i`.
- **Lower/main/upper diagonal:** Three coefficient bands `a`, `b`, and `c`.
- **RHS:** Right-hand-side vector `d`.
- **Alpha/Beta:** PCR scale factors used to eliminate left/right neighbors.
- **Diagonal dominance:** Main-diagonal magnitude exceeds off-diagonal row sum;
  benchmark fixtures use this property.
- **Numerical stability:** Sensitivity of computed results to rounding and input
  perturbations.
- **Floating-point:** Finite binary representation of real-number approximations.
- **FP32:** IEEE-754 single precision, typically 32 bits.
- **FP64:** IEEE-754 double precision, 64 bits; tridiagonal code uses `double`.
- **Precision:** Amount/range of representable numerical information.
- **Rounding error:** Difference introduced when an exact value is rounded to a
  representable floating-point value.
- **Determinism:** Same configured execution produces the same result/order;
  these one-thread-per-equation stages avoid atomic ordering nondeterminism.

## 26. Things I should not say in an interview

| Misconception | Better statement |
|---|---|
| “Higher occupancy always means faster.” | Occupancy helps hide latency until enough warps are available; registers/shared memory may lower occupancy while reducing total time. |
| “Shared memory is always faster than global memory.” | Shared access is low latency, but staging, halos, barriers, redundancy, and occupancy costs can outweigh saved global traffic. |
| “`O(N)` is always faster than `O(N log N)`.” | Asymptotic work omits span and hardware parallelism; Thomas has lower work, PCR has much shorter parallel span. |
| “The GPU is faster than the CPU.” | Performance depends on algorithm, shape, batching, residency, transfer scope, and hardware. |
| “A memory-bound kernel should automatically use shared memory.” | Memory-bound identifies a limit; shared memory is one hypothesis whose reuse and costs must be measured. |
| “More parallelism always makes code faster.” | Parallel work has launch, synchronization, communication, and resource overhead. |
| “Kernel time equals application time.” | Allocation and H2D/D2H can dominate end-to-end latency. |
| “A profiler recommendation is the correct optimization.” | A recommendation suggests an experiment; only controlled measurements validate it. |
| “A high cache-hit rate proves memory is fast.” | Hit rate needs request volume, bandwidth, latency, and duration context. |
| “Low occupancy caused every small-grid slowdown.” | Occupancy and total grid size differ; a grid can be too small even if theoretical occupancy is high. |
| “Fusion is always good because it removes launches.” | Fusion may increase registers, shared memory, barriers, redundant computation, and code size. |
| “The fastest measured point defines the dispatch rule.” | Noise and overfitting require repeats, tie bands, and validation across a grid. |
| “Zero Compute Sanitizer errors prove correctness.” | It supports memory safety; references and algorithmic tests prove numerical behavior. |
| “True batching means every system finishes simultaneously.” | It exposes all systems in one flattened grid; hardware schedules blocks over time. |
| “Persistent workspace means persistent kernel.” | Only allocations persist here; each solve still launches kernels. |

## 27. Interview questions and answers

### Foundations

1. **What is a tridiagonal system?**  A linear system whose matrix has nonzero
   values only on the main diagonal and the immediately lower and upper
   diagonals. That structure permits specialized solvers with much less work and
   storage than dense Gaussian elimination. *(Tests recognition of structure.)*

2. **What is a warp?**  A group of 32 CUDA threads scheduled together on current
   NVIDIA GPUs. Threads have their own logical state, but a warp issues common
   instructions under lane masks. Divergent branches serialize paths.

3. **What does occupancy mean?**  Resident active warps divided by the hardware
   maximum per SM. It indicates potential latency-hiding capacity, not speed by
   itself. I also inspect eligible warps, grid size, stalls, and duration.

4. **Why is Thomas efficient?**  It exploits tridiagonal structure for `O(N)`
   work, regular contiguous access, and small constants. Its limitation for GPUs
   is an `O(N)` dependency chain in elimination and back substitution.

5. **Why did PCR make sense on a GPU?**  PCR raises work to `O(N log N)` but
   reduces span to `O(log N)`. Within each stage all equations are independent,
   matching one-thread-per-equation SIMT execution.

6. **What is work versus span?**  Work is total operations; span is the longest
   dependent path. Parallel performance depends on both. PCR accepts more work
   to expose much more parallelism than Thomas.

7. **Why are PCR offsets powers of two?**  Each stage eliminates current
   neighbors and doubles the distance to remaining couplings. Offsets 1,2,4,...
   remove all off-diagonal dependencies in `ceil(log2 N)` stages.

8. **Why use ping-pong buffers?**  Every output in a stage must read a consistent
   previous-stage snapshot. Separate current and next arrays prevent one thread's
   write from corrupting another's input; pointer swaps avoid copying arrays.

### CUDA design

9. **How is a batched equation mapped to CUDA?**  Flatten `B*N` entries, compute
   `system=global/N` and `equation=global%N`, then form neighbors relative to
   `system*N`. Logical equation checks stop accesses crossing systems.

10. **Why did true batching help?**  The serial baseline launched roughly
    `B*log2(N)` stages. True batching launches about `log2(N)` stages across all
    systems, reducing launch overhead and exposing system-level parallelism.

11. **Why not just parallelize Thomas?**  Its forward and reverse recurrences
    carry dependencies. Batched systems offer inter-system parallelism, but PCR
    also exposes intra-system parallelism. More advanced parallel Thomas-like
    decompositions exist, but were outside this experiment.

12. **What makes non-power-of-two PCR safe here?**  The loop continues while
    `offset<N`, and each thread independently guards left and right neighbors.
    No padded mathematical system is assumed.

13. **What is kernel-launch overhead?**  Fixed host/runtime/device scheduling cost
    to start a grid. It matters when kernels do little work, which motivated true
    batching and later fusion.

14. **What is the difference between persistent allocation and a persistent
    kernel?**  Persistent allocation reuses device buffers across ordinary
    launches. A persistent kernel stays resident and processes multiple work
    items without relaunch; this project implements only the former.

15. **Why use RAII for device buffers?**  It ties `cudaFree` to C++ object
    lifetime, prevents leaks on exceptions, expresses unique ownership, and
    makes workspace movement explicit.

### Performance reasoning

16. **What is memory-bound versus compute-bound?**  A memory-bound kernel's
    limiting resource is data movement; a compute-bound kernel saturates an
    arithmetic pipeline. Nsight showed early global stages near 90% DRAM, while
    the large fused kernel shifted to about 89% SM and 42% DRAM utilization.

17. **What is a scoreboard stall?**  A warp waits because an operand producer has
    not completed. Long scoreboard commonly points to L1TEX/global memory;
    short scoreboard/MIO often points to shared-memory or similar dependencies.

18. **Why did shared memory fail to improve performance consistently?**  It
    reduced some global reads but added cooperative loads, halos, barriers,
    redundant work, and shared-resource pressure. It also retained full global
    writes between every logical stage, and hardware caches already captured
    some locality.

19. **Why did fusion succeed at `4096x512`?**  It removed two launches and two
    intermediate global-memory round trips while a large grid amortized shared
    setup and barriers. The fused early kernel took about 1.06 ms versus roughly
    1.4 ms for three separate early global stages.

20. **Why was fusion workload-dependent?**  Small grids cannot keep the GPU busy
    long enough to amortize the fused kernel's shared memory, halo computation,
    and barriers. The `1024x32` profile had only 0.75 waves/SM.

21. **What is the difference between occupancy and grid size?**  Occupancy is
    resident warps per SM relative to capacity. Grid size is total blocks. High
    theoretical occupancy cannot help if too few blocks exist to fill all SMs.

22. **How do you decide what to optimize?**  Define a representative scope,
    establish correctness and a baseline, measure, profile the dominant case,
    form one falsifiable hypothesis, change one variable, and remeasure. Preserve
    negative results.

23. **Why use median?**  It reduces sensitivity to occasional long outliers.
    I still inspect mean/min/max, repeat boundary points, use a tie band, and do
    not treat one median as universally representative.

24. **How do you protect against benchmark noise?**  Warmups, repeated CUDA-event
    samples, controlled shapes/data, identical timing scopes, a ±2% tie band,
    longer boundary reruns, and documentation of conflicting short-duration
    results.

25. **Why separate kernel-only and end-to-end timing?**  Kernel timing explains
    device execution. End-to-end timing answers user-visible cost and may be
    dominated by allocation or transfers. Mixing them obscures the bottleneck.

26. **What is the importance of workload shape?**  `N` controls stages and
    within-system dependencies; `B` controls independent systems; `N*B` controls
    grid work. Equal products can still differ in stage count, so product is a
    useful but intentionally simple proxy.

### Dispatch and portability

27. **Why build adaptive dispatch?**  Global and fused paths win in different
    regimes. A measured dispatcher captures stable large-workload wins without
    deleting the robust baseline or forcing fused regressions.

28. **What rule does this repository use?**  Fused when
    `N*B >= 786,432`, otherwise global. It is versioned as
    `adaptive_work_v1:n_times_batch_gte_786432` and recorded in CSV output.

29. **What are false fusion and false global?**  False fusion selects fused where
    decisive data favored global; false global misses a fused win. The chosen
    rule prioritized zero observed false-fusion cases on the dense grid.

30. **Why not use the oracle?**  The oracle knows every measured winner after the
    fact. Runtime dispatch lacks that future measurement, and memorizing points
    overfits noise. A simple explainable rule generalizes more safely.

31. **How would this change on another GPU?**  SM count, caches, DRAM bandwidth,
    shared capacity, clocks, and launch overhead change the crossover. I would
    rerun the dense grid, repeat its boundary, analyze false choices and aggregate
    time, then publish a new versioned threshold.

32. **How would you validate the threshold on another architecture?**  Keep code,
    inputs, scopes, warmups, and iterations fixed; capture hardware/toolchain
    metadata; compare global/fused pairs; use the tie rule; validate the candidate
    on held-out or repeated shapes; retain per-architecture provenance.

33. **How do you distinguish kernel optimization from algorithm optimization?**
    Kernel optimization changes implementation while preserving the algorithm,
    such as staging or fusion. Algorithm optimization changes the mathematical
    procedure, such as Thomas versus PCR. True batching changes scheduling;
    fused stages preserve PCR updates.

34. **How would you port this to Triton?**  Express the flattened program IDs,
    masked per-system neighbor loads, and stage launches first. Establish global
    parity, then explore block-local fused offsets with static block shapes.
    Validate FP64 support/performance, non-power-of-two masks, and generated IR.

35. **How might this translate to another accelerator?**  Preserve the algorithm
    and dependency graph, then remap stages to that platform's cores, vector
    units, on-chip SRAM, DMA, and launch/runtime model. Re-derive the cost model;
    CUDA's block/warp assumptions are not universal.

36. **What would you optimize next?**  First validate across GPUs and applications.
    Then profile whether a fourth fused stage, register/shared balance, CUDA
    Graphs, or warp-level methods addresses a measured limit. I would not stack
    all changes before isolating each effect.

## 28. “Tell me about this project” answers

### 30 seconds

I built and measured a tridiagonal-solver experiment comparing CPU Thomas with
CUDA Parallel Cyclic Reduction. The major gain came from exposing both equation
and batch parallelism, then separating allocation, transfer, and device-resident
costs. Profiling showed early stages were memory-bound. A shared-memory attempt
did not help consistently, but fusing offsets 1, 2, and 4 helped large workloads.
I finished with a measured, versioned dispatch rule that improved aggregate grid
time about 5.5% over always-global while preserving zero observed false-fusion
regressions among decisive points.

### 60 seconds

I treated tridiagonal solving as an algorithm-and-hardware study. Thomas is
`O(N)` but sequential; PCR does `O(N log N)` work with `O(log N)` parallel span.
I implemented CPU references and CUDA PCR, then discovered that launching a full
solve per system hid the algorithm's potential. Flattening `B*N` equations cut
stage launches from roughly `B log N` to `log N`. I added persistent workspaces
and device-resident timing to separate allocation and PCIe costs. Nsight showed
early global stages near 90% DRAM utilization. Caching individual stages in
shared memory was neutral or slower, so I profiled again and fused the first
three stages, keeping 17,280 bytes of intermediate state per block. That improved
`4096x512` about 10.4% but remained workload-dependent. A 100-point study led to
the simple `N*B >= 786,432` fused/global dispatcher. Everything retained CPU and
global-GPU references, awkward-size tests, labeled CTest suites, and sanitizer
validation.

### Two-minute technical version

The core comparison is work versus span. Thomas has linear work and span because
both passes are recurrences. PCR uses offsets 1,2,4,..., performs linear work per
stage, and has logarithmically many parallel stages. My first CUDA baseline used
one thread per equation but a host loop per system. That issued about
`B*ceil(log2 N)` launches. I flattened all systems and decoded system/equation
from one global index, reducing it to `ceil(log2 N)` launches while guarding
system boundaries.

I then decomposed cost. One-shot timing included nine allocations, transfers,
kernels, and cleanup. A RAII workspace reused buffers. A device-resident mode
kept immutable coefficient snapshots on GPU and timed D2D reset separately from
the solve. This made profiling meaningful for an accelerator pipeline.

NVTX labeled every stage. At `4096x512`, offsets through 256 were around 90–92%
DRAM utilization with long-scoreboard waits. A shared/global hybrid cached each
stage through offset 256 but was about 2.1% slower at the target because halos,
barriers, and shared work did not remove inter-stage global traffic. The next
hypothesis fused offsets 1,2,4. Their combined dependency radius is seven, so a
256-output block loads 270 equations and uses two four-array FP64 shared states:
17,280 bytes. The large fused kernel shifted from DRAM to about 89% SM utilization
and improved the full solve about 10.4%.

Small grids did not consistently benefit, so I measured 100 shape pairs, treated
±2% as tied, reran the boundary, and implemented an auditable threshold. The
lesson was not “shared memory is fast”; it was to measure the whole execution
structure, accept failed hypotheses, and dispatch among preserved variants.

### Five-minute walkthrough outline

1. Draw the tridiagonal matrix and contrast Thomas's recurrence with PCR stages.
2. Show `system=global/N`, `equation=global%N`, and explain boundary safety.
3. Quantify launch-count reduction from host-loop to true batch.
4. Draw one-shot, reusable, and resident timelines; identify what each excludes.
5. Present Nsight evidence: global early stages near 90% DRAM and long scoreboard.
6. Explain why single-stage shared memory was a hypothesis and why it failed.
7. Derive fused radius `1+2+4=7` and footprint `2*4*270*8=17,280` bytes.
8. Compare the large profile (1.06 ms fused early kernel, 89% SM) with the small
   underfilled profile (128 blocks, 0.75 waves/SM).
9. Explain the 100-point dispatcher and its conservative error tradeoff.
10. Close with validation and the general lesson: change one variable, profile
    winners and losers, preserve baselines, and version hardware-specific policy.

## 29. Debugging story: the wrong right-side coefficient

During development, a right-neighbor PCR update wrote the new coefficient to
`next_a` instead of `next_c`. The surviving commits contain the corrected form:

```cpp
next_c[i] = beta * c[right];
```

The symptom was a CUDA/CPU result mismatch even though indexing and launch error
checks passed. Final-output inspection did not identify whether initialization,
one stage, pointer swapping, or final division was responsible. Intermediate
state was therefore captured and compared after each PCR stage. The initial
state matched; the first bad stage showed the right-side coupling appearing in
the lower-coefficient output while the upper coefficient remained wrong. That
narrowed the fault to the `beta` branch rather than later swaps or solve logic.

The fix was to write `beta*c[right]` to `next_c`. Then stage-by-stage comparison,
final CPU/CUDA comparison, non-power-of-two cases, and memory checks were rerun.
The bug was fixed during development and is not present as a faulty surviving
commit, so it should be described as a debugging episode, not as repository
history that reviewers can check out.

Interview framing: I owned the faulty update, resisted patching final outputs,
instrumented the algorithm at its stage boundaries, found the first divergence,
mapped it back to one algebraic coefficient, fixed the root cause, and expanded
tests around the mechanism that failed. The lesson is to exploit algorithmic
invariants and compare the earliest observable state, especially in parallel
code where downstream errors amplify.

## 30. Optimization story (STAR)

**Situation:** The CUDA PCR solver was correct, and true batching exposed useful
parallelism, but each stage moved four coefficient arrays through global memory.
Device-resident profiling showed early stages near 90–92% DRAM utilization and
large L1TEX/long-scoreboard waits.

**Task:** Improve device solve time with evidence while keeping global-only PCR,
timing semantics, precision, and correctness baselines intact.

**Action:** I added deterministic NVTX labels for each offset and profiled the
target workload. I tested a shared-memory stage path through offset 256. It was
correct but about 2.1% slower at `4096x512`, so I preserved and documented the
negative result. The remaining traffic was inter-stage, so I fused offsets 1,2,4
using a seven-equation halo and 17,280 bytes of shared state. I profiled both a
large winner and small loser, then measured a 100-point grid, applied a ±2% tie
band, repeated threshold cases, and centralized a versioned dispatch rule.

**Result:** The documented large solve improved about 10.4%; other large points
improved roughly 9–13%, while small results remained noisy or regressed. The
adaptive rule improved aggregate measured grid runtime about 5.5% over
always-global with zero false-fusion cases among decisive points. Full CPU/GPU
tests and Compute Sanitizer passed. The strongest result was the method: a
failed hypothesis directly informed a more effective design.

## 31. Potential follow-up research

| Direction | Why it might help | Risk / evidence required first |
|---|---|---|
| Cross-GPU policy validation | Reveal architecture-dependent crossover | Requires identical dense grids and provenance; do not reuse RTX threshold blindly |
| Fuse offset 8 | Remove another launch and round trip | Radius becomes 15 and footprint/work grow; profile whether offset 8 remains traffic-limited |
| Register/shared balance | Keep repeatedly used scalars closer | Register pressure may reduce residency; inspect compiler allocation and spills |
| Warp shuffles | Exchange near-neighbor values without shared memory for small offsets | Cross-warp and system-boundary handling complicate logic; justify with shared/MIO stalls |
| Cooperative groups | Express subgroup/block coordination | Added constraints and complexity; needed only if synchronization structure benefits |
| Persistent kernel | Remove repeated launches across stages/solves | Grid-wide coordination and scheduling become hard; launch overhead must still dominate |
| CUDA Graphs | Reduce repeated host launch overhead without changing kernels | Helps repetitive fixed workflows, less useful if kernels dominate; measure submission cost |
| Triton implementation | Explore portable block programs and autotuning | FP64/backend behavior and inter-stage launches may differ; establish exact parity first |
| Trainium/NKI implementation | Test the same work/span tradeoff on another accelerator | Must redesign around its execution and SRAM/DMA model; CUDA mapping is not transferable verbatim |
| Mixed precision | Increase throughput/bandwidth efficiency | Numerical error and stability risks; needs application tolerances and reference studies |
| Autotuning | Select thresholds/configurations per device | Tuning cost and overfitting; use held-out shapes and versioned caches |
| Analytical cost model | Explain launches, traffic, stages, and saturation | Simplifications may miss caches and scheduling; validate against measurements |
| Real application fixtures | Test distributions, conditioning, and pipeline residency | Harder reproducibility; first identify representative domains and correctness criteria |

## 32. Practice exercises

1. How many PCR stages execute for `N=33`? List their offsets.
2. How many stages execute for `N=4096`?
3. Derive the fused dependency radius after offsets 1,2,4.
4. Calculate the fused shared-memory footprint for a 256-output block, seven-value
   halo on both sides, two states, four FP64 arrays.
5. Calculate total equations and blocks for `N=1024`, `B=32`, block size 256.
6. Calculate total equations and blocks for `N=4096`, `B=512`.
7. Which adaptive path is selected for `1536x511`? For `1536x512`?
8. A kernel reports 91% DRAM and 55% SM throughput with long-scoreboard stalls.
   What is the first bottleneck hypothesis, and what must you do before changing code?
9. A fused kernel reports 89% SM, 42% DRAM, high MIO and barrier stalls. Has the
   bottleneck moved? What evidence determines whether this is acceptable?
10. Explain why 83% theoretical occupancy can outperform 100% theoretical
    occupancy.
11. Compare the launch counts for `N=1024`, `B=512` in serial-host-loop and
    true-batched execution, including the final solve launches.
12. Explain why device-resident timing should exclude D2D reset in this benchmark.
13. A candidate is 1.01x faster than global. How is it classified by the policy
    study, and why?
14. Given global 6.573568 ms and fused 5.956096 ms, calculate speedup and percent
    latency reduction.
15. Why might results differ on a GPU with more SMs but similar memory bandwidth?

<details>
<summary>Exercise answers</summary>

1. Six stages: offsets `1,2,4,8,16,32`; the loop stops at 64 because `64>=33`.
2. Twelve stages: offsets 1 through 2048.
3. `1+2+4=7`. Each stage consumes prior-stage values at its offset, so the
   original-input influence expands cumulatively.
4. Span `256+2*7=270`; bytes `2*4*270*8=17,280`.
5. `1024*32=32,768` equations and `32,768/256=128` blocks.
6. `4096*512=2,097,152` equations and 8,192 blocks.
7. `1536*511=784,896`, so global. `1536*512=786,432`, so fused.
8. The current hypothesis is off-chip memory latency/bandwidth pressure. Inspect
   source-correlated accesses, cache behavior, coalescing, traffic volume, and
   compare a controlled candidate; “use shared memory” is not yet a conclusion.
9. Yes, pressure appears shifted from DRAM toward compute/shared synchronization.
   It is acceptable only if total duration for the same work decreases and
   correctness/resources remain acceptable.
10. The lower-occupancy kernel may eliminate launches and global traffic, doing
    less costly work overall. Enough warps can hide latency without reaching the
    maximum occupancy ratio.
11. `ceil(log2 1024)=10`. Serial: `512*10=5,120` stage launches plus 512 solves.
    True batch: 10 stage launches plus one solve.
12. The scope asks for PCR solve cost given valid mutable inputs already present.
    D2D reset is required for repeated measurement but is a distinct pipeline
    operation, so a separate row keeps both costs visible.
13. Tie: 1.01 lies inside 0.98–1.02. This avoids building policy from noise-sized
    differences.
14. Speedup `6.573568/5.956096 ≈ 1.104x`; latency reduction
    `(6.573568-5.956096)/6.573568 ≈ 9.39%`. “10.4% faster” and “9.4% less time”
    are related but not identical percentages.
15. Small grids may become even more underfilled, changing fusion crossover;
    bandwidth per SM and cache/resource ratios also change. Recalibration is
    required.

</details>

## 33. Flashcards

**Q: What are the three nonzero bands in a tridiagonal matrix?**  
**A:** Lower, main, and upper diagonals.

**Q: What is the Thomas algorithm's work complexity?**  
**A:** `O(N)`.

**Q: What is the Thomas algorithm's span?**  
**A:** `O(N)` because of forward and backward recurrences.

**Q: What is PCR's work complexity?**  
**A:** `O(N log N)`.

**Q: What is PCR's ideal parallel span?**  
**A:** `O(log N)`.

**Q: What are the first PCR offsets?**  
**A:** 1, 2, 4, 8, 16, and so on.

**Q: What happens after all PCR stages?**  
**A:** The system is diagonal, so `x[i]=d[i]/b[i]`.

**Q: Why use ping-pong arrays?**  
**A:** To read one immutable stage while writing the next.

**Q: What is one CUDA thread responsible for here?**  
**A:** One equation in one tridiagonal system.

**Q: How is the system index decoded?**  
**A:** `system=global/N`.

**Q: How is the equation index decoded?**  
**A:** `equation=global%N`.

**Q: What block size do these PCR kernels use?**  
**A:** 256 threads.

**Q: What is a warp?**  
**A:** A 32-thread NVIDIA scheduling group.

**Q: What is an SM?**  
**A:** A Streaming Multiprocessor that schedules and executes blocks/warps.

**Q: What is true batching?**  
**A:** Launching a stage once over flattened equations from all systems.

**Q: How does true batching change stage launch count?**  
**A:** From about `B log2 N` to `log2 N`.

**Q: What is kernel-launch overhead?**  
**A:** Fixed host/runtime/device cost to schedule a grid.

**Q: What does RAII provide?**  
**A:** Resource ownership tied to C++ object lifetime.

**Q: What persists in `CudaPcrBatchedWorkspace`?**  
**A:** Device buffers, not a running kernel.

**Q: What is H2D?**  
**A:** Host-to-device transfer.

**Q: What is D2H?**  
**A:** Device-to-host transfer.

**Q: What is D2D reset?**  
**A:** Copying immutable device coefficients back into mutable working buffers.

**Q: What does device-resident timing include?**  
**A:** PCR stage kernels and final solve only.

**Q: Why use CUDA events?**  
**A:** To measure elapsed time on the device timeline.

**Q: Why use warmups?**  
**A:** To reduce cold-start and initial clock/runtime effects.

**Q: Why report median?**  
**A:** It is less sensitive to isolated outliers.

**Q: What is a long-scoreboard stall?**  
**A:** Waiting on a high-latency dependency, commonly L1TEX/global memory.

**Q: What is a short-scoreboard/MIO stall?**  
**A:** Waiting commonly associated with shared-memory or MIO dependencies.

**Q: What is a CTA barrier stall?**  
**A:** A warp waits for sibling warps at a block-wide barrier.

**Q: What does L1TEX include?**  
**A:** L1/texture/load-store paths, including ordinary global-memory traffic.

**Q: What does occupancy measure?**  
**A:** Active resident warps relative to the SM maximum.

**Q: Does maximum occupancy guarantee maximum speed?**  
**A:** No.

**Q: What are waves per SM useful for?**  
**A:** Seeing whether the total grid supplies enough block scheduling depth.

**Q: What is coalescing?**  
**A:** Serving adjacent warp memory accesses with efficient transactions.

**Q: What is a halo?**  
**A:** Neighbor data loaded beyond a block's owned outputs.

**Q: What shared-stage cutoff was tested?**  
**A:** Shared through offset 256, global from 512 onward.

**Q: Did the shared/global hybrid consistently win?**  
**A:** No; `4096x512` was about 2.1% slower.

**Q: Which stages are fused?**  
**A:** Offsets 1, 2, and 4.

**Q: What is their combined dependency radius?**  
**A:** Seven equations.

**Q: How many equations does a fused block stage?**  
**A:** 270 for 256 owned outputs.

**Q: What is the fused shared-memory footprint?**  
**A:** 17,280 bytes per block.

**Q: How many shared states are used?**  
**A:** Two, each containing `a,b,c,d`.

**Q: What was the large fused-kernel duration?**  
**A:** About 1.06 ms for the early fused launch at `4096x512`.

**Q: What was its SM throughput?**  
**A:** About 89.01% in the captured profile.

**Q: What was its DRAM throughput?**  
**A:** About 41.77%.

**Q: What made `1024x32` underfilled?**  
**A:** Only 128 blocks, about 0.75 profiler waves/SM for the fused kernel.

**Q: What is the adaptive workload proxy?**  
**A:** Total equations `N*B`.

**Q: What is the adaptive threshold?**  
**A:** 786,432 equations.

**Q: What is the rule identifier?**  
**A:** `adaptive_work_v1:n_times_batch_gte_786432`.

**Q: What tie band was used?**  
**A:** Speedup within ±2%.

**Q: What is false fusion?**  
**A:** Selecting fused where decisive measurements favor global.

**Q: What is false global?**  
**A:** Selecting global where decisive measurements favor fused.

**Q: What is an oracle policy?**  
**A:** A retrospective best-per-point selector.

**Q: What aggregate improvement did adaptive report?**  
**A:** About 5.5% versus always-global on the measured grid.

**Q: How many false-fusion cases did it have among decisive grid points?**  
**A:** Zero.

**Q: What is the key shared-memory lesson?**  
**A:** A memory bottleneck makes shared memory a hypothesis, not a guaranteed fix.

**Q: What is the key fusion lesson?**  
**A:** Eliminating launches and intermediate traffic helps only when enough work
amortizes added resources and synchronization.

**Q: What does Compute Sanitizer memcheck add?**  
**A:** Detection of illegal and out-of-bounds CUDA memory accesses.

**Q: What does it not replace?**  
**A:** CPU references, numerical tests, race checks, and algorithmic validation.

**Q: What is the experimental method in one line?**  
**A:** Correct baseline, measure, profile, hypothesize, isolate, validate, repeat.

## 34. Final cheat sheet

### CUDA hierarchy

```text
host launches grid
grid -> blocks
block -> warps (32 threads)
warps execute on an SM
this project: 256 threads/block, one thread/equation
```

### Memory hierarchy

```text
registers: thread-private, fastest, scarce
shared: block-local, explicit, barrier-coordinated
L1/L1TEX: per-SM hardware-managed path/cache
L2: GPU-wide cache
global DRAM: large/high-bandwidth/high-latency
host memory: across PCIe for discrete GPU
```

### PCR versus Thomas

| | Thomas | PCR |
|---|---|---|
| Work | `O(N)` | `O(N log N)` |
| Span | `O(N)` | `O(log N)` |
| Strength | Low CPU work | GPU parallelism |
| Weakness | Dependency chain | More arithmetic/traffic/stages |

### Execution progression

```text
serial host loop -> too many launches
true batch       -> one launch/stage over B*N
reusable         -> remove repeated allocation
device resident  -> isolate solve from PCIe
shared hybrid    -> reasonable hypothesis, no consistent win
fused 1/2/4      -> remove launches + intermediate global traffic
adaptive         -> retain workload-dependent winners
```

### Profiler questions

1. Is duration actually bad for the relevant scope?
2. Is the grid large enough? Check blocks and waves/SM.
3. Which resource is near saturation: DRAM, L1TEX, L2, or SM compute?
4. What do long scoreboard, MIO, and barrier stalls say about dependencies?
5. Are registers/shared memory limiting resident blocks?
6. Did the candidate reduce total time, or merely move the bottleneck?

### Numbers to remember

- Shared hybrid: offsets `<=256`; `4096x512` about **2.1% slower**.
- Fused offsets: **1, 2, 4**; radius **7**.
- Fused block: **256 outputs**, **270 staged equations**.
- Shared footprint: **17,280 bytes/block**.
- Large fused profile: **1.06 ms**, **89.01% SM**, **41.77% DRAM**.
- Documented full `4096x512`: **6.573568 -> 5.956096 ms**, **1.104x**.
- Small fused profile: `1024x32`, **128 blocks**, **0.75 waves/SM**.
- Dispatch: fused when **`N*B >= 786,432`**.
- Rule ID: **`adaptive_work_v1:n_times_batch_gte_786432`**.
- Policy result: about **5.5% aggregate improvement**, **zero decisive
  false-fusion cases** on the measured grid.

### Interview lessons

- Discuss work and span, not Big-O work alone.
- State timing scope before quoting speedup.
- Separate occupancy from grid utilization.
- Treat profiler guidance as a hypothesis generator.
- Preserve failed experiments and baselines.
- Profile winners and losers.
- Validate non-power-of-two sizes, boundaries, reuse, and changed inputs.
- Version hardware-specific dispatch policy.
- Say what is measured, what is inferred, and what remains unknown.

### Red flags

Avoid absolute claims: “GPU is faster,” “shared memory always wins,” “higher
occupancy is better,” “kernel time is end-to-end time,” or “one benchmark proves
portability.” The defensible answer always names the algorithm, shape, hardware,
scope, evidence, and tradeoff.
