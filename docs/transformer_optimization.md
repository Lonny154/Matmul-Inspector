# Transformer Projection Optimization Case Study

## Overview

This case study documents a performance investigation of a miniature Transformer block on an NVIDIA GeForce RTX 4060 Ti.

The goal was not simply to write a faster matrix multiplication kernel. The objective was to follow a realistic GPU optimization workflow:

1. Profile an end-to-end Transformer workload.
2. Identify the dominant GPU operations.
3. Isolate a representative GEMM.
4. Develop and tune a custom Triton Tensor Core kernel.
5. Inspect compiler IR and PTX to understand the generated implementation.
6. Validate kernel-level improvements with repeatable measurements.
7. Integrate optimized kernels back into the Transformer workload.
8. Measure whether local kernel improvements translated into end-to-end gains.

The investigation ultimately showed that Transformer GEMM performance is strongly shape-dependent. A configuration that performed well for one projection did not necessarily generalize to another, and kernel-level improvements did not automatically produce an end-to-end Transformer speedup.

---

## Workload

The benchmark uses a decoder-style Transformer block with:

- Batch size: 1
- Sequence length: 512
- Hidden size: 768
- Attention heads: 12
- Head dimension: 64
- MLP hidden size: 3072
- Data type: FP16

The block contains:

- LayerNorm
- Combined QKV projection
- Flash attention
- Attention output projection
- LayerNorm
- MLP FC1
- GELU
- MLP FC2
- Residual connections

The principal GEMM shapes are:

```text
QKV:
(512 x 768) @ (768 x 2304)

FC1:
(512 x 768) @ (768 x 3072)

FC2:
(512 x 3072) @ (3072 x 768)
```

NVTX ranges were added around each major operation so Nsight Systems could attribute GPU time to individual Transformer stages.

---

## Baseline Profiling

The original PyTorch workload was profiled with Nsight Systems.

The approximate GPU-time distribution was:

```text
MLP FC2                     ~25.8%
MLP FC1                     ~25.1%
QKV projection              ~21.5%
Flash attention             ~10.1%
Attention output projection  ~9.0%
```

Approximately 70% of the measured GPU kernel time was therefore spent in GEMM-based projections.

FC1 was selected as the first optimization target because it was both a major hotspot and a straightforward isolated GEMM:

```text
(512 x 768) @ (768 x 3072)
```

---

## Isolated FC1 Benchmark

An isolated benchmark was created for the exact FC1 shape.

The PyTorch/cuBLAS baseline was approximately:

```text
~64 us
~38 TFLOP/s
```

This closely reproduced the FC1 timing observed inside the Transformer workload, making it a suitable standalone optimization target.

A custom Triton GEMM was then implemented using:

- FP16 inputs
- FP32 accumulation
- Tensor Core MMA operations
- tiled shared-memory staging
- configurable block dimensions
- configurable warp count
- configurable pipeline stage count

Correctness was verified against PyTorch throughout the tuning process.

---

## Nsight Compute Analysis

The original cuBLAS FC1 kernel exhibited substantial register pressure.

Representative measurements included:

```text
Registers per thread:       ~254
Achieved occupancy:         ~14%
Active warps per SM:        ~7
Compute throughput:         ~41%
```

The kernel was not primarily DRAM-bandwidth bound. Instead, low occupancy and execution-pipeline stalls limited the number of eligible warps available to the scheduler.

The Triton implementation allowed direct exploration of this resource tradeoff.

---

## Tile and Pipeline Tuning

Multiple tile shapes were evaluated, including:

```text
32 x 64
32 x 128
64 x 64
64 x 128
128 x 64
128 x 128
```

Additional experiments varied:

- BLOCK_K
- number of warps
- pipeline stages
- CTA grouping

The best overall FC1 configuration was:

```text
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 32
num_warps = 4
num_stages = 3
GROUP_SIZE_M = 1
```

Compared with the original cuBLAS profile, the Triton kernel substantially reduced register pressure and improved occupancy.

Representative Triton measurements were:

```text
Registers per thread:       ~121
Dynamic shared memory:      ~24.6 KB
Achieved occupancy:         ~27%
```

The optimized FC1 kernel reached approximately:

```text
~60 us
```

in the integrated Nsight Systems profile, compared with roughly 64 us for the earlier PyTorch/cuBLAS FC1 profile.

This represented an approximate 6% kernel-level reduction in FC1 latency under those profiling conditions.

---

## PTX Inspection

The generated Triton PTX was inspected to verify the hardware execution path.

The kernel contained instructions such as:

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

This confirmed that the Triton `tl.dot` operation lowered to Tensor Core MMA instructions.

The generated implementation therefore followed the expected path:

```text
Global memory
    ↓
Shared-memory staging
    ↓
ldmatrix
    ↓
Tensor Core MMA
    ↓
FP32 accumulator fragments
```

PTX inspection also exposed shared-memory stores and loads in the output epilogue:

```text
st.shared.v4.b32
ld.shared.v4.b32
st.global.v4.b32
```

These accesses became the basis for a deeper investigation of the result-layout conversion.

---

## TTGIR Layout Analysis

Triton's GPU intermediate representation was inspected to understand why the output epilogue used shared memory.

The key operation was:

```text
ttg.convert_layout
```

which converted the Tensor Core accumulator from:

```text
tensor<64x128xf16, #mma>
```

into:

```text
tensor<64x128xf16, #blocked>
```

before the final global-memory store.

The effective path was therefore:

```text
tl.dot
    ↓
tt.dot
    ↓
MMA accumulator layout
    ↓
ttg.convert_layout
    ↓
blocked output layout
    ↓
tt.store
```

The PTX shared-memory shuffle corresponded to this compiler-generated MMA-to-blocked redistribution.

This explained why rewriting the output pointer arithmetic did not eliminate the shared-memory traffic: the expensive conversion was determined by the logical data layout, not merely the pointer expression.

---

## Shared-Memory Layout Experiments

Nsight Compute reported excessive shared-memory wavefronts during the conversion.

Tile experiments showed that the amount of shared-memory overhead depended strongly on the compiler-selected warp layout.

Measured excessive shared-wavefront rates included approximately:

```text
32 x 64    10%
32 x 128    7%
64 x 64    14%
64 x 128    9%
128 x 64   17%
128 x 128  13%
```

TTGIR inspection showed two primary MMA warp distributions:

```text
[1, 4]
[2, 2]
```

Configurations using `[1,4]` generally produced lower shared-memory redistribution overhead than comparable `[2,2]` configurations.

However, reducing shared-memory conflicts did not necessarily improve total GEMM performance.

For example, the 32x128 configuration reduced excessive shared wavefronts to approximately 7%, but its overall kernel runtime was worse than the 64x128 configuration because the smaller tile increased CTA count and memory-side overhead.

This demonstrated an important optimization tradeoff:

```text
Lower register pressure
Higher occupancy
Lower shared-memory conflicts
```

do not automatically imply:

```text
Lower total kernel latency
```

The best kernel configuration was instead the one that balanced these competing factors.

---

## Integration Into the Transformer

The custom FC1 kernel was integrated into the original Transformer workload.

A first round of timing initially appeared to show an end-to-end improvement, but later Nsight profiling revealed that the custom kernel had not been used during every timed iteration.

The benchmark harness was corrected so the Triton path was consistently selected during both warmup and measurement.

This reinforced an important benchmarking lesson:

> End-to-end performance claims must be verified against actual kernel execution, not just application-level timing.

After correction, Nsight Systems confirmed that all FC1 invocations used the custom Triton kernel.

---

## Extending the Kernel to QKV and FC2

The FC1 implementation was refactored into a reusable Triton GEMM launcher and applied to:

```text
QKV:
(512 x 768) @ (768 x 2304)

FC2:
(512 x 3072) @ (3072 x 768)
```

The original FC1 configuration did not generalize well.

FC2 was substantially slower using the reused FC1 configuration, demonstrating that GEMM tuning is shape-specific.

A controlled per-projection autotuning study was therefore introduced.

---

## Projection Autotuning

The autotuner searched 72 configurations per projection across:

- BLOCK_M
- BLOCK_N
- BLOCK_K
- warp count
- pipeline stage count

Every candidate was required to pass numerical validation against PyTorch.

Candidates were measured using alternating CUDA-event benchmarks to reduce bias from GPU clock, thermal, and execution-state changes.

The three strongest candidates were then subjected to longer verification runs.

A configuration was selected only if:

1. All seven verification pairs favored Triton.
2. The median paired PyTorch/Triton ratio was at least 1.02x.
3. Numerical correctness passed.

This prevented a single favorable timing run from being treated as an optimization.

---

## QKV Result

QKV successfully produced a repeatable Triton configuration:

```text
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64
num_warps = 4
num_stages = 2
GROUP_SIZE_M = 1
```

Verification measurements were:

```text
PyTorch median: 59.78 us
Triton median:  55.32 us
Median ratio:   1.0801x
```

The median kernel improvement was therefore approximately 8%.

Additional post-selection studies also favored Triton:

```text
1.0417x
1.1616x
```

This showed that shape-specific tuning could produce a repeatable improvement where simple reuse of the FC1 configuration could not.

---

## FC2 Result

No FC2 configuration passed the verification criteria.

Several candidates appeared promising during the initial search but reversed or became inconsistent under longer measurement.

Rather than selecting a favorable but unstable result, the autotuner rejected all candidates.

This was an important outcome.

The optimization process was designed to distinguish:

```text
fast once
```

from:

```text
reliably faster
```

FC2 remained the largest GPU-time contributor in the fully integrated profile.

---

## Final Integrated Profile

With the selected QKV kernel and the custom Triton projection path enabled, Nsight Systems reported the largest GPU contributors as approximately:

```text
FC2                 ~84 us
FC1                 ~64 us
QKV                  ~49 us
Flash attention      ~26 us
Attention output     ~23 us
```

The complete all-enabled Transformer block did not outperform the PyTorch baseline.

Two clean alternating studies measured:

```text
Study 1:
PyTorch       0.2739 ms
Triton path   0.3246 ms
Ratio         0.8438x

Study 2:
PyTorch       0.3021 ms
Triton path   0.3223 ms
Ratio         0.9373x
```

The QKV improvement was therefore outweighed by FC2 regression and other system-level costs.

---

## Key Findings

### 1. GEMM tuning is shape-specific

A configuration optimized for:

```text
512 x 768 x 3072
```

did not automatically perform well for:

```text
512 x 3072 x 768
```

or:

```text
512 x 768 x 2304
```

Each projection required independent evaluation.

### 2. Kernel-level wins do not guarantee application-level wins

FC1 and QKV both demonstrated competitive or improved kernel-level performance, yet the complete Transformer block did not improve when all custom projections were enabled.

Optimization must therefore be evaluated at both the kernel and workload level.

### 3. Occupancy is only one part of performance

Reducing register pressure substantially increased occupancy, but higher occupancy did not necessarily mean lower kernel latency.

Memory traffic, CTA count, pipeline stalls, shared-memory behavior, and instruction overhead all interacted.

### 4. Shared-memory conflicts were not the dominant bottleneck

The MMA-to-blocked layout conversion could be improved from approximately 17% excessive shared wavefronts to as low as 7%.

However, configurations with fewer conflicts were not necessarily faster.

The experiment demonstrated why profiler warnings should be treated as clues rather than direct speedup estimates.

### 5. Compiler inspection can explain profiler behavior

Following the kernel through:

```text
Triton source
→ TTGIR
→ PTX
→ Nsight
```

made it possible to connect a profiler warning directly to a compiler-generated layout conversion.

### 6. Robust benchmarking matters

Several apparent performance wins disappeared during longer or alternating measurements.

The final tuning procedure therefore required repeated paired measurements and explicit qualification rules before selecting a configuration.

---

## Engineering Lessons

The most important lesson from this project was that GPU kernel optimization is not simply a matter of minimizing one metric.

The practical objective is to balance:

```text
Tensor Core utilization
register pressure
occupancy
memory traffic
shared-memory behavior
warp scheduling
CTA geometry
pipeline depth
launch count
```

while still improving the real workload.

Equally important is knowing when a local optimization has reached diminishing returns.

The FC1 investigation eventually reached a point where further manipulation of shared-memory layouts was unlikely to provide meaningful system-level value. At that point, expanding the analysis to additional Transformer projections provided much more useful information.

---

## Current Status

The project now includes:

- A realistic miniature Transformer workload.
- NVTX-instrumented GPU profiling.
- A reusable Triton Tensor Core GEMM.
- Shape-specific projection configurations.
- Automated QKV/FC2 configuration search.
- Correctness validation against PyTorch.
- Alternating repeated benchmarks.
- Nsight Systems and Nsight Compute analysis.
- TTGIR and PTX inspection.
- Cross-layer reasoning from Transformer operations down to GPU instructions.

The current results establish a repeatable QKV kernel improvement and competitive FC1 performance, while also documenting why FC2 and the complete Transformer block remain optimization opportunities.

Future work can focus on improving the FC2 kernel design, kernel fusion, larger multi-block Transformer workloads, and evaluating whether improvements accumulate at more realistic model scale.