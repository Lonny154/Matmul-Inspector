# Triton GEMM Experiments

## Overview

This document summarizes the Triton GEMM work added to Matmul Inspector after the handwritten CUDA optimization milestone.

The goal of this phase was not simply to implement matrix multiplication in Triton. The larger objective was to understand how Triton exposes kernel-optimization decisions such as:

- tile shape,
- warp count,
- software-pipeline depth,
- register usage,
- shared-memory usage,
- occupancy,
- scheduler efficiency,
- instruction count,
- and GPU cycle consumption.

The experiments progressed from a simple fixed Triton kernel to manual parameter sweeps, autotuning, and finally Nsight Compute analysis of both a deliberately weak configuration and a tuned configuration.

The resulting profiles provide a concrete example of an important GPU-performance principle:

> Higher occupancy does not necessarily imply higher performance.

The tuned Triton kernel uses substantially more registers and shared memory, resulting in much lower occupancy than the smaller kernel, yet it completes the same GEMM with fewer instructions, fewer elapsed cycles, and better scheduler efficiency.

---

## Initial Triton implementation

The first Triton kernel implemented a conventional blocked GEMM for FP32 matrices.

Each Triton program computes one `BLOCK_M × BLOCK_N` output tile. The kernel walks through the reduction dimension in chunks of `BLOCK_K`, loading tiles from matrices A and B and accumulating their dot products into an FP32 accumulator.

The initial fixed configuration was:

```text
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
num_warps = 4
```

For FP32 inputs, the kernel explicitly used:

```python
input_precision="ieee"
```

to avoid silently switching the experiment to TF32-style input precision.

Correctness was tested against `torch.matmul`.

For a 256 × 256 × 256 multiplication:

```text
max absolute error:  4.38690186e-05
mean absolute error: 3.60808167e-06
```

The result passed the configured `torch.allclose` comparison.

---

## Initial Triton vs cuBLAS benchmark

The first fixed Triton kernel was benchmarked against the `torch.matmul` path, which uses NVIDIA's optimized GEMM implementation underneath.

Results:

| Matrix size | Triton GFLOP/s | cuBLAS GFLOP/s | Triton / cuBLAS |
|---|---:|---:|---:|
| 256³ | 2,730.67 | 1,927.53 | 1.417× |
| 512³ | 6,898.53 | 6,553.60 | 1.053× |
| 1024³ | 8,490.49 | 10,729.69 | 0.791× |
| 2048³ | 8,825.61 | 12,510.97 | 0.705× |

The simple 32 × 32 × 32 configuration was competitive at small sizes but fell increasingly behind cuBLAS as matrix size increased.

This suggested that the kernel implementation itself was functional, but the fixed tile configuration was not making efficient use of the GPU for larger workloads.

---

## Tile-size sweep

The next experiment varied the output and reduction tile dimensions while holding other parameters constant.

Configurations tested included:

```text
16 × 16 × 16
32 × 32 × 32
64 × 64 × 32
64 × 64 × 64
128 × 64 × 32
64 × 128 × 32
```

### 1024³

| Tile | Time | Performance |
|---|---:|---:|
| 16×16×16 | 0.726016 ms | 2,957.90 GFLOP/s |
| 32×32×32 | 0.252928 ms | 8,490.49 GFLOP/s |
| 64×64×32 | 0.166912 ms | 12,865.96 GFLOP/s |
| 64×64×64 | 0.216064 ms | 9,939.11 GFLOP/s |
| **128×64×32** | **0.161792 ms** | **13,273.11 GFLOP/s** |
| 64×128×32 | 0.167936 ms | 12,787.51 GFLOP/s |

### 2048³

| Tile | Time | Performance |
|---|---:|---:|
| 16×16×16 | 5.252096 ms | 3,271.05 GFLOP/s |
| 32×32×32 | 1.935360 ms | 8,876.83 GFLOP/s |
| 64×64×32 | 1.277888 ms | 13,443.96 GFLOP/s |
| 64×64×64 | 1.599296 ms | 10,742.14 GFLOP/s |
| **128×64×32** | **1.265664 ms** | **13,573.80 GFLOP/s** |
| 64×128×32 | 1.292288 ms | 13,294.15 GFLOP/s |

The most important finding was that larger tiles substantially improved throughput.

At 1024³, moving from 32×32×32 to 128×64×32 increased throughput from approximately 8.49 TFLOP/s to 13.27 TFLOP/s.

At 2048³, the same change increased throughput from approximately 8.88 TFLOP/s to 13.57 TFLOP/s.

Increasing only the K dimension from 32 to 64 did not help. The 64×64×64 configuration was significantly slower than 64×64×32 at both sizes.

This demonstrated that "larger tile" is not by itself an optimization rule. Tile shape and reduction depth interact with register pressure, shared-memory use, instruction scheduling, and data reuse.

---

## Warp-count sweep

The best tile configuration from the previous experiment was then held fixed:

```text
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
```

The number of warps per Triton program was varied.

### 1024³

| Warps | Performance |
|---:|---:|
| 2 | 5,825.68 GFLOP/s |
| **4** | **13,273.11 GFLOP/s** |
| 8 | 13,189.63 GFLOP/s |

### 2048³

| Warps | Performance |
|---:|---:|
| 2 | 7,228.44 GFLOP/s |
| **4** | **13,695.69 GFLOP/s** |
| 8 | 13,562.83 GFLOP/s |

Two warps were insufficient to execute the tile efficiently.

Moving from two to four warps nearly doubled throughput.

Increasing from four to eight warps provided no additional benefit and slightly reduced performance.

The result illustrates a common GPU-performance pattern: additional parallelism helps until sufficient execution resources are already available, after which added warps may only increase resource pressure or scheduling overhead.

---

## Pipeline-stage sweep

The next experiment varied Triton's `num_stages` parameter while keeping the best tile shape and four warps fixed.

`num_stages` controls software pipelining depth and can allow data movement for future iterations to overlap with current computation.

### 1024³

| Stages | Performance |
|---:|---:|
| 1 | 9,157.87 GFLOP/s |
| 2 | 11,153.21 GFLOP/s |
| **3** | **13,273.11 GFLOP/s** |
| 4 | 11,915.64 GFLOP/s |
| 5 | 11,983.73 GFLOP/s |

### 2048³

| Stages | Performance |
|---:|---:|
| 1 | 13,056.20 GFLOP/s |
| **2** | **13,865.47 GFLOP/s** |
| 3 | 13,628.93 GFLOP/s |
| 4 | 12,671.61 GFLOP/s |
| 5 | 12,681.19 GFLOP/s |

The optimal pipeline depth changed with matrix size.

For 1024³, three stages performed best.

For 2048³, two stages performed best.

This is another example of why fixed hand-selected parameters are often insufficient across different workloads.

---

## Triton autotuning

After the manual parameter sweeps established a reasonable search space, the kernel was converted to use `@triton.autotune`.

The autotuner evaluated several configurations derived from the manual experiments, including:

```text
64×64×32,   4 warps, 2 stages
128×64×32,  4 warps, 2 stages
128×64×32,  4 warps, 3 stages
64×128×32,  4 warps, 2 stages
```

The tuning key used the matrix dimensions:

```python
key=["M", "N", "K"]
```

This allows Triton to select a different kernel configuration for different matrix shapes.

### Autotuned benchmark results

| Matrix size | Triton GFLOP/s | cuBLAS GFLOP/s | Triton / cuBLAS |
|---|---:|---:|---:|
| 256³ | 1,927.53 | 1,927.53 | 1.000× |
| 512³ | 9,039.45 | 6,553.60 | 1.379× |
| 1024³ | 13,273.11 | 10,699.76 | 1.241× |
| 2048³ | 13,831.18 | 12,501.65 | 1.106× |

For these square FP32/IEEE GEMMs on the tested RTX 4060 Ti, the small Triton autotuning search space produced configurations that outperformed the measured `torch.matmul` path at 512³, 1024³, and 2048³.

These results should not be interpreted as a general claim that Triton is faster than cuBLAS.

They apply to the specific matrix shapes, datatype, precision settings, hardware, software versions, and benchmark methodology used in this experiment.

---

# Nsight Compute analysis

The final phase used Nsight Compute to compare kernel behavior at the GPU architecture level.

The main comparison used a 1024³ GEMM with two fixed Triton configurations:

### Weak configuration

```text
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
num_warps = 4
num_stages = 3
```

### Tuned configuration

```text
BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32
num_warps = 4
num_stages = 3
```

A cuBLAS SGEMM kernel was also profiled as an external baseline.

---

## Weak Triton configuration

The 32×32×32 Triton kernel completed the 1024³ GEMM in approximately **270.08 µs**, consuming **663,696 elapsed cycles**. It reached approximately **37% of FP32 peak performance**.

The profiler showed that the kernel was heavily dependent on the load/store pipeline. LSU was the most highly utilized pipeline, reaching approximately **85.9% utilization**, while FMA utilization was approximately **37.7%**.

Memory-pipeline utilization was similarly high:

```text
Mem Busy:       79.55%
Mem Pipes Busy: 85.94%
```



The scheduler statistics showed:

```text
Active warps / scheduler:   4.80
Eligible warps / scheduler: 1.05
No eligible warp:           44.32%
```



Each warp required approximately **8.62 cycles per issued instruction**. Nsight attributed roughly 35% of that interval to short-scoreboard stalls, commonly associated with shared-memory dependencies.

The weak kernel executed approximately **48.7 million instructions**.

Resource usage was relatively modest:

```text
Registers/thread:      48
Dynamic shared memory: ~16.38 KB/block
Achieved occupancy:    40.05%
```



---

## Tuned Triton configuration

The tuned 128×64×32 configuration completed the same GEMM in approximately **173.89 µs**, consuming **423,458 elapsed cycles**. It reached approximately **58% of FP32 peak performance**.

The tuned kernel was more compute-oriented.

Its measured compute throughput was approximately **65.97%**, and FMA was the highest-utilized execution pipeline.

Scheduler statistics showed:

```text
Active warps / scheduler:   1.91
Eligible warps / scheduler: 1.19
No eligible warp:           29.49%
```



The most striking result was the improvement in instruction-issue efficiency:

```text
Warp cycles per issued instruction: 2.71
```

compared with 8.62 cycles for the weak configuration.

The tuned kernel also executed fewer instructions:

```text
Executed instructions: 37,943,296
```

compared with approximately 48.7 million for the weak configuration.

However, these improvements came at a substantial resource cost:

```text
Registers/thread:        166
Dynamic shared memory:   ~49.15 KB/block
Theoretical occupancy:   16.67%
Achieved occupancy:      15.95%
```



Nsight explicitly reported that occupancy was limited by shared-memory requirements.

The profiler also detected approximately **4.5-way shared-store bank conflicts**, representing an additional optimization opportunity.

---

# Bad vs tuned Triton comparison

The two profiles provide a useful direct comparison:

| Metric | 32×32×32 | 128×64×32 |
|---|---:|---:|
| Duration | 270.08 µs | **173.89 µs** |
| Elapsed cycles | 663,696 | **423,458** |
| FP32 peak achieved | 37% | **58%** |
| Executed instructions | 48.69M | **37.94M** |
| Registers/thread | **48** | 166 |
| Achieved occupancy | **40.05%** | 15.95% |
| Active warps/scheduler | **4.80** | 1.91 |
| Eligible warps/scheduler | 1.05 | **1.19** |
| Warp cycles/issued instruction | 8.62 | **2.71** |

The tuned kernel reduced runtime by approximately **36%**.

It also eliminated approximately **240,000 elapsed GPU cycles** for the same mathematical operation.

The most important observation is that this speedup occurred while occupancy fell from approximately **40% to 16%**.

---

# Why the larger tile wins

The smaller kernel makes more thread blocks available to the GPU and therefore achieves higher occupancy.

However, higher occupancy does not compensate for inefficient work.

The 32×32 configuration performs proportionally more load/store activity relative to arithmetic. Nsight shows heavy LSU utilization, significant memory-pipeline pressure, and scoreboard stalls associated with shared-memory operations. 

The larger tile increases reuse.

Values loaded for a 128×64 output tile participate in more arithmetic before being discarded. Although this requires substantially more registers and shared memory, it reduces the amount of instruction and memory overhead required per floating-point operation.

The result is:

```text
32×32×32
    ↓
higher occupancy
more active warps
less register pressure
more instructions
more LSU pressure
more dependency stalls
lower arithmetic efficiency
slower execution

128×64×32
    ↓
lower occupancy
fewer active warps
high register pressure
fewer instructions
greater data reuse
better issue efficiency
higher arithmetic utilization
faster execution
```

The tuned kernel had fewer active warps per scheduler than the weak kernel, yet slightly more eligible warps per scheduler.

This distinction is important.

An **active warp** occupies GPU resources.

An **eligible warp** is actually ready to issue its next instruction.

Having many resident warps is only useful if enough of them are ready to execute.

---

# Occupancy is not the optimization target

One of the clearest lessons from this experiment is that occupancy should be treated as a means rather than an objective.

The weak Triton kernel achieved approximately 40% occupancy.

The tuned kernel achieved only approximately 16%.

Yet the tuned kernel was substantially faster.

The same pattern appeared in the cuBLAS comparison. The profiled cuBLAS SGEMM kernel used **122 registers/thread** and achieved approximately **32% occupancy**, while still maintaining strong compute utilization.

Optimization therefore requires balancing several competing factors:

```text
occupancy
register pressure
shared-memory capacity
instruction-level parallelism
data reuse
memory traffic
scheduler readiness
pipeline utilization
```

Maximizing any one metric in isolation can make overall performance worse.

---

# Comparison with cuBLAS

A profiled cuBLAS SGEMM kernel completed the comparable GEMM in approximately **188.8 µs**, consuming approximately **453,870 elapsed cycles**.

Its scheduler and resource characteristics included:

```text
Registers/thread:             122
Achieved occupancy:           ~32%
Active warps/scheduler:       ~3.84
Eligible warps/scheduler:     ~1.86
Warp cycles/issued inst:      5.22
Executed instructions:        ~44.0M
SM throughput:                ~71.5%
```

 

The tuned Triton kernel therefore used more registers and had lower occupancy than cuBLAS, but completed this particular 1024³ workload in fewer elapsed cycles and with fewer executed instructions.

Again, this should not be generalized into a claim that Triton universally outperforms cuBLAS.

The result demonstrates that kernel performance depends strongly on workload shape, tuning parameters, compiler decisions, and hardware architecture.

---

# Connection to GPU clock cycles

These experiments provide a concrete interpretation of GPU cycle-level optimization.

For the same 1024³ GEMM:

```text
Weak Triton:
663,696 elapsed cycles

Tuned Triton:
423,458 elapsed cycles
```

The tuned configuration saves approximately 240,000 GPU cycles.

That improvement does not come from increasing clock frequency.

It comes from doing a better job of using the available cycles.

The tuned kernel:

- executes fewer instructions,
- spends fewer cycles between issued instructions,
- increases arithmetic utilization,
- improves data reuse,
- and reduces the fraction of execution dominated by load/store work.

This is the practical connection between high-level kernel tuning and low-level GPU architecture.

Kernel optimization is ultimately about deciding what useful work the processor performs during each available cycle.

---

# Current conclusions

The Triton experiments produced several useful findings.

1. **Tile shape has a major effect on GEMM performance.**

   Moving from 32×32×32 to 128×64×32 increased throughput dramatically for large matrices.

2. **Larger reduction tiles are not automatically better.**

   Increasing `BLOCK_K` from 32 to 64 reduced performance in the tested configurations.

3. **Warp count has a clear optimum.**

   Four warps substantially outperformed two, while eight provided no additional benefit.

4. **Pipeline depth is workload dependent.**

   Three stages performed best for 1024³, while two stages performed best for 2048³.

5. **Autotuning is valuable because optimal configuration depends on shape.**

6. **Occupancy alone is a poor predictor of kernel performance.**

7. **Data reuse can justify high register and shared-memory usage.**

8. **Eligible warps matter more than simply having many resident warps.**

9. **Instruction count and issue efficiency help explain differences that runtime alone cannot.**

10. **Cycle-level profiling provides an architectural explanation for benchmark results.**

---

# Future work

Several directions remain open.

## Triton memory-layout analysis

The tuned kernel still exhibits shared-memory bank conflicts.

Future experiments can investigate whether layout changes reduce these conflicts without increasing register pressure or reducing arithmetic efficiency.

## Additional matrix shapes

The current experiments focus primarily on square GEMMs.

Future tests should include:

- tall-skinny matrices,
- short-wide matrices,
- non-power-of-two dimensions,
- transformer-style matrix shapes,
- and small-batch GEMMs.

## Precision modes

The current Triton study deliberately uses FP32 input precision with:

```python
input_precision="ieee"
```

Future work can separately evaluate:

- TF32,
- FP16,
- BF16,
- Tensor Core execution,
- and the numerical differences between those modes.

## Larger autotuning space

The initial autotuner intentionally uses a small search space derived from manual experiments.

Future work could explore:

- more tile shapes,
- additional warp counts,
- more pipeline stages,
- architecture-specific configurations,
- and automated search-space generation.

## Generated-code inspection

The next level of analysis could inspect the code emitted by Triton and compare it with handwritten CUDA and cuBLAS at the PTX and SASS levels.

This would help explain differences in:

- instruction scheduling,
- memory instructions,
- FMA usage,
- register allocation,
- loop structure,
- and dependency chains.

## Cross-hardware tuning

A particularly useful future experiment would run the same Triton autotuning process across different NVIDIA GPUs.

This would test whether configurations that perform well on the RTX 4060 Ti remain optimal on architectures such as Turing, Ampere, or higher-end Ada GPUs.

That would extend the project's existing cross-hardware philosophy from numerical reproducibility into kernel scheduling and performance portability.

---

# Summary

The Triton phase of Matmul Inspector demonstrates the progression from implementing a correct GPU kernel to explaining its performance architecturally.

The project moved through:

```text
basic Triton GEMM
        ↓
tile-size sweep
        ↓
warp-count sweep
        ↓
pipeline-stage sweep
        ↓
autotuning
        ↓
Nsight Compute profiling
        ↓
cycle-level explanation
```

The most important result is not simply that one Triton configuration was faster than another.

The useful result is understanding **why**.

The tuned 128×64×32 configuration used more registers and shared memory and achieved much lower occupancy than the 32×32×32 configuration, yet it completed the same GEMM with fewer instructions, fewer elapsed cycles, better scheduler efficiency, and higher FP32 utilization.

That result captures one of the central lessons of GPU kernel optimization:

> The goal is not to maximize occupancy or any other isolated hardware metric. The goal is to organize computation so that the GPU spends as many cycles as possible performing useful work.