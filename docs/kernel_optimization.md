# CUDA Kernel Optimization Case Study

## Baseline
- 16×16 tiled kernel
- 256 threads/block
- one output per thread

## Profiling diagnosis
- LSU ~92%
- ~330M issued instructions
- ~98% occupancy
- ~1.27 ms at 1024³

## Hypothesis
Reduce load/store pressure by increasing per-thread reuse.

## Optimization
- 2×2 register blocking
- 8×8 threads/block
- four outputs/thread
- four accumulators/thread

## Correctness
- bitwise-identical outputs
- awkward square sizes pass
- rectangular case passes
- 29/29 tests pass

## Performance
- ~1.48× faster at 1024³
- ~1.47× faster at 2048³
- ~2.58 TFLOP/s at 2048³

## Profiler before/after
- registers/thread: 24 → 40
- occupancy: ~98% → ~94%
- LSU utilization: ~92% → ~57%
- issued instructions: ~330M → ~136M
- profiler duration: ~1.27 ms → ~0.86 ms

## Interpretation
- more registers
- slightly lower occupancy
- dramatically less instruction/load-store work
- faster overall execution

## 4x4 register blocking
- 4×4 register blocking reduced LSU pressure further, but crossed the optimal point for this kernel design. Using only 16 threads per block halved theoretical occupancy from 100% to 50%, reduced active warps per SM from ~45 to ~23, increased register usage to 72/thread, and increased issued instructions. The added per-thread reuse did not compensate for reduced parallelism and higher instruction overhead.

## Limitations
- custom kernel, not cuBLAS
- single GPU
- FP32 only
- no tensor cores
- profiler replay timing not used for benchmark claims