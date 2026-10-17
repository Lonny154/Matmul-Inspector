# Tensor Game GPU hardware-cost calibration

## Research question

This study benchmarks the 12 cost-diverse change-of-basis decompositions on an NVIDIA RTX 4060 Ti. It asks whether static preparation and reconstruction costs predict complete-operation GPU latency within a fixed bilinear rank. The literal conventional and Strassen decompositions are included as paired anchors, for 14 implementations total. No coefficients, Triton kernels, tile sizes, or search policies were changed.

## Environment and compatibility

The run used an RTX 4060 Ti with 16 GB, compute capability 8.9, driver 616.92, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime 12.8, and Triton 3.6.0. PyTorch TF32 was disabled. Triton used IEEE FP32 dot precision, 32×32×16 tiles, four warps, and three stages.

All selected records were loaded directly from `change-of-basis-v1/representatives.json`. Their content IDs and complete `U/V/W` arrays were preserved. Each record passed the exact integer tensor validator and `create_lowering()` before CUDA execution. The generic fused-input and fused-reconstruction paths consume these coefficients directly. No algorithm-specific arithmetic adapter was needed.

The selected set contains six rank-7 and six rank-8 candidates. Preparation additions range from 0 to 18 in this subset, reconstruction additions from 4 to 14, total support from 24 to 54, and maximum coefficient magnitude from one to two. The two literal anchors are separately identified and are not substituted for basis-equivalent selected records.

## Correctness

Correctness covered zero, identity, negative, and seeded random inputs at sizes 128, 256, 512, and 1024. This produced 224 candidate/case/size checks. All 14 implementations passed `torch.allclose` against PyTorch with `rtol=5e-3` and `atol=5e-3`.

Maximum absolute error ranged from 0 to `5.187988e-4`. The largest per-case mean absolute error was `3.295298e-5`. The maximum elementwise relative error over nonzero reference entries was 6.50, caused by division by very small reference values. It is preserved for transparency but is not a useful aggregate correctness measure. No tolerance was relaxed and no candidate was excluded.

## Measurement protocol

The timed sizes were 256, 512, and 1024. Every size used identical seeded inputs across candidates, 20 warmups, 15 independent rounds, and 30 CUDA-event samples per implementation per round. Five separately synchronized host-visible samples were also recorded per candidate and round. Compilation and first use completed before timing.

Even rounds use a seeded randomized order. Each following odd round reverses the preceding order exactly. Round medians are the replicate unit. Candidate/reference ratios are paired within each round against the literal anchor of the same rank. Confidence intervals bootstrap the 15 paired round ratios with 5,000 resamples. Inner-loop samples are not treated as independent replicates.

CUDA events surround the complete fused operation: seven or eight fused-input GEMMs, rank-product materialization, and one fused reconstruction kernel. Python dispatch and ordinary host allocator time are outside the device interval. Synchronized host timing includes dispatch, execution, allocation effects, and synchronization.

```sh
PYTHONPATH=src TRITON_CACHE_DIR=artifacts/tensor-game/gpu-cost-calibration-cache-v1 \
  /home/daniel/IsaacLab/.venv/bin/python \
  scripts/benchmark_tensor_game_gpu_cost_calibration.py \
  --dataset artifacts/tensor-game/change-of-basis-v1 \
  --output artifacts/tensor-game/gpu-cost-calibration-v1 \
  --correctness-sizes 128 256 512 1024 \
  --benchmark-sizes 256 512 1024 \
  --warmups 20 --iterations 30 --host-iterations 5 --rounds 15 --seed 20261016
```

## Device latency

The table shows the 12 selected candidates at 1024. Latency statistics operate on 15 round medians. Ratios above one are slower than the literal same-rank anchor.

| Candidate | Rank | Pre | Post | Scale | Median ms | IQR ms | CV | Paired ratio | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `30b13f831111` | 8 | 0 | 4 | 0 | 0.4013 | 0.0605 | 14.3% | 0.998 | 0.984–1.002 |
| `0f0a7a60a1ed` | 7 | 10 | 8 | 0 | 0.4116 | 0.0304 | 10.7% | 0.998 | 0.982–1.010 |
| `00ad1a3f52e8` | 7 | 12 | 10 | 0 | 0.4193 | 0.0251 | 9.0% | 1.022 | 1.013–1.038 |
| `0b88a39c7ca3` | 7 | 14 | 12 | 4 | 0.4296 | 0.0229 | 9.5% | 1.058 | 1.040–1.065 |
| `0075fe649958` | 8 | 2 | 8 | 0 | 0.4301 | 0.0383 | 12.8% | 1.029 | 1.008–1.047 |
| `01d0d3ca3b34` | 7 | 14 | 10 | 0 | 0.4352 | 0.0291 | 7.9% | 1.074 | 1.056–1.081 |
| `05402bd09ac1` | 8 | 4 | 14 | 0 | 0.4371 | 0.0363 | 11.9% | 1.077 | 1.032–1.101 |
| `192a246e8a82` | 8 | 4 | 4 | 0 | 0.4382 | 0.0295 | 12.0% | 1.084 | 1.009–1.089 |
| `0b5b9cccafd3` | 7 | 14 | 8 | 0 | 0.4626 | 0.0242 | 6.4% | 1.126 | 1.095–1.142 |
| `00e8aee9996b` | 8 | 7 | 8 | 0 | 0.4782 | 0.0187 | 8.8% | 1.148 | 1.088–1.182 |
| `10a894434fb7` | 7 | 18 | 8 | 4 | 0.4847 | 0.0203 | 4.8% | 1.183 | 1.149–1.187 |
| `00ccf0422090` | 8 | 10 | 14 | 0 | 0.5350 | 0.0243 | 4.9% | 1.271 | 1.174–1.286 |

The literal anchors measured 0.4039 ms for Strassen and 0.4054 ms for conventional. PyTorch/cuBLAS measured 0.2360 ms. These generic multi-kernel lowerings therefore remain slower than cuBLAS.

At 256, selected rank-7 medians span 0.1771–0.1831 ms and rank-8 medians span 0.1945–0.2037 ms. CVs are 4.3–6.9%. At 512, medians span 0.1961–0.2093 ms for rank 7 and 0.2116–0.2237 ms for rank 8, but CVs rise to 22.8–43.1%. Reliable fine-grained ranking was not established at 512. PyTorch medians are 0.0444, 0.0798, and 0.2360 ms at 256, 512, and 1024 respectively.

## Within-rank cost relationships

Spearman correlations use six selected candidates per rank. This is a small descriptive sample of equivalent transformations, not six independent algorithm discoveries per family.

| Size | Rank | Preparation | Reconstruction | Total support |
|---:|---:|---:|---:|---:|
| 256 | 7 | 0.455 | -0.370 | 0.088 |
| 256 | 8 | 0.456 | 0.788 | 0.657 |
| 512 | 7 | -0.152 | -0.463 | -0.441 |
| 512 | 8 | 0.986 | 0.478 | 0.853 |
| 1024 | 7 | **0.941** | -0.278 | 0.618 |
| 1024 | 8 | **0.986** | 0.478 | 0.853 |

Preparation cost has a strong monotonic relationship with latency at 1024 in both ranks. The rank-8 relationship also appears at 512, while the noisy rank-7 512 result reverses sign. Reconstruction cost is inconsistent and is not independently identifiable from support and input complexity in this selected set. Total support is promising at 1024, especially for rank 8.

Candidates with the same aggregate addition point can still differ slightly. The selected `0f0a...` Strassen-equivalent candidate has the same `(10,8,0)` cost as literal Strassen, and `30b1...` has the same `(0,4,0)` cost as literal conventional. Their paired intervals include parity. This experiment therefore does not establish a difference at equal aggregate costs, but it shows that the descriptors are not a complete execution model.

## Ordering stability and uncertainty

At 1024, direct-versus-reversed ranking correlation is 1.000 for rank 7 and 0.829 for rank 8. The high-cost candidates remain slow under both orders. At 256 the corresponding correlations are only 0.257 and 0.486. At 512 they are -0.257 and 0.600. Small-size rankings are sensitive to operating order and state.

Telemetry snapshots cover 44–59 °C, 18.8–134.8 W, graphics clocks from 210–2745 MHz, memory clocks from 405–8751 MHz, and utilization from 12–83%. Idle values appear at session boundaries. Snapshots do not measure the state of each individual event interval and cannot attribute a latency difference to DVFS or another limiter.

Rank 7 has lower group-median latency at 256 and 512. At 1024 the group medians are close: 0.4324 ms for rank 7 and 0.4376 ms for rank 8, while the fastest selected candidate is rank 8. Rank reduction alone is therefore not a sufficient performance predictor.

## Compiler evidence

Low-, medium-, and high-cost candidates were compiled separately at size 1024 for each rank. TTIR, TTGIR, PTX, SASS, compiler metadata, CUBIN hashes, and resource-tool output are preserved. CUBIN binaries themselves are not committed.

| Candidate | Rank | Kernels | Max registers | Max dynamic shared | Local/stack | Summed PTX FFMA |
|---|---:|---:|---:|---:|---:|---:|
| `30b13f831111` | 8 | 9 | 48 | 12,288 B | 0 / 0 | 1,088 |
| `05402bd09ac1` | 8 | 9 | 48 | 20,480 B | 0 / 0 | 1,088 |
| `00ccf0422090` | 8 | 9 | 50 | 28,672 B | 0 / 0 | 1,088 |
| `0f0a7a60a1ed` | 7 | 8 | 44 | 20,480 B | 0 / 0 | 952 |
| `0b88a39c7ca3` | 7 | 8 | 50 | 24,576 B | 0 / 0 | 960 |
| `10a894434fb7` | 7 | 8 | 50 | 28,672 B | 0 / 0 | 968 |

Rank-8 aggregate PTX/SASS FFMA, global-load, and global-store text counts are identical across the three representatives, even though latency and shared-memory requirements differ. Rank-7 FFMA counts rise modestly with the selected costs. Register and dynamic shared-memory requirements also rise from the low-cost representatives to the high-cost representatives. This is observable compiler evidence consistent with more complex input combinations. It does not by itself measure occupancy, traffic, cache behavior, or dynamic instruction execution.

## Answers to the calibration questions

1. Lower preparation cost corresponds strongly to lower 1024 latency within both ranks. The relationship is not stable for rank 7 at 512 or for either rank at 256.
2. Reconstruction cost does not show a consistent independent relationship. One fused reconstruction launch reduces launch-count variation, and its arithmetic remains confounded with support layout.
3. Equal aggregate addition points are compatible with parity in the available paired comparisons. The model remains incomplete because support placement and compiler resources are omitted.
4. Register count and dynamic shared memory distinguish low- and high-cost representatives where rank-8 instruction-category totals do not. They are plausible explanatory descriptors, not causal proof.
5. Rank-7 candidates are generally lower latency at 256 and 512, but not uniformly at 1024. Rank is a major confounder, not a guaranteed ordering.
6. The 1024 ranking is stable under reversal. Rankings at 256 and 512 are not stable enough for fine distinctions.
7. Preparation additions, total input support, maximum specialized-kernel shared memory, and register count are promising inputs for future hardware-aware search. Reconstruction additions alone are weaker under fused output.

## Artifacts and limitations

Evidence is stored in `artifacts/tensor-game/gpu-cost-calibration-v1/`. `results.json` contains the candidate manifest, 224 correctness rows, 23,625 raw timing samples, round medians, telemetry, and aggregate uncertainty. `analysis.json` contains within-rank correlations, order sensitivity, and compiler summaries. The `compiler/` tree contains bounded text artifacts for six representatives. `artifact_hashes.json` records SHA-256 hashes and sizes for every retained file. The results hash is `6679fdf238581962633e285f2562eff6ec8c1c20d4000d0b9f1ae8814699a907`.

This is one GPU, one session, one fixed tile geometry, and a small deliberately selected candidate set. Static descriptors are correlated with each other. Change-of-basis variants within a family are not independent mathematical algorithms. CUDA-event timing excludes host dispatch; synchronized host results answer a different question. No profiler measured traffic, occupancy, or dynamic instructions. The bootstrap describes round-to-round uncertainty in this session and is not a confidence interval over GPUs or workloads.

The recommended next experiment is a second-session replication at 1024 using the same frozen candidates and balanced order. It should profile the low and high representative from each rank with Nsight Compute to test whether shared-memory use, register pressure, achieved occupancy, and memory traffic explain the latency gradient. Search policy changes should wait until that replication confirms the 1024 preparation-cost signal.
