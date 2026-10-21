# Tensor Game timing stability diagnostics

## Motivation and research question

Earlier RTX 4060 Ti experiments showed large shifts in absolute latency and a
small apparent Strassen advantage at 1024. This study asks whether that crossover
survives paired rounds and what the benchmark is actually measuring. It does not
change decomposition coefficients, kernel arithmetic, precision, or geometry.

## Existing benchmark audit

The existing harness records a CUDA start event, invokes the complete Python
lowering, records an end event, then synchronizes the end event. The elapsed time
is GPU timeline time between events. It includes every queued GEMM and fused
reconstruction kernel. It excludes Python dispatch and ordinary allocator CPU
time, although an allocator operation that synchronizes the device can affect
the window. PyTorch and Triton use the same boundaries.

Inputs are reused. Each lowering invocation creates seven or eight rank-product
tensors and one output through PyTorch's caching allocator. cuBLAS also returns a
new output. Work is issued on the current default stream. There is no explicit
synchronization inside the runtime. Per-sample end-event synchronization prevents
overlap between samples. Correctness and warmups trigger compilation before
recording. Implementations are interleaved, but one randomized order is retained
for every iteration in a round. Prior reports summarized independent medians and
did not pair conventional and Strassen within each round.

Other GPU processes remain a possible interference source. The harness cannot
isolate the device or control OS scheduling. No power limit or clock setting was
changed.

## Controlled setup

The standalone runner compares PyTorch/cuBLAS, allocated conventional and
Strassen fused-input/fused-output paths, and matched preallocated variants at
512 and 1024. It uses IEEE FP32, seed `20261016`, 20 warmups, 15 independent
rounds, and 30 samples per implementation per round. Even rounds use direct
order; odd rounds reverse it. The round median is the replicate unit.

Device time uses CUDA events around one complete invocation. Host-visible time
uses `perf_counter_ns` with a full device synchronization immediately before and
after the invocation. Host time therefore includes Python dispatch, allocator
work, kernel execution, and synchronization overhead. These windows answer
different questions and are not compared as equivalent measurements.

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/diagnose_tensor_game_timing.py \
  --output artifacts/tensor-game/timing-diagnostics-v1/results.json \
  --sizes 512 1024 --warmups 20 --iterations 30 --rounds 15 \
  --seed 20261016
```

The environment was Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
Triton 3.6.0, NVIDIA driver 616.92, and an RTX 4060 Ti (SM 8.9).

## Paired device results

Statistics below operate on 15 round medians. CV is the coefficient of variation
across rounds. A conventional/Strassen ratio above 1 favors Strassen.

| Size | Implementation | Median | IQR | Range | CV |
|---:|---|---:|---:|---:|---:|
| 512 | cuBLAS | 0.0640 ms | 0.0263 | 0.0465-0.1068 | 27.1% |
| 512 | Conventional allocated | 0.2084 ms | 0.1962 | 0.1731-0.5034 | 45.4% |
| 512 | Strassen allocated | 0.2002 ms | 0.1354 | 0.1679-0.4493 | 43.0% |
| 1024 | cuBLAS | 0.2077 ms | 0.0140 | 0.1951-0.2272 | 4.8% |
| 1024 | Conventional allocated | 0.3994 ms | 0.0219 | 0.3912-0.6000 | 15.6% |
| 1024 | Strassen allocated | 0.3986 ms | 0.0243 | 0.3916-0.5443 | 11.4% |

At 1024 the paired conventional/Strassen median ratio is `1.0091`. Its IQR is
`0.9951-1.0147`, range is `0.9549-1.1414`, and round-bootstrap 95% interval for
the median is `0.9947-1.0149`. The interval includes parity. The apparent
Strassen advantage is therefore not repeatable at a defensible confidence level.

At 512 the paired ratio is `1.1268`, with bootstrap interval
`1.0471-1.1758`. This run favors Strassen, but absolute CVs exceed 40% and a
strong order effect limits generalization.

## Ordering and warm-state effects

At 1024, direct-order rounds have median paired ratio `1.0144`; reversed rounds
have `0.9947`. At 512, the corresponding values are `1.1709` and `1.0471`.
The implementation that runs earlier or later in the repeated sequence can
change the inferred gap. Alternating order is necessary, but it does not remove
all state carryover.

The 1024 device medians settle near 0.39-0.42 ms in most rounds, with isolated
slower rounds up to 0.60 ms. Prior output-fusion medians of 0.4970/0.4821 ms fall
inside the new observed ranges, while prior extremes above 1 ms do not recur
after the longer warmup. This supports treating warmup and sustained device state
as material, without proving a single cause.

## Host-visible timing

At 1024, synchronized host medians are 0.3867 ms conventional and 0.3818 ms
Strassen. Their paired ratio is `1.0077`, with bootstrap interval
`0.9889-1.0286`. This also includes parity. Host CVs are 10.6% and 8.5%.
The values can be slightly below event medians because the device and host
samples occur in separate sequences and operating state changes; this is not
negative dispatch overhead.

At 512, host medians are 0.2143 and 0.1984 ms. Variability remains high. Host
timing does not resolve the order sensitivity seen in device timing.

## Telemetry observations

`nvidia-smi` snapshots were available before and after every round. Memory clock
remained 8751 MHz. Graphics clock was 2565 MHz during 512 and rose from 2565 to
2760 MHz during 1024. Temperature covered 44-58 C. Reported power rose from
roughly 32 W to 116 W, below the 165 W limit. Utilization snapshots ranged from
5% to 50% because they are point samples rather than interval averages.

No active throttle mask was reported during 512. Nonzero hexadecimal throttle
masks appeared during 1024, but this experiment does not decode them and cannot
attribute latency to a specific limiter. Clock changes correlate with workload
progression but do not establish causation. Snapshot telemetry is too coarse to
explain individual samples.

## Allocation and synchronization findings

The preallocated mode reuses all rank-product and output tensors while launching
the same kernels and arithmetic. At 1024, device medians changed from 0.3994 to
0.3926 ms conventional and 0.3986 to 0.3925 ms Strassen, about 1.7% and 1.6%.
Synchronized host medians changed by about 3.1% and 1.5%. Allocation is measurable
but is not large enough to explain the earlier 2x shifts.

At 512, preallocated medians are lower, but their round CVs remain about 41-44%.
Allocation and operating-order effects are confounded there. The PyTorch caching
allocator means repeated `empty` calls usually reuse cached device memory. CUDA
events do not directly measure its host bookkeeping.

## Comparison, uncertainty, and likely variability sources

The strongest finding is common-mode state variation plus ordering sensitivity,
not a stable algorithm crossover. Longer warmup produces more repeatable 1024
latencies, yet occasional slow rounds remain. Telemetry confirms changing clocks
and power across the run. It does not prove DVFS is the cause. External GPU work,
driver scheduling, cache state, allocator state, and fixed within-round ordering
remain unmeasured contributors.

Individual iterations are not independent replicates. Bootstrap intervals use
15 round-level paired ratios. Fifteen rounds are still a small sample, and the
alternating order is deterministic rather than a full balanced crossover design.
Telemetry is sampled outside timing loops and may miss short transitions.

## Recommended measurement protocol

Future Tensor Game claims should use at least 20 warmups, 15 or more independent
rounds, paired per-round ratios, balanced direct/reversed order, and round-level
IQR/CV plus a confidence interval. Record device and synchronized-host timing
separately. Capture per-round telemetry and report any active throttle masks.
Use preallocated buffers for kernel-focused comparisons, while retaining an
allocated complete-operation result for end-user cost. A future runner should
randomize order per iteration with a balanced Latin-square schedule and sample
telemetry continuously. No further kernel optimization should rely on a 1024
Strassen crossover until its paired interval excludes parity under that protocol.
