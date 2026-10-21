# MI-Research-7C — Perminov parallelism and memory-traffic attribution

## Decision

**B — Multiple contributors identified.** The 36-CTA mapping activated all 34 SMs, compared with four SMs for the original four-CTA mapping. It also executed fewer local-memory instructions and requested fewer local-memory bytes at L1. It executed substantially more total and tensor-pipe instructions and read more bytes from DRAM. The measured differences help explain why more arithmetic work can coexist with lower latency, but this profile does not isolate the latency contribution of any one effect.

The MI-Research-7B performance conclusion is unchanged: the 36-CTA implementation beat the four-CTA fused implementation in two ordinary CUDA Graph timing sessions and remained slower than cuBLAS. The Nsight Compute runs below are counter collection, **not** another timing comparison.

## Frozen control and workload

The control A is the original MI-Research-6C fused Perminov kernel. Candidate B is the single higher-parallelism output-block mapping from MI-Research-7B commit `d146df92e1ab9a4c908c0182cdd26caf9d9674e7`. Neither kernel or compiler configuration changed in this study. The exact source SHA-256 values are A `c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db` and B `10b94103a2dbf3d1649d1e3d79560d45830eeb5d960161905d69a6d5947db418`.

The target was one complete N=96 FP16 GEMM at fixed tensor addresses. The one-node CUDA Graph was captured after compilation and ten eager warmups. Ten graph warmups preceded correctness checks and another ten preceded profiling. Seeds 0, 1, and 2, zero, identity, and cancellation inputs all passed the unchanged `atol=rtol=0.01` check against FP32 matmul of the exact FP16 inputs rounded to FP16. Eager and graph outputs agreed bitwise in each case. The harness compared tensor pointers before and after graph use. Separate preflight and both profile-process records retain the checks for each kernel. The profiled region enclosed exactly one graph replay after these gates.

The device was an RTX 4060 Ti with 34 SMs, compute capability 8.9, and driver 616.92 under WSL2 Linux 6.18.54.1. Nsight Compute 2026.2.1.0 and Nsight Systems 2026.1.3 were installed. Hardware-counter access worked in this environment. Nsight Systems was not needed for the targeted counter question; Nsight Compute identified `_fused`, a `[128,1,1]` thread block, and the expected grid in every capture.

## Collection method

Each kernel had **two** targeted Nsight Compute captures, within the requested limit. The first was a one-metric occupancy and permission probe. The second collected the predefined occupancy section plus 21 named CTA, scheduler, instruction, L1, L2, and DRAM metrics. [`protocol.json`](../artifacts/mi-research-7c/protocol.json) records the first capture. [`targeted_capture_2.json`](../artifacts/mi-research-7c/targeted_capture_2.json) records the second metric list before that capture ran. The first capture used only `sm__warps_active.avg.pct_of_peak_sustained_active`, which read 8.33% for A and 8.76% for B. The occupancy section in the second capture read 8.33% and 8.81%, respectively. These separate readings agree within 0.05 percentage points.

Nsight Compute used `--graph-profiling node`, kernel replay, one matched `_fused` launch, `--profile-from-start off`, and CUDA profiler start/stop around the single replay. It used `--cache-control none` and `--clock-control none` to avoid requested cache flushing or clock locking. Nsight Compute explicitly warned that caches were uncontrolled and GPU clocks unmodified. Kernel replay and counter passes can still perturb cache state, execution, and duration. No profiler-instrumented duration is compared with ordinary CUDA-event measurements. The raw CSV console records retain the warnings, process marker, kernel identity, grid, and metrics. No generated profiler report or CUBIN was committed.

## Parallelism and occupancy

| Measured or reported field | A: four CTAs | B: 36 CTAs |
| --- | ---: | ---: |
| Grid | 2×2×1 | 2×2×9 |
| Threads per CTA | 128 | 128 |
| Compiler registers per thread | 255 | 255 |
| Compiler `n_spills` | 422 | 166 |
| Nsight register limit, blocks per SM | 2 | 2 |
| Nsight theoretical occupancy | 16.67% | 16.67% |
| Measured CTA launches, sum / min / max per SM | 4 / 0 / 1 | 36 / 1 / 2 |
| **SMs that received work** | **4 of 34** | **34 of 34** |
| Achieved occupancy on active SMs | 8.33% | 8.81% |
| Achieved active warps per active SM | 4.00 | 4.23 |
| Eligible warps per active scheduler cycle | 0.18 | 0.16 |
| Issue-active percent of peak, active cycles | 17.74% | 15.79% |
| Long-scoreboard stall fraction | 27.16% | 15.57% |
| LSU throttle stall fraction | 3.32% | 0.07% |

CTA-launch sum and extrema make the coverage inference exact for these captures. A launched four CTAs and no SM got more than one, so four SMs received work. B launched 36, and every one of 34 SMs received at least one. The achieved-occupancy values describe active SMs. They are **not** the fraction of the GPU that received work. B did not meaningfully raise occupancy or scheduler issue activity on an active SM. The register limit and theoretical occupancy were the same. The stall fractions differ, but a single profiled graph node per kernel does not establish their contribution to ordinary replay latency.

## Local memory, cache, and DRAM

| Executed counter for one profiled GEMM | A | B | B / A |
| --- | ---: | ---: | ---: |
| SASS local-load warp instructions | 9,632 | 4,000 | 0.415 |
| SASS local-store warp instructions | 7,728 | 3,840 | 0.497 |
| L1 local-load requested bytes | 1,236,992 | 512,000 | 0.414 |
| L1 local-store requested bytes | 1,099,776 | 507,904 | 0.462 |
| All L1 requested bytes | 2,428,928 | 1,476,608 | 0.608 |
| L1 hit / miss requested bytes | 568,064 / 1,860,864 | 492,800 / 983,808 | — |
| L2 requested bytes | 2,232,480 | 2,002,752 | 0.897 |
| DRAM read bytes | 109,824 | 387,584 | 3.529 |
| DRAM write bytes in the captured node | 0 | 0 | — |

Executed local-memory instructions fell by 58.5% for loads and 50.3% for stores. This is measured SASS-level activity, unlike the compiler's `n_spills` field. The local-memory L1 request counters also fell. Those L1 requests are not DRAM transfers. Total DRAM **reads increased** 3.53× for B despite lower local activity, consistent with the new mapping's duplicated operand work but not an isolated byte-level attribution. Zero DRAM writes in this short, cache-preserving capture do not mean the output was never written; the output was checked, and cache writeback may fall outside the measured node. Cache state was uncontrolled. These counters cannot divide DRAM bytes into spill, operand, and other causes.

## Executed arithmetic work

| Evidence layer | A | B | B / A |
| --- | ---: | ---: | ---: |
| Source-level dots per complete operation | 92 | 256 | 2.783 |
| Static `tt.dot` in one TTIR kernel | 23 | 64 | 2.783 |
| Static `mma.sync` in one PTX kernel text | 552 | 1,536 | 2.783 |
| Executed tensor-pipe warp instructions | 8,832 | 24,576 | 2.783 |
| All executed warp instructions | 108,864 | 310,368 | 2.851 |

The source, TTIR, PTX, and executed counter layers happen to have a similar ratio here. They are different units: the source count includes CTA multiplicity, TTIR and PTX counts include B's nine branches, and the hardware counters count executed warp instructions over the grid. The measured tensor-pipe and total instruction counts confirm that the added work was not simply removed by compilation. The TTIR and PTX hashes in [`analysis.json`](../artifacts/mi-research-7c/analysis.json) refer to the unchanged 7B compiler artifacts. PTX is not SASS; the executed counters supply a separate hardware observation.

## Interpretation and limits

The largest directly observed parallelism difference is **coverage**: four SMs received A's work and all 34 received B's. B also had substantially less executed local-memory activity and less L1/L2 request traffic. Those differences are plausible contributors to B's lower ordinary graph latency despite 2.78× more executed tensor-pipe instructions. Active-SM achieved occupancy and issue activity did not improve; DRAM reads increased. The study therefore supports **multiple contributors**, not a single dominant mechanism or a calibrated latency breakdown.

The counter snapshots are one profiled graph node per kernel in each of two captures. They do not provide repeat distributions, per-branch B attribution, directly measured spill-origin DRAM bytes, or normal-run occupancy. Nsight kernel replay, uncontrolled caches, and unmodified clocks limit quantitative comparison. The stable two-session 7B timing result remains the performance evidence; no new speedup is claimed from profiler-instrumented durations.

## Evidence, verification, and one next action

All retained records are under [`artifacts/mi-research-7c/`](../artifacts/mi-research-7c/). The offline analyzer checks source hashes, the 7B manifest, every correctness record, profiler kernel/grid identity, exact metric sets, counter interpretation gates, and every file hash. It requires no GPU:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_mi_research_7c.py --verify
PYTHONPATH=src .venv/bin/python -m pytest tests/mi_research_7c_test.py
```

The target harness accepts `--kernel A` or `B`, `--profile`, and a new `--output` path. For a new permission/occupancy probe, use the Nsight options in `protocol.json` with `--metrics sm__warps_active.avg.pct_of_peak_sustained_active`. For a new targeted capture, use the exact options and comma-joined metric names in `targeted_capture_2.json`. Both commands must run with `PYTHONPATH=src` and `--profile-from-start off`; the harness itself brackets the warmed graph replay. The retained output paths are intentionally protected against overwrite.

**Next research action:** Use one controlled per-SM timeline or occupancy trace of these same frozen graph nodes to measure how long each SM stays active. That would test whether B's wider SM coverage overlaps in time enough to account for the latency improvement, while keeping the current counter results as separate evidence.
