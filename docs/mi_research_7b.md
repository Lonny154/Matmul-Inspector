# MI-Research-7B — Higher-parallelism fused Perminov

## Decision

**A — Higher-parallelism performance improvement.** The single new 36-CTA mapping passed the numerical, compiler, and launch gates. It beat the frozen four-CTA fused Perminov kernel in two independent CUDA Graph sessions under the unchanged paired rule. In the one conditional matched comparison, cuBLAS remained faster. This result does not establish the cause of the improvement.

## Question and control

The hypothesis was that assigning each 16×16 output tile of one 32×32 output block to its own CTA could reduce per-CTA live state and expose enough additional parallelism to offset duplicated circuit work. The control is the exact MI-Research-6C fused source retained by MI-Research-7 at commit `0b6c741f92073b20ec9009ef808704225e47826c`. Its source SHA-256 is `c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db`. The MI-Research-7 evidence was independently verified before this study and again by the 7B offline analyzer.

The frozen original uses a `(2, 2, 1)` grid. Each of four CTAs owns the same local 16×16 tile in all nine 32×32 output blocks. It computes all 23 rank products, with one 16×16×32 `tl.dot` per product, and writes nine disjoint 16×16 tiles. Four CTAs therefore execute 92 source-level dot instances per complete operation. Some input rows or columns are reused across CTAs. Each numerical product tile is specific to its row and column tile.

## One new mapping

The new grid is `(2, 2, 9)`. `program_id(2)` selects one output block; the first two IDs select its 16×16 tile. Thus 36 CTAs each own exactly 256 output elements. Each output element has one writer. The mapping uses one launch, no atomics, and no intermediate product scratch.

The new kernel is generated from the exact transitive dependency slice of each original output store. Every retained input load, `Ug`/`Vg`/`Wg` gate, rank-product `tl.dot`, and output store keeps its original expression and source order. The union of the nine branches contains the original 55 gate definitions. Shared gates and products are recomputed by independent output owners. FP16 inputs and output, FP32 operand preparation and reconstruction, `tf32x3` dots, four warps, and two stages match the control. The generated source SHA-256 is `10b94103a2dbf3d1649d1e3d79560d45830eeb5d960161905d69a6d5947db418`.

| Output block, row major | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dots per CTA | 8 | 6 | 6 | 12 | 3 | 10 | 9 | 3 | 7 |
| Gate additions per CTA | 20 | 16 | 20 | 33 | 2 | 31 | 30 | 2 | 22 |

For each local tile coordinate, the branch dots sum to 64 rather than 23. Across four tile coordinates, the new grid executes 256 source-level dot instances, **2.78×** the original 92. The nine branches execute 176 gate additions per local tile coordinate rather than 55, **3.20×** the original source-level count. These are source-level instances, not hardware instruction or memory-traffic measurements. A simple named-array liveness model found a maximum of 20 simultaneously live branch values versus 32 in the original full program. It is an abstract model and does not predict registers or occupancy. The plan and these counts were frozen in [`mapping_plan.json`](../artifacts/mi-research-7b/mapping_plan.json) before the new kernel was compiled.

## Compiler, correctness, and topology

The RTX 4060 Ti has 34 SMs and compute capability 8.9. Both primary sessions used driver 616.92, CUDA runtime 12.8, PyTorch 2.11.0+cu128, Triton 3.6.0, and Python 3.13.11. Compiler output was captured in TTIR, TTGIR, LLVM IR, and PTX. The hashes are in each session's raw record and manifest.

| Metric | Original A | New B |
| --- | ---: | ---: |
| Configured CTAs | 4 | 36 |
| Maximum source-level dots per CTA | 23 | 12 |
| Compiler registers per thread | 255 | 255 |
| Compiler-reported `n_spills` | 422 | 166 |
| Shared memory per CTA | 1,024 B | 1,024 B |
| Static `mma.sync` occurrences in PTX text | 552 | 1,536 |
| `ld.local` / `st.local` text occurrences | 0 / 0 | 0 / 0 |

The PTX count includes all nine branches in B's compiled kernel and is a **static text count**. It is not a dynamic executed instruction count. PTX is not SASS. Neither `n_spills` nor the lack of PTX local-load/store text proves a particular amount of physical spill traffic. The compiler reported no resource failure. The PyTorch profiler printed its standard event-cycle warning during trace capture; it did not invalidate the recorded 20-kernel traces.

Both implementations passed seeds 0, 1, and 2, zero input, identity input, and cancellation-sensitive input against FP32 matmul of the exact FP16 inputs rounded to FP16. The original absolute and relative tolerances remained 0.01. Each case was checked twice in eager execution and twice after fixed-address graph replay, poisoning the output before each check. Graph and eager outputs were bitwise equal for each checked case. All recorded checks passed without nonfinite output or failing elements. Each graph replay contained 20 complete operations and the profiler observed 20 physical kernels with one `cudaGraphLaunch` call. The observed grids were `[2,2,1]` for A and `[2,2,9]` for B, each with `[128,1,1]` threads. The graph's captured tensor addresses remained stable by construction; the run record's address flag was set by the harness rather than by a separate pointer comparison.

## Paired graph timing

Two fresh Python processes ran eight balanced paired rounds each. Each measurement replayed one fixed-address graph with 20 complete operations. Each session also retained the existing A/A control, CUDA-event timing, RAW clock, `perf_counter`, all samples, and host-clock preflight. Compilation, warmup, and graph capture were outside the timed rounds. No outliers were removed.

| Session | A median µs/op | B median µs/op | Median paired A−B µs/op | Max A/A gap µs | Paired rule |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary 1 | 15.742 | 9.564 | 6.158 | 1.595 | B_FASTER |
| Primary 2 | 16.338 | 10.045 | 6.227 | 1.376 | B_FASTER |

The new mapping is about 39% lower in median complete-operation latency in these sessions. The unchanged rule reported separated event ranges, agreement in both host-clock directions, and every A/B gap larger than the A/A bound. This is a supported improvement relative to the frozen fused kernel, not a general claim across sizes, devices, or eager execution.

Both primary sessions met the frozen trigger for one matched cuBLAS comparison. In that fresh session, timing slot A was the new Perminov B implementation and timing slot B was cuBLAS C. The new mapping's median was **10.048 µs/op** and cuBLAS's was **2.532 µs/op**. The paired rule returned `B_FASTER`, meaning **cuBLAS was faster**. The cuBLAS implementation was a preallocated `torch.mm(..., out=...)` under the same FP16 precision policy and 20-operation CUDA Graph measurement. It passed the same numerical cases and had one physical kernel per operation. MI-Research-7's earlier cuBLAS median is historical context; only this matched session supports the 7B comparison. One external-anchor session cannot establish a broad performance distribution.

## Interpretation and limits

The mapping increases configured CTAs ninefold and lowers compiler-reported spills while leaving the reported register count at 255. It also adds 2.78× source-level dot work and 2.78× static PTX MMA text. Despite that extra work, both primary sessions show lower graph latency. This supports testing CTA ownership as an important mapping variable. It does not isolate whether more concurrent CTAs, reduced per-CTA live state, altered scheduling, or another compiler/runtime effect caused the result. Achieved occupancy and measured DRAM traffic were not captured. The 36-CTA grid only slightly exceeds the GPU's 34 SMs, and the branch workloads differ, so the configured CTA count is not an occupancy measurement.

No source or numerical adjustment was made after the mapping was frozen. There was no tiling, warp, or algebraic sweep.

## Evidence and reproduction

The frozen protocol is [`protocol.json`](../artifacts/mi-research-7b/protocol.json). Raw correctness, graph traces, timing rounds, A/A controls, compiler IR/PTX, environment, source snapshots, and SHA-256 manifests are retained under [`artifacts/mi-research-7b/`](../artifacts/mi-research-7b/). [`analysis.json`](../artifacts/mi-research-7b/analysis.json) replays the decisions and resource comparison. Each session manifest hashes every retained file; `analysis_manifest.json` hashes the protocol, plan, analysis, and session manifests.

From the repository root, with the project's CUDA environment active:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_mi_research_7b_kernel.py
PYTHONPATH=src .venv/bin/python scripts/run_mi_research_7b.py --output artifacts/mi-research-7b/primary-s1 --session 1
PYTHONPATH=src .venv/bin/python scripts/run_mi_research_7b.py --output artifacts/mi-research-7b/primary-s2 --session 2
PYTHONPATH=src .venv/bin/python scripts/run_mi_research_7b.py --output artifacts/mi-research-7b/cublas-s1 --session 1 --comparison cublas
PYTHONPATH=src .venv/bin/python scripts/analyze_mi_research_7b.py --verify
PYTHONPATH=src .venv/bin/python -m pytest tests/mi_research_7b_test.py
```

The run command refuses to overwrite retained output. Use new output paths for a new measurement. The offline verifier needs no GPU. Reproduction on other hardware is a new experiment, not an exact replay of these timings.

**Next research direction:** Profile achieved occupancy and measured memory traffic for these two frozen kernels under matched graph replay. That controlled observation can test whether the lower reported spill count corresponds to physical memory behavior and whether CTA parallelism is actually realized.
