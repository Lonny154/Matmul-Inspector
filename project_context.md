# Matmul Inspector — Project Context

**Last context update:** October 10, 2026  
**Repository:** https://github.com/Lonny154/Matmul-Inspector  
**Purpose:** Persistent technical context for future ChatGPT Project conversations. This file summarizes historical work; check Git, current hardware, and saved benchmark artifacts for live state.

## 1. Project mission

Matmul Inspector is a hands-on research and developer-tooling project for understanding how numerical kernels map to GPU and AI-accelerator hardware. The work emphasizes **correctness, reproducible measurement, compiler output, memory behavior, and instruction-level explanations** rather than unverified speedup claims.

Main technical tracks:

- NVIDIA CUDA and PTX kernel construction and inspection.
- Triton GEMM implementation, autotuning, and comparison with PyTorch/cuBLAS.
- A miniature Transformer workload to connect individual kernels with end-to-end throughput.
- AWS Trainium1 Neuron Kernel Interface (NKI) GEMM implementation, precision and tiling sweeps, hardware profiling, and compiler IR analysis.
- Adjacent numerical computing experiments, including Thomas and parallel cyclic reduction (PCR) tridiagonal solvers.

Longer-term aims include packaging Matmul Inspector for `pip`, automating PTX/MLIR and profiler-artifact comparisons, and developing the expertise to contribute meaningfully to FlashAttention, FlashInfer, and vLLM.

## 2. Development environments

### NVIDIA workstation

- **GPU:** NVIDIA RTX 4060 Ti, Ada architecture, compute capability SM 8.9.
- **Tools used:** CUDA C++, `nvcc`, PTX/CUBIN output, PyTorch, Triton, Nsight Systems (`nsys`), Nsight Compute (`ncu`), and NVTX instrumentation.
- **Repository on local Linux/WSL:** Previously used `~/projects/matmul-inspector`; verify current location and branch before editing.

### AWS Trainium

- **Instance:** EC2 `trn1.2xlarge`, Trainium1; a prior `neuron-ls` check showed one Neuron device with two NeuronCores, with experiments generally restricted to **one core**.
- **Working directory:** `~/Matmul-Inspector` on the EC2 Ubuntu instance.
- **Virtual environment:**

  ```bash
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
  ```

- **NKI API as observed October 10:** `nki.jit` exists; `nki.benchmark`, `nki.baremetal`, and `nki.profile` are **not** attributes on the installed top-level `nki` module. Don't assume older examples work.
- **Compilation artifact workflow:** Set `NKI_ARTIFACTS_DIR` before running an `@nki.jit` kernel to capture `kernel.neff`, `module.mlir`, compiler logs, and other artifacts.
- **Measurement tools:** `neuron-bench exec`, Neuron Explorer (`capture`, `view`), and Perfetto for trace visualization. On hardware, ensure `NKI_SIMULATOR` is unset.

## 3. CUDA / Triton / Transformer track

### Transformer workload

A miniature Transformer benchmark was created around a representative workload:

| Parameter | Value |
|---|---:|
| Batch | 1 |
| Sequence length | 512 |
| Hidden size | 768 |
| Attention heads | 12 |
| MLP intermediate size | 3072 |
| Data type | FP16 |

Historical baseline from October 7:

- Mean block latency approximately **0.2663 ms** (a separate `nsys`-instrumented run measured about **0.2976 ms**).
- Approximately **3,754.85 blocks/s** and **1,922,485 tokens/s** under the reported benchmark's accounting. These are synthetic benchmark throughput figures, **not** model-serving generation rates.
- NVTX profiling identified `qkv_projection` as approximately **89.9% of the summed NVTX-range time** in one report. This is a range-accounting observation; do not automatically interpret it as 89.9% of nonoverlapping wall time.
- Nsight Compute was also used to target the `mlp_fc1` cuBLAS kernel.

### FC1 GEMM optimization

Isolated FC1 shape: **M=512, K=768, N=3072**, FP16. A Triton GEMM was benchmarked against PyTorch/cuBLAS, with tile shape, number of warps, pipeline stages, and grouping varied.

Selected historical observations:

- Initial cuBLAS / PyTorch: about **0.0639 ms**, ~37.8 TFLOP/s.
- Initial Triton: about **0.0664 ms**, ~36.36 TFLOP/s; correctness matched the tested reference.
- One tile/stage sweep measured a **64×128×32** configuration at roughly **0.059087 ms**, ~40.89 TFLOP/s, compared with a ~0.0639 ms reference from a separate measurement setting.
- **Interleaved final comparison** (more suitable for a fair claim): PyTorch median **0.059892 ms**, Triton `64×128_g1` median **0.059697 ms**; roughly **0.3% lower latency**, a difference small enough to warrant caution. Other tested Triton configurations were close. Correctness passed with reported max absolute error 0.0 in that test.
- A controlled QKV/FC2 autotuning study was added through `scripts/transformer_projection_autotune.py`, testing 72 configurations per projection, validating against PyTorch, alternating CUDA-event measurements, and writing CSV/JSON summaries. Refer to the actual artifacts for final projection-specific gains rather than inventing percentages.

**Important:** Kernel-level changes must be separately measured in the end-to-end Transformer workload before claiming token-throughput improvement.

### PTX / machine-code investigation

- CUDA source was compiled with `nvcc -O3 -arch=sm_89 -ptx` and `-cubin`, producing PTX and CUBIN for inspection.
- Explored PTX labels, predicates, branching, address calculations, and the relationship between high-level GEMM, PTX, and Tensor Core execution.
- Potential future Matmul Inspector feature: automated PTX inspection, static instruction categories, register/shared-memory-related analysis when available, and diffs across kernel variants. Avoid conflating PTX with final SASS machine instructions.

## 4. AWS Trainium1 / NKI track

### Basic kernel architecture

The tiled GEMM accepts `a_t` in transposed shape `[K, M]` and `b` in shape `[K, N]`, producing an FP32 output `[M, N]`. It uses:

- HBM/shared HBM for tensors and output.
- SBUF tiles for operands, loaded via `nisa.dma_copy`.
- `nisa.nc_matmul` with a stationary A tile and moving B tile.
- FP32 accumulation in PSUM using `accumulate=(ki > 0)`.
- `nisa.tensor_copy` from PSUM to SBUF and DMA to HBM for output.

The established N=512 kernel uses `TM=128`, `TN=512`, `TK=128`. Preserve original kernels as benchmarks rather than overwriting them. NKI source is not a sufficient proxy for actual machine behavior: compare NEFF, MLIR, hardware instruction statistics, and measured latency.

### Correctness and early experiments

- Initial 16×16 NKI GEMM passed correctness (max absolute error around `1.43e-06`).
- Tiled 512×512 GEMM ran correctly on physical Trainium1.
- BF16 input with FP32 PSUM accumulation and FP32 output passed tests against the BF16-rounded-input reference. An earlier 512×512 test reported max kernel/reference error `2.4795532e-05` and mean `1.9467248e-06`.
- Distinguish **kernel error versus BF16-rounded inputs** from **precision loss versus original FP32 inputs**.

### Explicit A-tile reuse experiment — completed

**Hypothesis:** Preloading A tiles across adjacent output N tiles could reduce redundant transfers or improve scheduling, using the original 128×128×128 geometry.

**Finding:** The revised kernel passed correctness but reported the same **40 µs NeuronCore p50** as the original at reporting precision. Host-visible runtime p50 was **94 µs versus 92 µs**, insufficient to establish a gain or regression. Compiler `dma_stats.txt` and `instruction_stats.txt` matched (including 128 MATMUL, 128 LDWEIGHTS, 16 COPY, 22 EVENT_SEMAPHORE). NEFF hashes and MLIR source differed. **Do not repeat this experiment without a new mechanism or better measurement.**

### BF16 precision and output-tile geometry — completed October 9

Controlled **512×512** GEMM study with `TM=128`, `TK=128`, and `TN=128/256/512`. One NeuronCore, `neuron-bench exec`, 20 warmups, 200 work iterations, five randomized rounds for the four primary configurations.

| Input type | TM×TN×TK | Mean run-level NeuronCore p50 |
|---|---|---:|
| FP32 | 128×128×128 | 40.2 µs |
| BF16 | 128×128×128 | 38 µs |
| FP32 | 128×512×128 | 28 µs |
| BF16 | 128×512×128 | **25 µs** |

An initial BF16 `TN=256` run measured 29 µs median NeuronCore latency.

**Strongest result:** FP32 `TN=128` to BF16 `TN=512`: **1.608× speedup**, **37.8% lower latency**, and calculated effective arithmetic throughput **~10.74 TFLOP/s** (`2 × 512³ / 25 µs`). The figure is an arithmetic-throughput estimate, not hardware peak-utilization measurement.

**Interpretation:** Widening the output N tile was a larger performance lever than changing precision alone on this exact workload. The optimum is not established for other matrix sizes or hardware.

Selected Neuron Explorer results for 512×512:

| Metric | BF16 N=128 | BF16 N=512 | FP32 N=512 |
|---|---:|---:|---:|
| MATMUL instructions | 64 | 16 | 32 |
| Tensor Engine active time | 7.79 µs | 5.85 µs | 14.37 µs |
| DMA active time | 22.68 µs | 14.91 µs | 17.47 µs |
| GPSIMD active time | 23.12 µs | 3.96 µs | 9.43 µs |
| HBM reads | 1 MiB | 1 MiB | 2 MiB |
| HBM writes | 1 MiB | 1 MiB | 1 MiB |
| Profile total execution time | 31.60 µs | 17.81 µs | 21.46 µs |

**Caveat:** Engine-active windows can overlap and must **not** be added together to estimate wall-clock kernel time. Profiler `total_exec_time` and `neuron-bench` NeuronCore latency are distinct measurement windows. Zero run-level SD at whole-microsecond resolution does not imply zero timing variability.

### 1024×1024 precision comparison — profiling completed, analysis ongoing

The 1024 scripts call the shared tiled kernel with `TN=512`; FP32 and BF16 variants supply different input types. Instruction breakdown:

| Hardware opcode | FP32 | BF16 |
|---|---:|---:|
| MATMUL | 256 | 128 |
| LDWEIGHTS | 256 | 80 |
| EVENT_SEMAPHORE | 37 | 36 |
| COPY | 16 | 16 |
| UNKNOWN(0xd4) | 106 | 112 |

These are **instruction counts, not time or bytes**. The `UNKNOWN(0xd4)` opcode remains unidentified. Files include:

- `scripts/nki_matmul_fp32_1024_n512.py`
- `scripts/nki_matmul_bf16_1024_n512.py`
- `artifacts/nki-fp32_1024_n512/sg00/instruction_stats.txt`
- `artifacts/nki-bf16_1024_n512/sg00/instruction_stats.txt`

### Logical K-blocking experiment — completed October 10

**Hypothesis:** Grouping K iterations might improve scheduling or accumulation overhead without changing physical TK.

**Comparison:** BF16 1024×1024 GEMM, `TM=128`, `TN=512`, `TK=128`:

- Baseline: eight consecutive physical K iterations per output tile.
- Candidate: four logical groups, two physical K iterations per group.
- `scripts/nki_k_blocking_experiment.py` and `scripts/nki_k_blocking_sweep.py`.
- Correctness: **PASS** vs FP32 matmul of BF16-rounded operands (`rtol=1e-3`, `atol=1e-3`); max absolute error **5.340576e-05**, mean **3.5062462e-06**.
- Controlled benchmark: five randomized rounds, 20 warmups and 200 work iterations per run, one core; **93 µs NeuronCore p50 in all rounds for both variants**. Observed speedup **1.000×**, no detectable difference at reported timing resolution.
- Both variants: **128 MATMUL, 80 LDWEIGHTS, 36 EVENT_SEMAPHORE, 16 COPY** and matching other opcode counts; `dma_stats.txt` files identical.
- NEFF SHA-256 hashes **differed**, so artifacts are not byte-identical.
- Initial `module.mlir` textual diff: **1,680 lines**. After stripping nested source-location metadata and normalizing function names, **normalized MLIR matched exactly (0 diff lines)**.

**Conclusion:** The captured intermediate operations are equivalent after normalization, and no hardware latency benefit was demonstrated. Identical normalized MLIR and instruction counts **do not prove** final executable instruction streams or schedules are byte-identical.

Relevant artifact locations:

```text
artifacts/nki-bf16_1024_n512/
artifacts/nki-bf16_1024_kblocked/
artifacts/nki-k-blocking-sweep-20261010T132953Z/
```

### Neuron Explorer / Perfetto investigation

Hardware traces were previously explored in Perfetto (trace file reportedly named `profile.pftrace`), including DMA and Tensor Engine overlap, synchronization events, and instruction timing. Prior discussion mentioned a small DMA transfer and partial overlap with compute; **reopen the saved trace before treating exact timing/dependency claims as verified**. Observed overlap alone does not establish that a specific load lies on the kernel's critical path or causes stalls.

**Next diagnostic goal:** Use timeline evidence to test whether matmul execution is blocked by DMA, synchronization, or another dependency, instead of proposing prefetching/double buffering without evidence.

## 5. Repository and workflow state as of October 10

- AWS branch was **`nki-bf16-optimization`** during the latest experiments.
- An October 10 commit used the message **`Add Trainium BF16 optimization and K-blocking analysis`**.
- A subsequent GitHub pull request view showed unresolved conflicts in **`.gitignore` and `README.md`**, with several failing CPU/CUDA checks and one passing CPU configuration. **Do not assume this PR is merged or CI is fixed**; inspect current GitHub status.
- The repository also has/had branches including `main`, `nki-gemm`, `nki-tiled-gemm`, `transformer-workload`, and other optimization branches. Confirm availability and current tips with Git before checking out or merging.
- Representative documentation: `docs/trainium1_nki_gemm_case_study.md`.
- Store core source and concise documentation in Git; large compiler outputs, NEFFs, `.ntff`/`.pftrace` traces, and repeated benchmark JSON belong in artifacts or appropriate external storage unless intentionally versioned.

Useful commands:

```bash
cd ~/Matmul-Inspector
git status --short
git branch --show-current
git fetch origin
```

Do **not** change branches, merge, overwrite kernels, or stage ignored artifacts without confirming state and intent.

## 6. Research backlog and suggested priorities

1. **Resolve/review open PR and CI failures** (separate infrastructure/build issues from NKI changes).
2. **Complete evidence-led Trainium timing investigation:** reopen existing Neuron Explorer/Perfetto traces and identify scheduling dependencies or true stalls before implementing DMA-overlap changes.
3. **Scaling tests:** repeat controlled precision/tile comparisons at 1024×1024 and 2048×2048, tracking correctness and NeuronCore metrics. Do not presume the 512×512 winner stays optimal.
4. **Inspector automation:** parse `neuron-bench` JSON/CSV, Neuron Explorer instruction/DMA statistics, and normalized MLIR; produce side-by-side evidence reports and flag source changes erased by compilation.
5. **GPU track:** automate PTX inspection, continue Transformer projection measurements, and only claim end-to-end token gains when measured.
6. **Packaging:** create/finish a Python package with a reproducible CLI, tests, sensible optional CUDA/Neuron extras, and `pip install` workflow.
7. **Open source:** study and contribute to relevant issues or reviews in FlashAttention, FlashInfer, or vLLM.

These are priorities to revisit, not assertions of work currently underway.

## 7. Collaboration and evidence rules

- Start new tasks by checking this context, relevant case-study Markdown, and live Git state. Don't ask to repeat completed experiments without a reason.
- State **hypothesis → controlled change → correctness → measurement → compiler/profile inspection → conclusion**.
- Report workload shape, data type, tile sizes, device/core count, compiler/SDK when available, repetitions, and measurement type.
- Do not mix 512×512 and 1024×1024 latency results, nor host runtime with NeuronCore or profiler-window timings.
- Preserve null results (explicit A reuse, K blocking) as valuable experimental outcomes.
- Separate observed hardware facts from plausible explanations (operand reuse, synchronization, caching, and compiler canonicalization).
- Use short, runnable terminal commands; label Python snippets meant for scripts distinctly from Bash commands.
- Prefer one actionable experiment at a time, with clear checkpoints before destructive operations or prolonged EC2 usage.
- Keep this file compact enough for future retrieval; detailed raw metrics should stay in experiment-specific case studies and artifacts.

## 8. Context maintenance

After each significant experiment, update:

1. The authoritative repository code and test/benchmark outputs.
2. The relevant experiment-specific Markdown case study.
3. This context document's completed findings, current branch/PR state, and next decision.

**This is a snapshot, not a live sync with GitHub or the EC2 instance.** Re-upload or refresh the file in the ChatGPT Project when it changes. Never infer that a prior branch, machine address, benchmark file, or PR status is still current.
