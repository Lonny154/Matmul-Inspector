# Matmul Inspector — Project Context

**Last verified:** October 18, 2026

**Repository:** `Lonny154/Matmul-Inspector`

**Verified checkout:** `feat/tensor-game-algebraic-w-completion` at `be4bd85`

**Purpose:** Compact persistent context. Detailed methods and raw measurements remain in the linked case studies and artifacts.

## Evidence boundary

This snapshot reports facts verified in the local checkout. It does not imply that an unmerged branch is on `main` or that historical hardware remains available. The remote-tracking `origin/main` ref was `5df8f5b`; local `main` was older at `0f7e57a`. No branch was changed for this update. `Project-Instructions.txt` is not present or tracked in this checkout. Repository rules are in `AGENTS.md`.

## Mission and architecture

Matmul Inspector connects numerical correctness, saved compiler artifacts, and measured performance while keeping each evidence layer separate:

```text
source -> compiler IR / PTX -> SASS / NEFF -> numerical output -> timing/profile
```

- `src/matmul_inspector/` is a CPU-compatible Python toolkit for PTX, SASS, NKI, benchmark, and unified comparison reports.
- Native CUDA/C++ laboratories and scripts support numerical, performance, and reproducibility experiments.
- GPU and Trainium experiments are optional. Ordinary analysis and unit tests do not require accelerator hardware.
- The CPU-only Tensor Game track covers exact 2x2 matrix-multiplication decomposition, bounded search, learning diagnostics, Triton lowering, and algebraic exploration. See `TENSOR_GAME_RESEARCH.md`.

Static instruction counts are not dynamic counts. PTX is not SASS. MLIR equality does not prove identical NEFF scheduling. A faster incorrect candidate is not an optimization.

## Release status

- Git tag `v0.3.0` points to merge commit `e7b3255` from October 15, 2026. `README.md` and `pyproject.toml` identify `matmul-inspector==0.3.0` with Python 3.10–3.13 support.
- Version 0.3.0 provides CPU-only saved-artifact analysis for PTX, SASS, NKI, and unified reports. GPU experiment commands retain optional PyTorch, Triton, CUDA, and driver requirements.
- Release evidence: `docs/release_notes_0_3_0.md` and `docs/python_toolkit_release.md`.
- Tensor Game research merged through PR #23 at `4ac4fd7`; its technical case study merged through PR #25 at `5df8f5b`. The October 16–17 feature sequence from `726599b` through `be4bd85` is on the current feature lineage and is not claimed merged to `origin/main`.

## Completed technical tracks

### CUDA, PTX, SASS, and Transformer

- PTX inspection reports selected entries, virtual registers, shared-memory declarations, normalized text, and bounded diffs. SASS inspection separates kernels from helpers and reports saved `nvdisasm` instructions and compiler register metadata.
- The controlled FMA study found 5 `FFMA` versus 5 `FMUL` plus 5 `FADD`, with 28 registers/thread in both saved `sm_89` builds. Timings changed direction by size and the no-FMA candidate failed configured tolerances at 512 and 1024. It is not a valid optimization. See `docs/cuda_ptx_sass_fma_case_study.md` and `docs/floating_point_experiments.md`.
- The Transformer workload uses batch 1, sequence 512, hidden 768, 12 heads, MLP 3072, and FP16. For isolated FC1 M=512, K=768, N=3072, an interleaved comparison measured PyTorch at 0.059892 ms and Triton `64x128_g1` at 0.059697 ms. The roughly 0.3% kernel difference is not an end-to-end throughput claim. QKV and FC2 work is in `docs/transformer_optimization.md`.
- CUDA Thomas versus PCR evidence is preserved in `docs/cuda_thomas_pcr_case_study.md` and `docs/tridiagonal_benchmarking.md`.

### Trainium1 and NKI

- The NKI GEMM uses transposed A `[K,M]`, B `[K,N]`, SBUF tiles, `nisa.nc_matmul`, FP32 PSUM accumulation, and HBM output. The main 512 geometry is TM=128, TN=512, TK=128.
- Explicit A-tile reuse passed correctness but retained the same 40 us NeuronCore p50. Host p50 was 94 us versus 92 us. No speedup was demonstrated.
- In controlled 512x512 runs, FP32 TN=128 measured 40.2 us and BF16 TN=512 measured 25 us: 1.608x arithmetic speedup and about 10.74 effective TFLOP/s for that workload. This is not a general optimum.
- BF16 1024x1024 logical K blocking and baseline both measured 93 us NeuronCore p50 across five randomized rounds. Normalized MLIR and opcode counts matched, NEFF hashes differed, and no speedup was demonstrated.
- See `docs/trainium1_nki_gemm_case_study.md`, `docs/nki_gemm_benchmarking.md`, and `docs/nki_compiler_artifact_comparison.md` for conditions and artifacts.

### Tensor Game

- Exact conventional rank-8 and Strassen rank-7 references, sampled MCTS, synthetic data, optional learning studies, held-out evaluation, bounded residual search, and extensive negative-result diagnostics are implemented.
- `bounded_128` matched exhaustive proposal coverage at 28/30 states and 31/44 positives with 29.03 ms versus 47.03 ms mean proposal latency. A later readiness study found 63/72 successes for both providers, with bounded 2.16x faster mean search. Exhaustive remained the default.
- Hardware descriptors, Pareto analysis, Triton lowering, TF32x3, input/output fusion, timing stability, GPU cost calibration, change of basis, multi-`w`, quotas, provenance admission, exact feasibility, MITM, algebraic pools, and pair-to-output completion have case studies and raw artifacts.
- No independent rank-7 decomposition was found. The strongest algebraic pool recalled six of seven Strassen mode-pairs and one complete term. Pair-to-output projection reached 2/7 complete-action recall but no MITM complement. See `TENSOR_GAME_RESEARCH.md`.

## Branch and workspace facts

- Current branch: `feat/tensor-game-algebraic-w-completion` at `be4bd85`.
- Named branch tips preserve the October 16–17 hardware-cost, lowering, fusion, timing, calibration, provider, feasibility, MITM, and algebraic studies.
- Existing untracked publication/review material and `controlled_triton_fc1/` predated this refresh. They are unrelated and must remain unstaged.
- Historical branches include `nki-gemm`, `nki-tiled-gemm`, `transformer-workload`, PTX/SASS analysis branches, and `release-0.2.0`. Presence does not imply current or merged status.

## Recommended next work (planned, not completed)

1. Build a bounded joint algebraic decomposition feasibility prototype with SAT/SMT or constraint programming. Start from frozen finite pools. Encode exact seven-term reconstruction and cancellation. Enforce strict CPU, memory, and reproducibility limits. Keep it outside production search until it beats exact controls.
2. Review and merge the October 16–17 Tensor Game chain in coherent units while preserving null results and hashes.
3. Continue Trainium work only from measured scheduling dependencies. Reopen saved traces before proposing overlap changes and repeat scaling before generalizing 512x512 results.
4. Measure CUDA/Triton changes both in isolation and end to end. Continue distinguishing PTX from SASS.
5. Keep the Python package CPU-compatible and hardware paths optional.

## Maintenance rules

- Verify branch, worktree, tags, and remote refs before changing state.
- Preserve benchmarks, artifacts, negative findings, and unrelated changes.
- Keep detailed evidence in case studies and this file compact.
- Report hypothesis, controlled change, correctness, measurement, compiler/profile evidence, and limitations.
- Never claim an optimization, release, merge, or discovery without repository evidence.
