# MI-Research-10 — Prospective slice-aware screening

> **Recovery note:** This is the original Codex report, preserved as historical
> text. The isolated recovery checkout has the regenerated source and frozen
> plan, but lacks the original GPU and final analysis artifacts. Its local plan
> hash differs from the original recorded hash. See the
> [recovery checkpoint](mi_research_10_recovery.md) before using the evidence
> or replay commands below.

## Decision

**B — Useful partial predictions with an important preregistration discrepancy.** The frozen source descriptors correctly identified equal four-CTA dot work and a 12-gate-per-grid increase for the uncompiled 58-addition circuit. Independent TTIR and PTX retained the equal dot/MMA counts and showed a small increase in total static FP32 arithmetic text. The compiler reported 28 more spills for 58. One matched CUDA Graph session was `INCONCLUSIVE`. No latency advantage is claimed.

The [frozen plan](../artifacts/mi-research-10/frozen_plan.json) contains a contradiction: its descriptor correctly records **63,488** modeled live bytes per CTA for 58 versus **59,392** for 55, but its prose incorrectly calls 58 **4,096 fewer**. The plan was committed before compilation at `5df762ae9e6732a19a899d1076a67d0d02b07ab0` and was not edited afterward. The prose error means this study cannot claim a clean prospective liveness-to-spill prediction. The [analysis audit](../artifacts/mi-research-10/analysis.json) records the discrepancy.

## Checkpoint, hypothesis, and candidate

Remote GitHub `main` was verified at `ff55666ea0dff33bdb5ee19061c5cd950ee60be9`, the merge of MI-Research-9 after the through-8 merge. The local `origin/main` in the user's existing checkout was stale at `5df8f5b`; a malformed unrelated Codex ref prevented fetch. Work used a clean isolated clone and branch `feat/mi-research-10-prospective-screening`, leaving that checkout and its uncommitted files untouched. The MI-Research-9 hash manifest and CPU replay passed before the freeze.

The hypothesis was that output-slice source work and liveness can screen compiler-resource tradeoffs before compiling a new combination. It did not assume a latency model. The certified 55- and 58-addition Perminov circuits have identical rank-23 factor maps and reconstruct all 729 ordered 3×3 block tensor coefficients exactly. The retained source `perminov_source.json` SHA-256 is `80bb4187ddf0cfc04654ca3e524f4d311c5a62245ee231a1101280b43fd2ae3a`; the certificate SHA-256 is `91c36b1165a71d4ba78cd8eb9c51ef9666d05e9162fdef5cba28d1d697a17dff`. All input hashes are in the frozen plan.

Only one circuit/mapping combination was previously uncompiled among the two certified circuits and the established four- and 36-CTA output ownership definitions: **58 additions, four CTAs**. The 55-addition four-CTA kernel was the retained compiler control. The two 36-CTA combinations were previously measured and were not selected as prospective candidates. Historical latency was not used in selection or ordering. The new source SHA-256 is `fa0715e83b773816966ec70ea6f536d16d0f9ca72be0753ed16875bb504d0cd0`. The generator reproduces the historical four-CTA 55 source byte for byte before emitting 58.

Each CTA owns the same local 16×16 tile across all nine output blocks. The grid is `(2,2,1)`, with 128 threads per CTA, four warps, two stages, FP16 inputs/output, FP32 forms/products, and `tf32x3` dots. There are no atomics or explicit product buffers. The source work counts include all four CTAs; duplicate counts refer to repeated expressions from output slicing at a single tile position.

| Frozen source descriptor | 55 control | 58 prospective |
| --- | ---: | ---: |
| Whole-circuit additions | 55 | 58 |
| Output-sliced gate instances/grid | 220 | 232 |
| Output-sliced dot instances/grid | 92 | 92 |
| Duplicated products/grid | 0 | 0 |
| Peak modeled live bytes/CTA | 59,392 | 63,488 |
| Peak modeled intermediates/CTA | 17 | 18 |
| Dependency path depth | 15 | 12 |
| CTA count | 4 | 4 |

Thus the true frozen vector poses a tradeoff: 58 has a shorter path, while 55 has fewer gates and lower modeled live storage. The plan's prose inverted the storage comparison. These bytes count logical FP32 arrays under one source order. They exclude layout replication and compiler temporaries; they are not physical register allocation.

## Independent compiler validation

The only new prospective combination compiled was four-CTA 58. The matched timing session also compiled the predeclared historical 55 control. The compiler comparison uses retained 55 artifacts from MI-Research-7B, whose PTX hash matches MI-Research-6C. Both control sessions recorded the same PTX and metadata. The new run used the same RTX 4060 Ti class, CUDA 12.8, PyTorch 2.11.0+cu128, Triton 3.6.0, and Python 3.13.11. The recorded driver was 616.92. TTIR and PTX are retained in the GPU validation bundle.

| Compiler observation | 55 retained control | 58 new | 58 minus 55 |
| --- | ---: | ---: | ---: |
| Static TTIR `tt.dot` | 23 | 23 | 0 |
| Static TTIR `arith.addf` + `arith.subf` | 59 | 60 | +1 |
| Static PTX `mma.sync` | 552 | 552 | 0 |
| Static PTX `add.f32` + `sub.f32` | 1,552 | 1,576 | +24 |
| Registers/thread | 255 | 255 | 0 |
| Compiler-reported `n_spills` | 422 | 450 | +28 |
| Shared memory/CTA | 1,024 B | 1,024 B | 0 |
| Global scratch | 0 B | 0 B | 0 |

The equal source dot work anticipated equal emitted TTIR dots and PTX MMA text. The source gate increase agrees with a small increase in total PTX FP32 add/sub text, though no gate-to-instruction conversion is implied. The compiler changed the add/sub mix: TTIR addf rose by five and subf fell by four; PTX add.f32 rose by 104 and sub.f32 fell by 80. The 58 compiler spill count is higher, consistent in direction with the **correct descriptor vector** but not with the erroneous frozen prose. We do not have a causal explanation for the spill difference. Neither PTX counts nor `n_spills` quantify dynamic instructions or local-memory traffic. PTX is not SASS.

## Correctness and optional performance

The new four-CTA 58 kernel passed two eager executions each on seed 0, seed 1, seed 2, zero, identity, and cancellation inputs. Outputs were poisoned before each check. The independent reference was FP32 matmul of exact FP16 inputs rounded once to FP16, with `atol=rtol=0.01`, highest FP32 precision, TF32 disabled for the reference, and reduced FP16 reduction disabled. The recorded profiler trace contains one physical `_fused` kernel with grid `[2,2,1]` and block `[128,1,1]`. The matched Graph session checked both circuits twice per case in eager and Graph modes before timing.

The one fresh fixed-address CUDA Graph session used 20 complete operations per replay and eight balanced paired rounds. A/A controls preceded each A/B round. The existing clock-aware decision rule was unchanged. Raw CUDA-event, host-clock, A/A, and preflight records are retained.

| Graph timing | 55 control A | 58 candidate B |
| --- | ---: | ---: |
| Median CUDA-event µs/operation | 15.762 | 16.026 |

The median paired A−B difference was **−0.256 µs/operation**; the maximum A/A gap was **0.205 µs/operation**. The clock preflight recorded `RATE_DIFFERENCE_OBSERVED`. The rule returned **`INCONCLUSIVE`**, so this single session does not rank latency or establish a speedup. No performance ordering was used to validate the structural screen.

## Scope, reproduction, and next step

This is one prospective new combination against a historical compiler control. It validates a limited source-dot to compiler-dot correspondence and observes an arithmetic resource tradeoff. The frozen prose error prevents a strong liveness-screening claim. Compiler behavior on another architecture, mapping, or schedule remains unknown. No broader candidate search followed.

The original [evidence directory](../artifacts/mi-research-10/) contained the frozen plan and source, exact provenance, new TTIR/PTX/LLIR, compiler metadata, raw numerical checks, profiler trace, raw paired timing and A/A controls, environment record, analysis, and SHA-256 manifests. Most of those files are unavailable in this recovery checkout. The replay command below cannot pass here without the missing historical evidence. From a checkout with that evidence, reproduce CPU checks with the project environment:

```bash
PYTHONPATH=src:. python scripts/generate_mi_research_10_kernel.py
PYTHONPATH=src:. python scripts/freeze_mi_research_10.py
PYTHONPATH=src:. python scripts/analyze_mi_research_10.py --verify
PYTHONPATH=src:. python -m pytest -q tests/mi_research_9_test.py tests/mi_research_10_test.py
```

The GPU runners refuse existing output directories. They are `scripts/run_mi_research_10.py` and `scripts/time_mi_research_10.py`; use a clean study checkout for a new measurement.

**One next step:** add a CPU freeze audit that checks every directional prose hypothesis against its machine-readable descriptor before allowing GPU compilation. This study's internal contradiction is direct evidence for that gate.
