# Controlled FC1 benchmark `controlled-8a29d7fe84994abe88b44164feecdb0f`

Workload: M=512, N=3072, K=768, precision=fp16. Target: sm_89. Detected GPU: NVIDIA GeForce RTX 4060 Ti.

| Implementation | Median latency (us) | Speedup vs baseline | Correctness |
|---|---:|---:|---|
| PyTorch/cuBLAS | 60.467 | 1.000× | Reference |
| `candidate-09991c7d870c92ae572f` | 60.129 | 1.007× | PASS |
| `candidate-086fcf07d56d98d3387d` | 59.981 | 1.009× | PASS |
| `candidate-8e3ff66de13a408caa23` | 60.220 | 1.007× | PASS |
| `candidate-2c7ad186cbae1bfb44ed` | 61.973 | 0.988× | PASS |
| `candidate-8ac12f04c00378f1a984` | 59.774 | 1.011× | PASS |

## Method

10 independently shuffled rounds; 3 samples per implementation per round; 100 launches per sample. CUDA events provide device elapsed time. Host submission and synchronized host-wall clocks remain separate in the raw manifest.

## Fastest measured candidate

`candidate-8ac12f04c00378f1a984` at 59.774 us median CUDA-event latency.

This ranking covers only successfully measured candidates in this experiment. It is not a global-optimum claim.

## Variability and limitations

The JSON manifest contains raw observations, per-round medians, sample standard deviations, and paired round speedups. Small differences are descriptive and are not claimed to be statistically meaningful. GPU clocks, thermals, background activity, and implementation-path differences remain possible confounders.

## Search-time discovery versus confirmation

The exploratory search latencies below determined which configurations entered this experiment. They are not confirmation measurements. The controlled results in the table above come only from this randomized, interleaved session.

Each distinct configuration was confirmed once. Repeated search discoveries are provenance, not independent confirmation samples.

- `candidate-09991c7d870c92ae572f`: trial-0-86c33b835465265d uniform at budget 12 (60.219 us exploratory), trial-0-3a2400ed63af34a5 uniform at budget 13 (59.399 us exploratory)
- `candidate-086fcf07d56d98d3387d`: trial-0-86c33b835465265d adaptive at budget 11 (60.812 us exploratory)
- `candidate-8e3ff66de13a408caa23`: trial-0-fa0b3bb94598ecc4 uniform at budget 12 (61.197 us exploratory)
- `candidate-2c7ad186cbae1bfb44ed`: trial-0-fa0b3bb94598ecc4 adaptive at budget 15 (61.394 us exploratory)
- `candidate-8ac12f04c00378f1a984`: trial-0-3a2400ed63af34a5 adaptive at budget 13 (60.280 us exploratory)

## Artifacts

- Raw manifest: `experiment.json`
- Candidate evaluations: `evaluations.jsonl`
