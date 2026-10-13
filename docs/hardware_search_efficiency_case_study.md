# Uniform versus adaptive Triton search on RTX 4060 Ti

## Summary

This case study compares uniform random sampling with the parameter-weighted
adaptive sampler in Matmul Inspector. It uses three paired hardware-search
trials for the FP16 Transformer FC1 GEMM:

```text
M=512, N=3072, K=768
```

Both arms received the same five initial candidates in each trial. Each arm then
continued to a total budget of 16 candidate submissions. Shared candidates were
executed once and credited to both arms. Later uniform and adaptive candidates
used independent random streams.

Uniform search found the lower final exploratory latency in all three trials.
Adaptive search did not win. It still behaved correctly. It rebuilt from only
its own eligible history, changed its sampling distribution, preserved
exploration, respected the budget, and reproduced every proposal from the saved
evidence and seed. That is an important result. A search strategy must be
auditable when it loses, not only when it finds the fastest candidate.

These are search-time discovery measurements. They are not confirmed kernel
performance results. A later randomized, interleaved experiment confirmed the
five distinct arm winners in one controlled session. Its results are reported
separately below.

## Experiment design

The settings were frozen before the full-budget trials:

| Setting | Value |
|---|---:|
| Hardware | NVIDIA GeForce RTX 4060 Ti, SM 8.9 |
| Precision | FP16 |
| Search-space size | 48 valid configurations |
| Trial seeds | 101, 202, 303 |
| Budget per arm | 16 logical submissions |
| Shared initial candidates | 5 |
| Adaptive batch size | 2 |
| Adaptive epsilon | 0.25 |
| Adaptive prior strength | 2.0 |
| Warmup launches per candidate | 100 |
| CUDA-event samples per candidate | 10 |
| Launches per sample | 500 |
| Reported candidate statistic | Median CUDA-event latency |
| Predefined threshold | 62 µs |

Every successful physical evaluation retained ten raw CUDA-event samples. Each
reported median represents 5,000 timed launches. Compilation, first use,
correctness validation, and host timing were kept separate from CUDA-event
kernel latency.

The scheduling seed selected which arm ran first in the first paired step. The
runner then alternated the first arm. Work was sequential. This counterbalanced
a fixed arm-order bias but did not eliminate clock, thermal, background-load, or
compilation-cache effects.

## Environment

All three trials recorded the same environment:

| Component | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Compute capability | 8.9 |
| GPU memory | 17,175,150,592 bytes |
| NVIDIA driver | 616.92 |
| Python | 3.13.11 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| CUDA runtime | 12.8 |

## Final search results

| Seed | Uniform best | Adaptive best | Adaptive − uniform | Lower final observation |
|---:|---:|---:|---:|---|
| 101 | 60.219393 µs | 60.812321 µs | +0.592928 µs | Uniform |
| 202 | 61.197247 µs | 61.393826 µs | +0.196579 µs | Uniform |
| 303 | 59.399168 µs | 60.279808 µs | +0.880640 µs | Uniform |

The mean paired adaptive-minus-uniform difference was about +0.557 µs. This is
a descriptive summary of three trials. It is not a confidence interval,
significance test, or general search-efficiency claim.

Absolute latency also shifted between trials. The lowest recorded uniform
median ranged from 59.399 to 61.197 µs. This is one reason not to treat candidates
measured in different search sessions as a matched performance comparison.

## Winning configurations

### Seed 101

Uniform found its winner at logical budget 12:

```text
candidate: candidate-09991c7d870c92ae572f
evaluation: evaluation-cda7cf5541f147c4a105de15d826fc50
block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3
latency=60.219393 µs
```

Adaptive found its winner at logical budget 11:

```text
candidate: candidate-086fcf07d56d98d3387d
evaluation: evaluation-f66e185b3a01416eb7e777ea02ce86b4
block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=2
latency=60.812321 µs
```

Adaptive crossed 62 µs at budget 6. Uniform crossed it at budget 12. Adaptive
therefore found a threshold-qualified candidate earlier, even though uniform
finished with the lower observed minimum.

### Seed 202

Uniform found its winner at logical budget 12:

```text
candidate: candidate-8e3ff66de13a408caa23
evaluation: evaluation-c7fd2a01c2f043778599a99bbd6daee4
block_m=64, block_n=128, block_k=32, num_warps=4, num_stages=3
latency=61.197247 µs
```

Adaptive found its winner at logical budget 15:

```text
candidate: candidate-2c7ad186cbae1bfb44ed
evaluation: evaluation-5de45c3a1e7d466dbc6be120baf3983c
block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=2
latency=61.393826 µs
```

Both arms crossed 62 µs at budget 4 through the shared initialization. The final
difference was only 0.197 µs. It should be treated as inconclusive without a
matched confirmation run.

### Seed 303

Uniform found its winner at logical budget 13:

```text
candidate: candidate-09991c7d870c92ae572f
evaluation: evaluation-2bd2ffdb9a4448169617dba235563d78
block_m=128, block_n=64, block_k=32, num_warps=8, num_stages=3
latency=59.399168 µs
```

Adaptive found its winner at logical budget 13:

```text
candidate: candidate-8ac12f04c00378f1a984
evaluation: evaluation-986cfc612ed440ceab58de3683025949
block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3
latency=60.279808 µs
```

Both arms crossed 62 µs at budget 4 through the shared initialization.

## Best-so-far discovery curves

The tables below preserve the best observed latency after each logical
submission. Equal values across the first five budgets come from shared physical
evaluations.

### Seed 101

| Budget | Uniform best | Adaptive best |
|---:|---:|---:|
| 1 | 64.712704 | 64.712704 |
| 2 | 64.712704 | 64.712704 |
| 3 | 64.712704 | 64.712704 |
| 4 | 64.313343 | 64.313343 |
| 5 | 64.313343 | 64.313343 |
| 6 | 64.313343 | 61.542400 |
| 7 | 62.631742 | 61.542400 |
| 8 | 62.631742 | 61.542400 |
| 9 | 62.631742 | 61.542400 |
| 10 | 62.631742 | 61.542400 |
| 11 | 62.631742 | 60.812321 |
| 12 | 60.219393 | 60.812321 |
| 13 | 60.219393 | 60.812321 |
| 14 | 60.219393 | 60.812321 |
| 15 | 60.219393 | 60.812321 |
| 16 | 60.219393 | 60.812321 |

### Seed 202

| Budget | Uniform best | Adaptive best |
|---:|---:|---:|
| 1 | 64.879616 | 64.879616 |
| 2 | 64.231136 | 64.231136 |
| 3 | 64.231136 | 64.231136 |
| 4 | 61.415424 | 61.415424 |
| 5 | 61.415424 | 61.415424 |
| 6 | 61.415424 | 61.415424 |
| 7 | 61.415424 | 61.415424 |
| 8 | 61.415424 | 61.415424 |
| 9 | 61.415424 | 61.415424 |
| 10 | 61.415424 | 61.415424 |
| 11 | 61.415424 | 61.415424 |
| 12 | 61.197247 | 61.415424 |
| 13 | 61.197247 | 61.415424 |
| 14 | 61.197247 | 61.415424 |
| 15 | 61.197247 | 61.393826 |
| 16 | 61.197247 | 61.393826 |

### Seed 303

| Budget | Uniform best | Adaptive best |
|---:|---:|---:|
| 1 | 74.780579 | 74.780579 |
| 2 | 64.306110 | 64.306110 |
| 3 | 64.306110 | 64.306110 |
| 4 | 61.015841 | 61.015841 |
| 5 | 61.015841 | 61.015841 |
| 6 | 61.015841 | 61.015841 |
| 7 | 61.015841 | 61.015841 |
| 8 | 61.015841 | 61.015841 |
| 9 | 61.015841 | 61.015841 |
| 10 | 61.015841 | 61.015841 |
| 11 | 61.015841 | 61.015841 |
| 12 | 61.015841 | 61.015841 |
| 13 | 59.399168 | 60.279808 |
| 14 | 59.399168 | 60.279808 |
| 15 | 59.399168 | 60.279808 |
| 16 | 59.399168 | 60.279808 |

## Correct adaptive behavior without a win

Each trial performed six adaptive rebuilds. The cumulative feedback sizes were
5, 7, 9, 11, 13, and 15 evaluations. Across all 18 rebuilds:

- every input evaluation was eligible;
- no observation was excluded;
- no failed evaluation entered the reward calculation;
- no uniform-only evaluation entered adaptive feedback;
- every feedback digest matched the referenced records;
- every saved proposal was reproduced from its seed and cumulative history;
- epsilon remained 0.25 and prior strength remained 2.0.

The learned distributions changed without collapsing. Normalized probability
entropy stayed between approximately 0.970 and 0.995. The largest candidate
probability observed in any rebuild was about 4.82%, compared with 2.08% under a
uniform distribution over 48 candidates. Exploration remained meaningful.

This separates two questions:

1. Did the adaptive mechanism operate as designed? **Yes.**
2. Did it find the lowest search-time latency in these trials? **No.**

That distinction is central to the experiment. Correct mechanics do not imply a
performance advantage. A negative result can still validate the framework and
expose limitations in the search model.

The current parameter-wise model scores values independently. It cannot fully
represent interactions between tile dimensions, warp count, and pipeline stages.
The broad distributions also show that 15 observations did not create a sharply
concentrated policy. Both points are plausible reasons to study, but neither is
established as the cause of the result.

## Accounting and failure checks

| Quantity | Per trial | Three-trial total |
|---|---:|---:|
| Logical submissions | 32 | 96 |
| Physical evaluations | 27 | 81 |
| Shared physical evaluations | 5 | 15 |
| Successful physical evaluations | 27 | 81 |
| Failed physical evaluations | 0 | 0 |
| Adaptive rebuilds | 6 | 18 |

Every arm evaluated 16 unique candidates. Candidate identities recomputed from
their configurations and workload context. Every evaluation reference resolved
across `experiment.json`, `evaluations.jsonl`, and `progress.json`. Reported
latencies matched the median of the stored raw samples. All candidates passed
correctness validation.

The recorded search wall times were:

- seed 101: 14.673 seconds;
- seed 202: 14.674 seconds;
- seed 303: 12.312 seconds.

These durations include search orchestration and candidate evaluation. They are
not kernel latencies and should not be used as a throughput comparison.

## Controlled winner confirmation

The confirmation experiment collected the six arm winners from the three
search trials and deduplicated them by candidate identity. The uniform winner
from seeds 101 and 303 was the same configuration, so it was confirmed once.
The resulting controlled session contained five candidates plus one shared
PyTorch/cuBLAS baseline.

The session used 10 independently shuffled rounds. Every round measured all six
implementations in a randomized order. Each implementation received three
CUDA-event samples per round and 100 kernel launches per sample. Compilation,
first-use setup, and 100 warmup launches were excluded. All implementations
reused the same input tensors. Host submission and synchronized host-wall times
were recorded separately from device timing.

| Confirmation setting | Value |
|---|---:|
| Experiment ID | `controlled-8a29d7fe84994abe88b44164feecdb0f` |
| Input seed | 0 |
| Scheduling seed | 8,675,309 |
| Rounds | 10 |
| Samples per implementation per round | 3 |
| Launches per sample | 100 |
| Warmup launches | 100 |
| Absolute tolerance | 0.01 |
| Relative tolerance | 0.01 |

| Implementation | Search source | Confirmation median | Paired-round median speedup vs baseline | Correctness |
|---|---|---:|---:|---|
| PyTorch/cuBLAS | Shared baseline | 60.467200 µs | 1.0000× | Reference |
| `candidate-09991c7d870c92ae572f` | Uniform, seeds 101 and 303 | 60.128961 µs | 1.0162× | PASS |
| `candidate-086fcf07d56d98d3387d` | Adaptive, seed 101 | 59.980640 µs | 1.0049× | PASS |
| `candidate-8e3ff66de13a408caa23` | Uniform, seed 202 | 60.219841 µs | 1.0044× | PASS |
| `candidate-2c7ad186cbae1bfb44ed` | Adaptive, seed 202 | 61.973119 µs | 1.0007× | PASS |
| `candidate-8ac12f04c00378f1a984` | Adaptive, seed 303 | **59.773760 µs** | **1.0107×** | PASS |

The fastest overall confirmation median belonged to the adaptive seed-303
winner:

```text
candidate: candidate-8ac12f04c00378f1a984
confirmation evaluation: evaluation-04844b4d26f843a7b9fafb8cb380f11a
block_m=128, block_n=64, block_k=32, num_warps=4, num_stages=3
median CUDA-event latency=59.773760 µs
```

This was about 0.693 µs below the baseline median in this session. It was not
the lowest search-time observation. Conversely, the configuration with the
lowest search-time observation did not have the lowest confirmation median.
That reversal is exactly why search-time discovery and confirmed performance
must remain separate.

All five candidates passed the shared-input PyTorch comparison with the stated
absolute and relative tolerances of 0.01. There were no compilation,
correctness, benchmark, or infrastructure failures. Each implementation has 30
raw CUDA-event observations and 10 paired round comparisons. All 10 rounds used
different execution orders.

The confirmation used the same GPU and software stack as the searches: RTX 4060
Ti, SM 8.9, NVIDIA driver 616.92, PyTorch 2.11.0+cu128, Triton 3.6.0, and CUDA
runtime 12.8.

These results do not establish a stable speedup. The gaps are small relative to
the recorded variability. The paired-round summaries also do not preserve the
same ordering as the overall medians. More independent controlled sessions
would be needed before making a performance claim.

## Why search discovery is not confirmed performance

The search runner measured each candidate when it was proposed. Those
measurements were sufficient to update the search policy and construct
best-so-far curves. They were not a matched final comparison among the winners.

In particular:

- winners were measured in different sessions and at different physical orders;
- absolute latency shifted across trials;
- the candidate set changed with each search path;
- no common PyTorch/cuBLAS baseline was interleaved with these searches;
- the later randomized confirmation is a separate experiment and cannot be
  substituted back into the search policy's historical observations.

The statement supported by this evidence is:

> Uniform search produced the lower observed search-time minimum in each of the
> three paired trials. Adaptive search found a threshold-qualified candidate
> earlier in seed 101 and otherwise behaved according to its recorded contract.

The search evidence does **not** establish that the uniform winners are reliably
faster than the adaptive winners. The separate confirmation found the lowest
overall median for an adaptive-discovered configuration, but its small margin is
descriptive rather than a confirmed speedup claim.

## Limitations

- There are only three trial seeds.
- All trials use one workload, precision, search space, GPU model, and software stack.
- Shared initialization correlates the two arms by design.
- Later measurements are sequential, not simultaneous.
- CUDA-event timing can still move with clocks, thermals, and background activity.
- Compilation caching and physical evaluation order can affect search conditions.
- No failures occurred, so real-hardware failure imbalance was not exercised here.
- The 62 µs threshold was fixed, but two trials crossed it during shared initialization.
- The adaptive model cannot express general parameter interactions.
- The experiment records descriptive outcomes. It does not provide statistical power
  for a broad superiority claim.
- The confirmation consists of one controlled session with 10 paired rounds.
- No confidence interval or hypothesis test was prespecified for confirmation.
- The confirmation ranking changes depending on whether overall medians or paired
  round summaries are considered.

## Evidence

Seed 101:

- [`experiment.json`](../artifacts/hardware-search/fc1-pilot-seed101/experiment.json)
- [`evaluations.jsonl`](../artifacts/hardware-search/fc1-pilot-seed101/evaluations.jsonl)
- [`progress.json`](../artifacts/hardware-search/fc1-pilot-seed101/progress.json)
- [`report.md`](../artifacts/hardware-search/fc1-pilot-seed101/report.md)

Seed 202:

- [`experiment.json`](../artifacts/hardware-search/fc1-pilot-seed202/experiment.json)
- [`evaluations.jsonl`](../artifacts/hardware-search/fc1-pilot-seed202/evaluations.jsonl)
- [`progress.json`](../artifacts/hardware-search/fc1-pilot-seed202/progress.json)
- [`report.md`](../artifacts/hardware-search/fc1-pilot-seed202/report.md)

Seed 303:

- [`experiment.json`](../artifacts/hardware-search/fc1-pilot-seed303/experiment.json)
- [`evaluations.jsonl`](../artifacts/hardware-search/fc1-pilot-seed303/evaluations.jsonl)
- [`progress.json`](../artifacts/hardware-search/fc1-pilot-seed303/progress.json)
- [`report.md`](../artifacts/hardware-search/fc1-pilot-seed303/report.md)

Controlled confirmation:

- [`experiment.json`](../artifacts/hardware-search/fc1-winner-confirmation/experiment.json)
- [`evaluations.jsonl`](../artifacts/hardware-search/fc1-winner-confirmation/evaluations.jsonl)
- [`report.md`](../artifacts/hardware-search/fc1-winner-confirmation/report.md)

## Current conclusion

Uniform search produced the lower exploratory minimum in all three paired
trials. Adaptive sampling still behaved correctly and contributed the
configuration with the lowest median in the later controlled session. Neither
result establishes that one search strategy is generally superior. Together,
they show that the framework can preserve a losing adaptive-search result,
deduplicate repeated discoveries, and test the resulting candidates under a
matched confirmation protocol.
