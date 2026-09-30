# Benchmark reliability

Use a Release build and record the complete run directory. Results describe the
recorded implementation and environment, not a universal hardware ranking.

```sh
./build-cuda/matmul-inspector benchmark --sizes 4,64,256 \
  --reference naive --candidate tiled --seed 42 \
  --warmups 3 --iterations 50 --trials 5 \
  --bootstrap-samples 2000 --confidence-level 0.95 --bootstrap-seed 42 \
  --percentiles 5,25,75,95 --output results/reliability
python3 scripts/report.py results/reliability
```

The same controls work with `crossover` and `scripts/crossover.py`. Development
defaults are 3 warmups, 20 measured iterations, 1 trial, 1,000 bootstrap replicates,
95% confidence, and p5/p25/p75/p95. The bootstrap seed defaults to the input seed.
`--bootstrap-samples 0` disables intervals. Compare mode remains untimed and rejects
these timing controls. Replay preserves all controls; replaying performance does
not reproduce the original timing noise.

## Measurement and trial boundaries

A **warmup** is an excluded execution before measurements. Each **trial** repeats
its own warmup block and configured **measurement iterations**. Trials are
sequential blocks within one process, not independent process restarts. All
reference trials precede candidate trials. Crossover runs CPU trials, kernel-only
GPU trials, then end-to-end GPU trials for each ascending size. This fixed order
can confound environmental drift; it is not randomized or a paired statistical
experiment.

GPU CLI benchmarks perform one untimed synchronous priming call per kernel/timing
mode/trial before configured warmups. It excludes initial context/module setup;
it is recorded as policy in metadata, not included in warmup or measurement counts.
Warmups are now timed with the same clock and per-launch synchronization policy as
measurements and retained separately. There is no adaptive warmup algorithm.

- Kernel-only: CUDA events around an individual launch, excluding allocations,
  copies, warmup samples, host analysis, and cleanup.
- CPU: steady clock around the existing single-threaded FP32 matmul, including
  output allocation/initialization.
- GPU end-to-end: steady clock around the synchronous CUDA wrapper, including
  allocation, pageable transfers, execution, synchronization and device frees.

Sample recording, previous host-output destruction, fingerprinting, statistics,
bootstrap and numerical checks are outside measured intervals. Every trial's final
output is checked bitwise against the first trial outside timing, so changing
outputs fail the run. Cross-implementation numerical checks retain their existing
tolerances and diagnostics. The benchmark does not inspect every launch's output.

## Statistics and bootstrap

**Median latency is primary.** Existing mean-based `gflops` and `speedup` summary
columns remain for compatibility; `median_speedup`, pairwise comparisons and
crossover analysis use medians. Overall descriptive statistics pool measured
samples while preserving each trial and its statistics in separate artifacts.

Percentiles use linear interpolation at index `(n-1)*p/100` in sorted samples.
Integer requested percentiles range from 0 through 100. IQR is p75−p25 regardless
of requested percentiles. MAD is the median absolute deviation from the median,
without normal-consistency scaling. Standard deviation uses population denominator
n; CV is standard deviation / mean and is unavailable for a zero mean.

Median confidence intervals use a **percentile bootstrap**:

1. With multiple trials, sample the same number of whole trials with replacement,
   retaining every measured value inside each selected trial, then calculate the
   pooled median. This preserves within-trial correlation. CLI trials have equal
   measurement counts.
2. With a single trial, sample n individual measurements with replacement and
   calculate their median. This assumes exchangeable/IID measurements; correlated
   or drifting samples can make the interval misleading.
3. Repeat `--bootstrap-samples` times. Take interpolated percentiles
   `100*(1-confidence_level)/2` and `100*(1+confidence_level)/2` of bootstrap medians.

MT19937 with an explicit rejection-based integer mapping provides reproducible
resampling from the seed and sample order, without SciPy. Per-trial intervals use
the single-trial method. Intervals are unavailable when disabled or when fewer
than two measurements exist. Constant measurements can produce zero-width
intervals; that does not prove absence of uncertainty outside this capture.

Multiple-trial inference assumes exchangeable trials; sequential trials affected
by common drift may violate this. A small number of trials supplies weak evidence.
These intervals quantify resampling uncertainty under those assumptions, not
universal performance guarantees, prediction intervals, or timer accuracy.

`pairwise.csv` reports candidate/reference median latency ratio, `(ratio-1)*100`
percent difference (negative means faster), reference/candidate median speedup,
and whether the separate median CIs overlap (touching counts as overlap). Missing
CIs remain blank. **Neither CI overlap nor non-overlap alone is a significance
test.** No formal hypothesis test or speedup CI is implemented.

## Conservative heuristic flags

Flags do not fail a benchmark or replace correctness checks:

| Flag | Threshold |
|---|---|
| High variation | CV > 0.20 |
| Wide spread | max/min > 3 |
| Wide middle range | IQR/median > 0.25 |
| Suspicious first measurement | >50% absolute relative difference from remaining median; n≥5 |
| Within-trial drift | >20% absolute relative difference between last-quarter and first-quarter medians; n≥8 |
| Limited evidence | Fewer than 10 measurements or 5 trials |
| Timer resolution | Zero recorded latency |

Zero denominators are not divided. These thresholds are diagnostic choices, not
universal stability criteria. Inspect raw traces and trial-to-trial variation
before making performance claims. No samples or outliers are removed automatically.

## Artifacts and backward compatibility

The root `metadata.json` schema remains **3**, with an additive
`timing_schema_version: "1"` extension. Older runs without samples still render
existing plots; unavailable reliability analyses are skipped with a warning.

- `summary.csv`: existing columns plus row identity, max/IQR/MAD/CV, median CI bounds,
  median speedup and stability flags. One row per kernel/configuration/timing mode.
- `timing_samples.csv`: row ID, M/N/K, backend, kernel/reference, timing mode/clock,
  zero-based trial, zero-based iteration within each phase, warmup/measurement phase,
  latency in milliseconds and input seed. This is the raw measurement record.
- `timing_statistics.csv`: overall and per-trial statistics, sample/warmup/trial
  counts, requested percentiles, bootstrap method, intervals and heuristic flags.
- `pairwise.csv`: median-based comparisons joined by summary row IDs.
- `metadata.json`: statistical controls, methods, thresholds, input seeds,
  environment, timestamp, and source/build provenance. The result directory is
  the run identity; associated CSV files share this metadata rather than repeating
  all hardware fields in every sample.

Raw values are retained without filtering; full-precision serialization permits
later independent analysis. `console.txt`, mismatch/output artifacts and
`crossover.csv` retain their existing roles. Raw samples are held in memory until
artifact writing, so huge trial/iteration counts cost memory and bootstrap time;
interrupted runs may lack partial timing data. Bounds on controls are safety limits,
not recommendations to use maximum values.

Reports add only three reliability figures: measured latency distributions,
warmup/measurement traces retaining trial boundaries, and median CIs with trial
medians. At most six configurations, evenly selected across work-ordered
configurations, appear in those figures; complete CSV data is always retained.

## Practical pitfalls

Keep the machine otherwise idle and inspect both trial traces and run metadata.
GPU initialization and module loading are primed, but clocks/power states can
still change. Thermal throttling, competing GPU work, background CPU load, CPU
frequency scaling, caches and allocation state can shift measurements. Tiny
operations can approach timer/launch overhead. An end-to-end result that includes
allocation and transfer cannot be interpreted as device-resident kernel latency.

Increase trials and measured iterations when practical, repeat captures at
separate times, and compare consistent timing boundaries. More bootstrap
replicates reduce Monte Carlo noise in the interval calculation; they do **not**
create new independent benchmark observations or cure systematic bias.
