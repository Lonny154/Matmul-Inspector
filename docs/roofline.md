# Simple matmul roofline analysis

A roofline model compares measured throughput with two configured ceilings:
how much arithmetic the device could perform per second, and how quickly memory
could supply the modeled bytes. It helps frame a performance question; it does
not identify the real bottleneck by itself.

Analyze a saved run without rebuilding or rerunning CUDA:

```sh
python3 -m pip install -r scripts/requirements-report.txt  # only for --report
python3 scripts/roofline.py --input results/reliability-smoke \
  --peak-fp32-tflops 10 --memory-bandwidth-gbps 100 \
  --compute-source user-supplied --bandwidth-source user-supplied \
  --source-note "Illustrative smoke-test ceilings, not specifications for the captured GPU" \
  --output results/roofline-smoke --report
```

**These example ceilings are assumptions, not RTX 4060 Ti specifications.** Supply
limits appropriate to your experiment. No GPU-name lookup or hardware database is
used. Sources can be labeled `user-supplied`, `documented-vendor-spec`, or
`measured-separately`; use `--source-note` for a citation, measurement procedure,
or caveat. Labels are user declarations, not independently verified evidence.
Use ordinary FP32 arithmetic throughput, not a Tensor Core or mixed-precision rate.

Without `--report`, analysis needs only Python's standard library and emits CSV
and JSON. With it, the existing optional matplotlib dependency adds one roofline
plot and a concise Markdown report. The default output directory is `<input>/roofline`.
Existing directories cannot be overwritten. The source run remains unchanged.

## Formulas and units

For A of shape M×K, B of shape K×N, and C=A×B of shape M×N:

```text
flop_count = 2*M*N*K
model_bytes = 4*(M*K + K*N + M*N)
intensity = flop_count / model_bytes                   [FLOP/byte]
achieved_gflops = flop_count / (latency_ms * 1e6)
model_bandwidth_gbps = model_bytes / (latency_ms * 1e6) [GB/s]
peak_compute_gflops = peak_fp32_tflops * 1000
ridge_point = peak_compute_gflops / memory_bandwidth_gbps
memory_roof_gflops = intensity * memory_bandwidth_gbps
attainable_gflops = min(peak_compute_gflops, memory_roof_gflops)
efficiency_vs_attainable = achieved_gflops / attainable_gflops
efficiency_vs_compute_peak = achieved_gflops / peak_compute_gflops
```

Multiplication and addition count as two floating-point operations, including
when fused into one instruction. This is the existing nominal matmul convention;
extra implementation operations in a reordered variant are not added to it.
All modeled elements are FP32, four bytes each. TFLOP/s and GB/s use decimal
powers (10¹² and 10⁹); `--memory-bandwidth-gbps` means **gigabytes**, not gigabits.

Arithmetic intensity says how many nominal operations are performed per modeled
byte. The model reads A once, reads B once, and writes C once. It does not read C,
count padding, output initialization, host transfers, repeated loads, or hardware
transactions. It is a compulsory/logical-traffic model, **not measured DRAM
traffic**. There are no hardware counters in this analysis. All variants at the
same shape have the same modeled intensity, despite different actual data reuse.

Below the ridge point, the simple model labels a point `memory-bound`. At or above
it, it labels it `compute-bound` (the two roofs coincide exactly at the ridge).
These labels are model predictions, not profiler diagnoses. Cache/shared-memory
behavior, scheduling, instruction dependencies, launch overhead, and additional
traffic can cause large gaps from the model. Vendor peaks may not be sustained.

`compute_peak_percent` and `model_bandwidth_peak_percent` express ratios as
percentages. The latter is **model bytes/time divided by configured bandwidth**,
not measured memory-controller utilization. Efficiencies are stored as fractions.
If a result exceeds any roof, its value is retained and a warning is emitted.
Check assumptions, units, timing noise and model mismatch; values are not clamped.

## Timing and correctness

Only CUDA `kernel_only` rows are analyzed. CPU and `end_to_end` crossover rows are
excluded with reasons in analysis metadata. Legacy benchmark schema 1–3 runs
without per-row timing modes use the repository's original CUDA kernel-only
benchmark contract; this inference is recorded and warned about. Missing modes
in crossover artifacts are not inferred.

Median latency is the default. `--statistic mean` explicitly selects mean latency
for all rows, including older captures without medians. Missing/nonpositive selected
latencies are excluded; the tool does not silently mix statistics. Recomputed
`achieved_gflops` may differ from the original mean-based `gflops` column, which
remains available. The source benchmark's timing procedure is recorded in the report.

Numerical tolerance failures are retained, flagged and plotted; performance does
not establish correctness. A completed `tolerance_failed` source run is analyzable.
Untimed, skipped, failed, unknown-dtype or unsupported-schema runs are rejected.
Output fingerprints, numerical diagnostics, kernel modes, tile size, and source
stability warnings remain in CSV columns, with links to the source artifacts.

## Derived artifacts and provenance

This is a separate **roofline analysis schema version 1**, not a change to the
benchmark's metadata/summary schema:

- `roofline.csv`: original source summary columns plus the formulas above,
  timing-statistic selection, any legacy timing inference, and ceiling warnings.
- `roofline_metadata.json`: explicit ceilings/units/source labels, model/version,
  excluded rows, warnings, source metadata snapshot, source-artifact SHA-256 hashes,
  tool hashes, Python/matplotlib versions, source path, and analysis timestamp.
  `measured_memory_traffic` is null because none was measured.
- `roofline.png` and `report.md` with `--report`: log/log roofs, measured points,
  ridge point, up to ten shape annotations, interpretation caveats and source links.

Metrics are deterministic for the same input artifacts, selected statistic and
ceilings; timestamps and local provenance paths can differ. Reports link back to
the original run, so retain or move both directories together. The analysis does
not copy full output matrices or make cross-hardware performance claims.
