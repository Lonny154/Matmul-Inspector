# RTX 4060 Ti controlled floating-point examples

Real Release captures from 2026-09-22; NVCC 13.3.73, GCC 13.3.0, RTX 4060 Ti,
compute capability 8.9, build target compute_75/sm_75. These are hardware/compiler
specific examples, not universal numerical or performance baselines. Every run
records `git_dirty: true`; the commit alone cannot reconstruct this source.

Each subdirectory contains metadata, summary, console output and, when saved,
mismatch samples. All sizes are 4, 256, 257 and 1024, seed 42 (B seed 123),
atol 1e-6, rtol 1e-5. At most five mismatches per size are saved; truncation is
explicit. Benchmark runs use three warmups and 50 CUDA-event measurements.

- `fma-vs-no-fma`: contraction-only comparison against explicit FMA.
- `fma-vs-reordered`: even/odd accumulation with explicit FMA in both kernels.
- `default-vs-fma`: verifies the ordinary naive output matched the explicit reference.
- `cancellation`, `fma-sensitive`: versioned deterministic fixtures.
- `benchmark-*`: each variant timed against an independently measured ordinary naive baseline.
- `fp_verification.json`: actual library PTX inspection report and SHA-256.

Runs with `status: tolerance_failed` intentionally preserve observed failures.
Replay any metadata file using the built binary, for example:

```sh
python3 scripts/reproduce.py results/examples/rtx4060ti-controlled-fp/fma-vs-reordered/metadata.json --executable build-cuda/matmul-inspector --output results/reordered-replay
```

Exit 1 is expected if tolerance still fails. Compare summaries/mismatches for
numerical replay; timestamps and kernel timings are not expected to be identical.
See [the full analysis](../../../docs/floating_point_experiments.md).
