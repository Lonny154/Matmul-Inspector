# RTX 4060 Ti experiment-format example

This small result set was captured on 2026-09-22 by the new experiment CLI:

```sh
matmul-inspector benchmark --sizes 4,256,257,1024 --seed 42 --warmups 3 --iterations 50 --output results/examples/rtx4060ti-naive-vs-tiled
```

The complete captured configuration/environment is in [metadata.json](metadata.json).
[summary.csv](summary.csv) contains eight rows (two kernels for each size), and
[console.txt](console.txt) is the actual experiment log. All four comparisons had
zero bitwise divergent elements and tolerance PASS, so there is no mismatches.csv.

This is a **fresh capture**, distinct from the original tiled-milestone timing
table in the repository README. Do not mix its metadata with those older timings.
The data has not been filtered to choose a fast run; timing variability is visible
in the mean/median/stddev fields. No performance claim across hardware is intended.

The run used an RTX 4060 Ti, CUDA 13.3.73, Release compilation, no fast-math flags,
the compiler-default architecture target and CUDA-event timing. Allocation,
transfers, warmups, comparisons and file I/O were excluded from kernel timing.
Both seeds, tolerances, actual device/runtime/driver details and build flags are
recorded in metadata rather than inferred from the directory name.

**Provenance limitation:** `git_dirty` is true. The base commit predates the tiled
and experiment changes, which were still uncommitted when measured. The metadata
does not contain the source patch or executable, so the base commit alone cannot
reconstruct this binary. This example demonstrates honest provenance reporting,
not an immutable numerical/performance reference baseline.

After building the current source, replay the input/configuration to a fresh path:

```sh
python3 scripts/reproduce.py results/examples/rtx4060ti-naive-vs-tiled/metadata.json --executable build-cuda/matmul-inspector --output results/replay
```

Replaying does not recreate the original software/hardware environment or promise
the same timings. See [experiment semantics](../../../docs/experiments.md).
