# NKI compiler-artifact comparison

`scripts/nki_compare_artifacts.py` compares two saved NKI compiler captures on a
CPU-only machine. It does not import NKI, invoke Neuron tools, execute a kernel, or
modify the source captures.

## Usage

```bash
python3 scripts/nki_compare_artifacts.py \
  artifacts/nki-bf16_1024_n512 \
  artifacts/nki-bf16_1024_kblocked \
  --output results/nki-k-blocking-compiler-comparison
```

An explicit A-tile-reuse comparison uses the same interface:

```bash
python3 scripts/nki_compare_artifacts.py \
  artifacts/nki-trn1-512 \
  artifacts/nki-trn1-512-v2 \
  --output results/nki-a-reuse-compiler-comparison
```

The archived experiments need not be present to install or test the tool. Its
CPU-only tests construct small captures representing both experiment outcomes.
If a capture contains more than one `sg*/instruction_stats.txt` or
`sg*/dma_stats.txt`, select one with `--subgraph sg00`.

Optional `neuron-bench` summaries can be attached without combining their timing
windows:

```bash
python3 scripts/nki_compare_artifacts.py BASELINE CANDIDATE \
  --baseline-benchmark BASELINE_BENCHMARK \
  --candidate-benchmark CANDIDATE_BENCHMARK \
  --output OUTPUT
```

Each benchmark directory must contain exactly one benchmark `info.json`. NeuronCore
latency and runtime latency remain separate fields. These summaries do not establish
statistical significance; use controlled, repeated, interleaved measurements for a
performance claim.

## Inputs and comparison layers

The tool discovers these files independently, so a missing optional artifact does
not suppress comparisons that remain possible:

- `module.mlir`: raw text and canonicalized text.
- `sg*/instruction_stats.txt`: aggregate hardware opcode counts.
- `sg*/dma_stats.txt`: parsed compiler DMA tables plus a normalized-text hash.
- `kernel.neff`: file size and SHA-256 only.
- `info.json`, `kernel_info.json`, and `neff.json`: available compiler, target,
  kernel, shape, and dtype provenance.

MLIR canonicalization removes balanced nested `loc(...)` annotations and replaces
only the captured entry-point symbol with a stable name. It preserves operations,
types, constants, memory spaces, attributes, SSA names, and ordering. Consequently,
canonical equality means equality under those two documented transformations, not
semantic equivalence of arbitrary MLIR programs.

Instruction-count equality does not establish equal instruction order or schedules.
DMA-summary equality does not establish equal timing or critical paths. A matching
NEFF hash establishes byte identity of the files; a different hash does not by
itself identify a machine-code or scheduling difference.

## Outputs

The output directory must not already exist. It contains:

```text
comparison.json
report.md
mlir/
  baseline.normalized.mlir
  candidate.normalized.mlir
  normalized.diff
```

`comparison.json` has schema version 1 and is the machine-readable result.
`report.md` is a concise evidence summary. The unified diff is bounded to 400 lines
by default; use `--diff-lines` to change the display limit. Hashes and full diff
statistics are calculated before truncation.

## Validation

Run the focused test directly:

```bash
python3 tests/nki_compare_artifacts_test.py
```

It is also registered with CTest under the `cpu` label:

```bash
ctest --test-dir build -R nki-compiler-artifact-tests --output-on-failure
```
