# PTX compiler analysis

`scripts/ptx_inspector.py` performs conservative, CPU-only inspection and
comparison of NVIDIA PTX. It requires neither CUDA tools nor an NVIDIA GPU at
runtime and uses only the Python standard library.

## Inspect PTX

```bash
python3 scripts/ptx_inspector.py inspect \
  --input artifacts/cuda/baseline.ptx \
  --output results/ptx-baseline
```

Inspection lists all entry points and device functions by default. Restrict the
detailed report with `--kernel NAME`. Outputs are `analysis.json`, `report.md`,
and `ptx/input.normalized.ptx`.

The report records module metadata, instruction categories and full mnemonics,
predicates, labels, PTX register declarations, and identifiable static or dynamic
shared-memory declarations. Unknown mnemonics remain visible in the `other`
category.

## Compare PTX

```bash
python3 scripts/ptx_inspector.py compare \
  --baseline artifacts/cuda/baseline.ptx \
  --candidate artifacts/cuda/optimized.ptx \
  --kernel my_kernel \
  --output results/ptx-comparison
```

Without `--kernel`, comparison requires exactly one common entry point. This
prevents silently comparing unrelated kernels. If intentionally comparing renamed
entries, use both `--baseline-kernel` and `--candidate-kernel`.

Comparison output is:

```text
comparison.json
report.md
ptx/
  baseline.normalized.ptx
  candidate.normalized.ptx
  normalized.diff
```

The JSON schema version is 1. Existing output directories are rejected. Unified
diff display defaults to 400 lines; `--diff-lines` changes that bound while hashes
and difference statistics continue to cover the complete input.

Raw and normalized text hashes/diffs cover the complete PTX modules. Instruction,
register, and shared-memory comparisons cover only the explicitly selected entry
point and exclude called device-function bodies. Both scopes are labeled in JSON
and Markdown output.

## Parsing model

The parser recognizes `.entry` and `.func` bodies, metadata, declarations, labels,
optional predicates, instructions, modifiers, and operands. Statement scanning
tracks parentheses, square brackets, strings, and braces independently. Therefore
multiline instructions, vector operands such as `{%f1, %f2}`, and brace-delimited
`mma` operand groups do not terminate a function body or split an instruction.

Instruction families are classified as global/shared/local/constant/parameter
memory, arithmetic, tensor, synchronization, control flow, conversion,
comparison/predication, memory-other, or other. Full mnemonics and modifiers are
always retained. Tensor recognition includes `mma`, `wmma`, and `wgmma`; barrier,
`mbarrier`, `membar`, and fence families are synchronization operations.

This is intentionally not a complete PTX grammar. Unsupported declarations fail
with a diagnostic when interpreting them would require guessing. The MVP does not
construct an interprocedural call graph, so per-entry counts exclude called device
function bodies.

## Normalization

Raw bytes, size, SHA-256, and textual differences are reported separately from
normalized text. Normalization only:

- converts line endings to LF;
- removes trailing whitespace;
- omits `.file` and `.loc` source annotations.

Comments, compiler annotations, symbols, predicates, constants, labels, operands,
memory expressions, instruction order, and other directives remain intact. A
normalized match is not a proof of semantic equivalence.

## Resource interpretation

Register counts are PTX virtual-register declarations. Shared-memory figures are
PTX declarations whose static extent can be identified. They are **not** final
physical register allocation, final hardware resource usage, or occupancy data.
Those require later compilation and hardware-specific tools. The inspector does
not estimate occupancy or predict performance from static counts.

PTX is also not final SASS machine code, and one PTX instruction does not
necessarily map to one SASS instruction. This tool does not analyze scheduling,
runtime memory traffic, profiler timelines, or performance.

## Tests

The tests use synthetic PTX and require no CUDA installation:

```bash
python3 tests/ptx_inspector_test.py
ctest --test-dir build -R ptx-inspector-tests --output-on-failure
```

When `nvcc` is available, generating temporary PTX is useful additional validation,
but generated PTX and CUBIN files remain excluded from Git.
