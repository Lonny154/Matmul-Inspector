# CUDA PTX/SASS FMA case study

This case study keeps four kinds of evidence separate: measured runtime results,
static machine-code observations, numerical tolerance results, and possible
performance explanations. The observations below describe one CUDA 13.3 capture
on an NVIDIA GeForce RTX 4060 Ti (compute capability 8.9); they are not general
claims about contraction or compiler performance.

## Inspect saved nvdisasm text

`scripts/sass_inspector.py` is a Python-standard-library utility for existing
text produced by `nvdisasm`. Inspection is CPU-only and needs neither a GPU nor a
CUDA installation. Producing the `.sass` input is a separate toolchain step.

```bash
python3 scripts/sass_inspector.py inspect \
  --input /tmp/matmul-release.sass \
  --kernel mi_naive_fma \
  --output /tmp/sass-fma-inspection

python3 scripts/sass_inspector.py compare \
  --baseline /tmp/matmul-release.sass \
  --candidate /tmp/matmul-release.sass \
  --baseline-kernel mi_naive_fma \
  --candidate-kernel mi_naive_no_fma \
  --output /tmp/sass-fma-comparison
```

Inspection writes `analysis.json`, `report.md`, and
`sass/input.normalized.sass`. Comparison writes `comparison.json`, `report.md`,
the two normalized selected-function listings, and a bounded unified diff.
Existing output directories are rejected.

## Parsing and comparison model

Executable `.text.<symbol>` sections are identified from section directives.
Within each section, `.type <symbol>,@function` declarations and an exact matching
symbol label define function ownership. This matters because nvdisasm can place a
kernel and one or more callable helper bodies in the same text section. A helper
body is reported separately and excluded from primary-kernel counts. If exact
ownership cannot be established, instructions remain unattributed and a boundary
ambiguity is reported instead of being assigned to the kernel.

Each parsed instruction retains its original hexadecimal address, numeric address,
predicate, full mnemonic, base operation, modifiers, operands, category, and any
recognized call target in structured JSON. Labels and instruction order are also
preserved. Metadata and debug sections are not counted as executable instructions;
unknown mnemonics remain visible as `other`.

The comparison scopes are deliberately distinct:

- raw and presentation-normalized statistics cover the complete input files;
- instruction and resource comparisons cover only the selected functions;
- called helper bodies are not folded into selected-function statistics.

Presentation normalization converts line endings, collapses whitespace, and omits
hexadecimal instruction-address comments. It retains symbols, labels, predicates,
mnemonics, modifiers, operands, constants, branch targets, and instruction order.
Original addresses remain in JSON. A normalized match is not proof of binary or
semantic equivalence.

## Static machine-code evidence

The CUDA 13.3 `sm_89` capture contains nine primary kernels and four embedded
helpers with no unresolved boundaries. For the two selected kernels the inspector
reports:

| Static observation | `mi_naive_fma` | `mi_naive_no_fma` |
|---|---:|---:|
| `FFMA` | 5 | 0 |
| `FMUL` | 0 | 5 |
| `FADD` | 0 | 5 |
| Floating-point arithmetic total | 5 | 10 |
| `SHI_REGISTERS` allocation/thread | 28 | 28 |
| `CALL.REL.NOINC` | 1 | 1 |

Both calls target nvdisasm-named unsigned 64-bit division helpers; their generated
suffixes differ. The calls occur at address `0x0130` in this capture. Helper bodies
are parsed as separate functions and are not included in the table's instruction
counts. `SHI_REGISTERS=28` is compiler-recorded physical 32-bit register allocation
per thread for each primary section function. It is not a PTX virtual-register
count and, by itself, does not determine occupancy or total hardware-resource use.

This is final-machine-code evidence that this capture uses fused instructions in
one selected kernel and separate multiply/add instructions in the other. It does
not establish execution counts, issue rates, dependencies, memory traffic, or the
cause of any timing difference.

## Measured runtime results

The existing benchmark used CUDA events, ten warmups, 100 measured iterations per
trial, and five trials. Allocations, transfers, initialization, warmups, and host
comparison were excluded from measured kernel time. Recorded medians were:

| Square size | FMA median (ms) | No-FMA median (ms) | No-FMA versus FMA |
|---:|---:|---:|---:|
| 256 | 0.034816 | 0.034752 | -0.18% |
| 512 | 0.260848 | 0.246784 | -5.39% |
| 1024 | 2.299328 | 2.364848 | +2.85% |

Reference and candidate were run in fixed reference-before-candidate order, not a
randomized order. The source tree was dirty, and clock, thermal, background-load,
and run-to-run variation remain possible. The 256 result is especially close;
confidence-interval overlap in the recorded report is descriptive, not a formal
significance test. These measurements therefore do not support a general claim
that either instruction sequence is faster.

## Numerical results

At the configured absolute and relative tolerances, the recorded No-FMA candidate
had 0 tolerance failures at size 256, 25 at size 512, and 1,332 at size 1024.
Those failures are numerical observations against the recorded FMA reference, not
performance measurements. Fused and non-fused evaluation can round differently;
this static inspection does not prove that contraction alone accounts for every
reported output difference.

## Carefully qualified interpretation

The static evidence verifies the intended instruction-shape distinction in this
specific capture and shows equal recorded register allocation plus analogous
division-helper calls. The timing direction changes with problem size, while the
candidate also fails the configured numerical tolerance at larger sizes. Possible
explanations include instruction dependency structure, compiler scheduling, and
measurement variability, but none is established by mnemonic counts. Performance
claims would require cleaner, repeated, preferably randomized measurements and
additional hardware evidence such as profiler counters.

## Limitations and tests

The parser consumes nvdisasm presentation text rather than decoding CUBIN bytes.
It does not model instruction encodings, control-flow graphs, scheduling,
dependency latency, dynamic execution counts, memory transactions, occupancy, or
runtime performance. Function ownership depends on explicit section, function-type,
and exact symbol-label structure; unusual or incomplete dumps may be reported as
ambiguous. Instruction classification is broad and preserves unfamiliar mnemonics
without attempting speculative interpretation.

Synthetic CPU-only tests cover addresses, predicates, modifiers, instruction
families, embedded helpers, metadata exclusion, ambiguous boundaries, selection,
normalization scope, structured output, and overwrite protection:

```bash
python3 tests/sass_inspector_test.py
ctest --test-dir build -R sass-inspector-tests --output-on-failure
```

Generated SASS, CUBIN, and report artifacts should remain outside version control.
