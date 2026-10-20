# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-5/schedule-n256-eager-s2/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-5/schedule-n256-eager-s2/B.ptx`
- Kernels: `_variant_kernel` → `_variant_kernel`

## Compatibility

| Field | Baseline | Candidate | Match |
|---|---|---|---|
| PTX version | 8.7 | 8.7 | True |
| Target | sm_89 | sm_89 | True |
| Address size | 64 | 64 | True |
| Kernel interface | — | — | True |

## Full-module text comparison

These raw and normalized results compare the complete PTX modules, not only the selected entry point.

| Layer | Result | Changed lines |
|---|---|---:|
| Raw | different | 4981 |
| Normalized | different | 4576 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 1258 | 1258 | +0 |
| comparison_predication | 146 | 146 | +0 |
| control_flow | 8 | 8 | +0 |
| conversion | 626 | 626 | +0 |
| memory_global | 4 | 4 | +0 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 20 | 20 | +0 |
| other | 158 | 158 | +0 |
| synchronization | 28 | 28 | +0 |
| tensor | 168 | 168 | +0 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.f32` | 312 | 312 | +0 |
| `add.s32` | 25 | 25 | +0 |
| `add.s64` | 56 | 56 | +0 |
| `and.b32` | 15 | 15 | +0 |
| `and.pred` | 1 | 1 | +0 |
| `bar.sync` | 28 | 28 | +0 |
| `bfe.s32` | 3 | 3 | +0 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 7 | 7 | +0 |
| `cp.async.ca.shared.global` | 48 | 48 | +0 |
| `cp.async.commit_group` | 48 | 48 | +0 |
| `cp.async.wait_group` | 14 | 14 | +0 |
| `cvt.f32.f16` | 384 | 384 | +0 |
| `cvt.rn.f16x2.f32` | 16 | 16 | +0 |
| `cvt.rna.tf32.f32` | 224 | 224 | +0 |
| `cvt.s64.s32` | 2 | 2 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.b32` | 4 | 4 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 24 | 24 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 24 | 24 | +0 |
| `mad.wide.s32` | 2 | 2 | +0 |
| `mad.wide.u32` | 2 | 2 | +0 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 168 | 168 | +0 |
| `mov.b32` | 311 | 311 | +0 |
| `mov.b64` | 23 | 23 | +0 |
| `mov.u32` | 3 | 3 | +0 |
| `mul.wide.s32` | 2 | 2 | +0 |
| `mul.wide.u32` | 4 | 4 | +0 |
| `or.b32` | 8 | 8 | +0 |
| `or.b64` | 2 | 2 | +0 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 24 | 24 | +0 |
| `selp.f32` | 56 | 56 | +0 |
| `setp.eq.b32` | 1 | 1 | +0 |
| `setp.lt.s32` | 1 | 1 | +0 |
| `setp.lt.u32` | 1 | 1 | +0 |
| `setp.lt.u64` | 7 | 7 | +0 |
| `setp.num.f32` | 56 | 56 | +0 |
| `shl.b32` | 14 | 14 | +0 |
| `shl.b64` | 2 | 2 | +0 |
| `st.global.b32` | 4 | 4 | +0 |
| `st.shared.b32` | 16 | 16 | +0 |
| `sub.f32` | 464 | 464 | +0 |
| `xor.b32` | 7 | 7 | +0 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b16 | 385 | 385 | +0 |
| b32 | 1817 | 1817 | +0 |
| b64 | 75 | 75 | +0 |
| pred | 68 | 68 | +0 |

- PTX-declared static shared bytes: 0 → 0
- Dynamic shared declaration: False → False

### Baseline shared-memory declarations

None in the selected entry.

### Candidate shared-memory declarations

None in the selected entry.

## Conclusions

- The input PTX files are different after the documented conservative normalization.
- Instruction and declaration differences are static PTX observations; they do not establish SASS changes or runtime performance effects.

## Limitations

Normalization removes only `.file`/`.loc` source annotations and whitespace differences described in the JSON report. Comments, predicates, constants, operands, labels, symbols, and instruction order are retained. Per-entry statistics exclude called device-function bodies. PTX instructions do not map one-to-one to SASS instructions.
