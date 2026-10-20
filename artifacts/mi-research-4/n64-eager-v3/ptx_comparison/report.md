# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4/n64-eager-v3/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4/n64-eager-v3/B.ptx`
- Kernels: `fc1_kernel` → `_strassen_kernel`

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
| Raw | different | 3037 |
| Normalized | different | 2833 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 94 | 870 | +776 |
| comparison_predication | 2 | 114 | +112 |
| control_flow | 1 | 1 | +0 |
| conversion | 4 | 368 | +364 |
| memory_global | 1 | 12 | +11 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 8 | 28 | +20 |
| other | 23 | 16 | -7 |
| synchronization | 6 | 23 | +17 |
| tensor | 8 | 168 | +160 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.f32` | 0 | 344 | +344 |
| `add.s32` | 13 | 12 | -1 |
| `add.s64` | 8 | 15 | +7 |
| `and.b32` | 17 | 17 | +0 |
| `and.pred` | 0 | 1 | +1 |
| `bar.sync` | 6 | 23 | +17 |
| `bfe.s32` | 1 | 3 | +2 |
| `bfe.u32` | 1 | 1 | +0 |
| `cp.async.cg.shared.global` | 6 | 0 | -6 |
| `cp.async.commit_group` | 6 | 0 | -6 |
| `cp.async.wait_group` | 3 | 0 | -3 |
| `cvt.f32.f16` | 0 | 128 | +128 |
| `cvt.rn.f16x2.f32` | 4 | 16 | +12 |
| `cvt.rna.tf32.f32` | 0 | 224 | +224 |
| `div.s32` | 1 | 0 | -1 |
| `ld.global.v2.b32` | 0 | 8 | +8 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.b32` | 4 | 4 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 4 | 8 | +4 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 8 | +4 |
| `mad.wide.u32` | 2 | 2 | +0 |
| `min.s32` | 1 | 0 | -1 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 8 | 0 | -8 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 0 | 168 | +168 |
| `mov.b32` | 11 | 122 | +111 |
| `mov.u32` | 2 | 19 | +17 |
| `mul.lo.s32` | 1 | 0 | -1 |
| `mul.wide.s32` | 2 | 2 | +0 |
| `mul.wide.u32` | 0 | 2 | +2 |
| `neg.s32` | 1 | 0 | -1 |
| `or.b32` | 9 | 7 | -2 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 1 | 0 | -1 |
| `selp.f32` | 0 | 56 | +56 |
| `setp.lt.s32` | 1 | 1 | +0 |
| `setp.lt.u32` | 0 | 1 | +1 |
| `setp.num.f32` | 0 | 56 | +56 |
| `shl.b32` | 10 | 12 | +2 |
| `shr.u32` | 2 | 0 | -2 |
| `st.global.b32` | 0 | 4 | +4 |
| `st.global.v4.b32` | 1 | 0 | -1 |
| `st.shared.b32` | 4 | 16 | +12 |
| `st.shared.v2.b32` | 0 | 8 | +8 |
| `sub.f32` | 0 | 304 | +304 |
| `sub.s32` | 2 | 0 | -2 |
| `xor.b32` | 10 | 7 | -3 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b16 | 0 | 129 | +129 |
| b32 | 124 | 1277 | +1153 |
| b64 | 16 | 25 | +9 |
| pred | 2 | 60 | +58 |

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
