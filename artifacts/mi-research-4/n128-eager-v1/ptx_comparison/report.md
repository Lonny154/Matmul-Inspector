# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4/n128-eager-v1/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4/n128-eager-v1/B.ptx`
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
| Raw | different | 4041 |
| Normalized | different | 3706 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 98 | 1228 | +1130 |
| comparison_predication | 2 | 139 | +137 |
| control_flow | 1 | 8 | +7 |
| conversion | 4 | 629 | +625 |
| memory_global | 1 | 4 | +3 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 8 | 20 | +12 |
| other | 41 | 158 | +117 |
| synchronization | 10 | 28 | +18 |
| tensor | 16 | 168 | +152 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.f32` | 0 | 424 | +424 |
| `add.s32` | 13 | 24 | +11 |
| `add.s64` | 12 | 52 | +40 |
| `and.b32` | 17 | 15 | -2 |
| `and.pred` | 0 | 1 | +1 |
| `bar.sync` | 10 | 28 | +18 |
| `bfe.s32` | 1 | 3 | +2 |
| `bfe.u32` | 1 | 2 | +1 |
| `bra` | 0 | 7 | +7 |
| `cp.async.ca.shared.global` | 0 | 48 | +48 |
| `cp.async.cg.shared.global` | 10 | 0 | -10 |
| `cp.async.commit_group` | 10 | 48 | +38 |
| `cp.async.wait_group` | 5 | 14 | +9 |
| `cvt.f32.f16` | 0 | 384 | +384 |
| `cvt.rn.f16x2.f32` | 4 | 16 | +12 |
| `cvt.rna.tf32.f32` | 0 | 224 | +224 |
| `cvt.s64.s32` | 0 | 2 | +2 |
| `cvt.u64.u32` | 0 | 3 | +3 |
| `div.s32` | 1 | 0 | -1 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.b32` | 4 | 4 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 24 | +16 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 8 | 24 | +16 |
| `mad.wide.s32` | 0 | 2 | +2 |
| `mad.wide.u32` | 2 | 2 | +0 |
| `min.s32` | 1 | 0 | -1 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 16 | 0 | -16 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 0 | 168 | +168 |
| `mov.b32` | 11 | 307 | +296 |
| `mov.b64` | 0 | 14 | +14 |
| `mov.pred` | 0 | 21 | +21 |
| `mov.u32` | 2 | 3 | +1 |
| `mul.lo.s32` | 1 | 0 | -1 |
| `mul.wide.s32` | 2 | 0 | -2 |
| `mul.wide.u32` | 0 | 2 | +2 |
| `neg.s32` | 1 | 0 | -1 |
| `or.b32` | 9 | 7 | -2 |
| `or.b64` | 0 | 9 | +9 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 1 | 24 | +23 |
| `selp.f32` | 0 | 56 | +56 |
| `setp.eq.b32` | 0 | 1 | +1 |
| `setp.lt.s32` | 1 | 1 | +0 |
| `setp.lt.u32` | 0 | 1 | +1 |
| `setp.num.f32` | 0 | 56 | +56 |
| `shl.b32` | 10 | 12 | +2 |
| `shl.b64` | 0 | 17 | +17 |
| `shr.u32` | 2 | 0 | -2 |
| `st.global.b32` | 0 | 4 | +4 |
| `st.global.v4.b32` | 1 | 0 | -1 |
| `st.shared.b32` | 4 | 16 | +12 |
| `sub.f32` | 0 | 304 | +304 |
| `sub.s32` | 2 | 0 | -2 |
| `xor.b32` | 10 | 7 | -3 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b16 | 0 | 385 | +385 |
| b32 | 156 | 1761 | +1605 |
| b64 | 20 | 100 | +80 |
| pred | 2 | 75 | +73 |

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
