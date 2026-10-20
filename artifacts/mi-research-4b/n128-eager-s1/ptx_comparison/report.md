# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4b/n128-eager-s1/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-4b/n128-eager-s1/B.ptx`
- Kernels: `_rank8_kernel` → `_strassen_kernel`

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
| Raw | different | 6797 |
| Normalized | different | 6322 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 1046 | 1228 | +182 |
| comparison_predication | 158 | 139 | -19 |
| control_flow | 9 | 8 | -1 |
| conversion | 534 | 629 | +95 |
| memory_global | 4 | 4 | +0 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 20 | 20 | +0 |
| other | 112 | 158 | +46 |
| synchronization | 31 | 28 | -3 |
| tensor | 192 | 168 | -24 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.f32` | 352 | 424 | +72 |
| `add.s32` | 15 | 24 | +9 |
| `add.s64` | 42 | 52 | +10 |
| `and.b32` | 15 | 15 | +0 |
| `and.pred` | 1 | 1 | +0 |
| `bar.sync` | 31 | 28 | -3 |
| `bfe.s32` | 3 | 3 | +0 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 8 | 7 | -1 |
| `cp.async.ca.shared.global` | 32 | 48 | +16 |
| `cp.async.commit_group` | 32 | 48 | +16 |
| `cp.async.wait_group` | 16 | 14 | -2 |
| `cvt.f32.f16` | 256 | 384 | +128 |
| `cvt.rn.f16x2.f32` | 16 | 16 | +0 |
| `cvt.rna.tf32.f32` | 256 | 224 | -32 |
| `cvt.s64.s32` | 2 | 2 | +0 |
| `cvt.u32.u64` | 1 | 0 | -1 |
| `cvt.u64.u32` | 3 | 3 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.b32` | 4 | 4 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 16 | 24 | +8 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 16 | 24 | +8 |
| `mad.wide.s32` | 2 | 2 | +0 |
| `mad.wide.u32` | 3 | 2 | -1 |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | 192 | 168 | -24 |
| `mov.b32` | 257 | 307 | +50 |
| `mov.b64` | 16 | 14 | -2 |
| `mov.pred` | 24 | 21 | -3 |
| `mov.u32` | 3 | 3 | +0 |
| `mul.wide.u32` | 1 | 2 | +1 |
| `or.b32` | 7 | 7 | +0 |
| `or.b64` | 8 | 9 | +1 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 27 | 24 | -3 |
| `selp.f32` | 64 | 56 | -8 |
| `setp.eq.b32` | 1 | 1 | +0 |
| `setp.lt.s32` | 1 | 1 | +0 |
| `setp.lt.u32` | 1 | 1 | +0 |
| `setp.num.f32` | 64 | 56 | -8 |
| `shl.b32` | 12 | 12 | +0 |
| `shl.b64` | 20 | 17 | -3 |
| `st.global.b32` | 4 | 4 | +0 |
| `st.shared.b32` | 16 | 16 | +0 |
| `sub.f32` | 256 | 304 | +48 |
| `xor.b32` | 7 | 7 | +0 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b16 | 257 | 385 | +128 |
| b32 | 1490 | 1761 | +271 |
| b64 | 93 | 100 | +7 |
| pred | 85 | 75 | -10 |

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
