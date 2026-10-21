# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-dogfood-2/k64-v1/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-dogfood-2/k64-v1/B.ptx`
- Kernels: `fc1_kernel` → `fc1_kernel`

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
| Raw | different | 1448 |
| Normalized | different | 1343 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 255 | 316 | +61 |
| comparison_predication | 13 | 13 | +0 |
| control_flow | 2 | 2 | +0 |
| conversion | 34 | 34 | +0 |
| memory_global | 8 | 8 | +0 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 16 | 16 | +0 |
| other | 38 | 68 | +30 |
| synchronization | 7 | 7 | +0 |
| tensor | 32 | 64 | +32 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.s32` | 39 | 64 | +25 |
| `add.s64` | 33 | 48 | +15 |
| `and.b32` | 13 | 13 | +0 |
| `bar.sync` | 7 | 7 | +0 |
| `bfe.s32` | 1 | 1 | +0 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 1 | 1 | +0 |
| `cp.async.cg.shared.global` | 18 | 36 | +18 |
| `cp.async.commit_group` | 6 | 6 | +0 |
| `cp.async.wait_group` | 2 | 2 | +0 |
| `cvt.rn.f16x2.f32` | 32 | 32 | +0 |
| `cvt.s64.s32` | 1 | 1 | +0 |
| `cvt.u32.u64` | 1 | 1 | +0 |
| `div.s32` | 1 | 1 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.v4.b32` | 8 | 8 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 16 | +8 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 8 | +4 |
| `mad.lo.s32` | 1 | 0 | -1 |
| `mad.wide.s32` | 10 | 12 | +2 |
| `mad.wide.u32` | 2 | 7 | +5 |
| `min.s32` | 1 | 1 | +0 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 32 | 64 | +32 |
| `mov.b32` | 69 | 69 | +0 |
| `mov.b64` | 1 | 1 | +0 |
| `mov.u32` | 2 | 2 | +0 |
| `mul.hi.u32` | 1 | 1 | +0 |
| `mul.lo.s32` | 14 | 17 | +3 |
| `mul.wide.s32` | 3 | 5 | +2 |
| `mul.wide.u32` | 2 | 1 | -1 |
| `or.b32` | 20 | 28 | +8 |
| `or.b64` | 2 | 4 | +2 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 6 | 6 | +0 |
| `setp.eq.b32` | 1 | 1 | +0 |
| `setp.gt.s32` | 2 | 2 | +0 |
| `setp.lt.s32` | 2 | 2 | +0 |
| `setp.lt.u32` | 1 | 1 | +0 |
| `setp.ne.b32` | 1 | 1 | +0 |
| `shl.b32` | 18 | 17 | -1 |
| `shl.b64` | 1 | 1 | +0 |
| `shr.u32` | 3 | 3 | +0 |
| `st.global.v4.b32` | 8 | 8 | +0 |
| `st.shared.v4.b32` | 8 | 8 | +0 |
| `sub.s32` | 3 | 3 | +0 |
| `xor.b32` | 13 | 15 | +2 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b32 | 318 | 402 | +84 |
| b64 | 56 | 82 | +26 |
| pred | 8 | 8 | +0 |

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
