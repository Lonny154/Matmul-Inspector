# PTX comparison

- Baseline: `artifacts/mi-dogfood-1/baseline-v1/candidate.ptx`
- Candidate: `artifacts/mi-dogfood-1/stages2-v1/candidate.ptx`
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
| Raw | different | 1045 |
| Normalized | different | 1037 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 255 | 232 | -23 |
| comparison_predication | 13 | 9 | -4 |
| control_flow | 2 | 2 | +0 |
| conversion | 34 | 35 | +1 |
| memory_global | 8 | 8 | +0 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 16 | 16 | +0 |
| other | 38 | 30 | -8 |
| synchronization | 7 | 6 | -1 |
| tensor | 32 | 32 | +0 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.s32` | 39 | 29 | -10 |
| `add.s64` | 33 | 27 | -6 |
| `and.b32` | 13 | 13 | +0 |
| `bar.sync` | 7 | 6 | -1 |
| `bfe.s32` | 1 | 1 | +0 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 1 | 1 | +0 |
| `cp.async.cg.shared.global` | 18 | 12 | -6 |
| `cp.async.commit_group` | 6 | 4 | -2 |
| `cp.async.wait_group` | 2 | 2 | +0 |
| `cvt.rn.f16x2.f32` | 32 | 32 | +0 |
| `cvt.s64.s32` | 1 | 1 | +0 |
| `cvt.u32.u64` | 1 | 2 | +1 |
| `div.s32` | 1 | 1 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.v4.b32` | 8 | 8 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 8 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 4 | +0 |
| `mad.lo.s32` | 1 | 1 | +0 |
| `mad.wide.s32` | 10 | 10 | +0 |
| `mad.wide.u32` | 2 | 2 | +0 |
| `min.s32` | 1 | 1 | +0 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 32 | 32 | +0 |
| `mov.b32` | 69 | 66 | -3 |
| `mov.b64` | 1 | 1 | +0 |
| `mov.u32` | 2 | 2 | +0 |
| `mul.hi.u32` | 1 | 1 | +0 |
| `mul.lo.s32` | 14 | 14 | +0 |
| `mul.wide.s32` | 3 | 3 | +0 |
| `mul.wide.u32` | 2 | 2 | +0 |
| `or.b32` | 20 | 20 | +0 |
| `or.b64` | 2 | 2 | +0 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 6 | 4 | -2 |
| `setp.eq.b32` | 1 | 1 | +0 |
| `setp.gt.s32` | 2 | 0 | -2 |
| `setp.lt.s32` | 2 | 2 | +0 |
| `setp.lt.u32` | 1 | 0 | -1 |
| `setp.ne.b32` | 1 | 2 | +1 |
| `shl.b32` | 18 | 14 | -4 |
| `shl.b64` | 1 | 1 | +0 |
| `shr.u32` | 3 | 3 | +0 |
| `st.global.v4.b32` | 8 | 8 | +0 |
| `st.shared.v4.b32` | 8 | 8 | +0 |
| `sub.s32` | 3 | 3 | +0 |
| `xor.b32` | 13 | 13 | +0 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b32 | 318 | 303 | -15 |
| b64 | 56 | 50 | -6 |
| pred | 8 | 6 | -2 |

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
