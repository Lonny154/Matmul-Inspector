# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-dogfood-2/n256-v1/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-dogfood-2/n256-v1/B.ptx`
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
| Raw | different | 1589 |
| Normalized | different | 1485 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 255 | 385 | +130 |
| comparison_predication | 13 | 11 | -2 |
| control_flow | 2 | 2 | +0 |
| conversion | 34 | 66 | +32 |
| memory_global | 8 | 16 | +8 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 16 | 16 | +0 |
| other | 38 | 70 | +32 |
| synchronization | 7 | 19 | +12 |
| tensor | 32 | 64 | +32 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.s32` | 39 | 54 | +15 |
| `add.s64` | 33 | 53 | +20 |
| `and.b32` | 13 | 12 | -1 |
| `bar.sync` | 7 | 19 | +12 |
| `bfe.s32` | 1 | 2 | +1 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 1 | 1 | +0 |
| `cp.async.cg.shared.global` | 18 | 30 | +12 |
| `cp.async.commit_group` | 6 | 6 | +0 |
| `cp.async.wait_group` | 2 | 2 | +0 |
| `cvt.rn.f16x2.f32` | 32 | 64 | +32 |
| `cvt.s64.s32` | 1 | 1 | +0 |
| `cvt.u32.u64` | 1 | 1 | +0 |
| `div.s32` | 1 | 1 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.v4.b32` | 8 | 0 | -8 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 8 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 24 | +20 |
| `mad.lo.s32` | 1 | 1 | +0 |
| `mad.wide.s32` | 10 | 18 | +8 |
| `mad.wide.u32` | 2 | 6 | +4 |
| `min.s32` | 1 | 1 | +0 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 32 | 64 | +32 |
| `mov.b32` | 69 | 133 | +64 |
| `mov.b64` | 1 | 1 | +0 |
| `mov.u32` | 2 | 2 | +0 |
| `mul.hi.u32` | 1 | 1 | +0 |
| `mul.lo.s32` | 14 | 22 | +8 |
| `mul.wide.s32` | 3 | 3 | +0 |
| `mul.wide.u32` | 2 | 2 | +0 |
| `or.b32` | 20 | 34 | +14 |
| `or.b64` | 2 | 2 | +0 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 6 | 5 | -1 |
| `setp.eq.b32` | 1 | 0 | -1 |
| `setp.gt.s32` | 2 | 2 | +0 |
| `setp.lt.s32` | 2 | 2 | +0 |
| `setp.lt.u32` | 1 | 1 | +0 |
| `setp.ne.b32` | 1 | 1 | +0 |
| `shl.b32` | 18 | 19 | +1 |
| `shl.b64` | 1 | 1 | +0 |
| `shr.u32` | 3 | 2 | -1 |
| `st.global.v4.b32` | 8 | 16 | +8 |
| `st.shared.v4.b32` | 8 | 16 | +8 |
| `sub.s32` | 3 | 3 | +0 |
| `xor.b32` | 13 | 10 | -3 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b32 | 318 | 495 | +177 |
| b64 | 56 | 88 | +32 |
| pred | 8 | 7 | -1 |

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
