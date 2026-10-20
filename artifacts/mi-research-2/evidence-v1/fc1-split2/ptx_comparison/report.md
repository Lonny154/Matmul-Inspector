# PTX comparison

- Baseline: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-2/evidence-v1/fc1-split2/A.ptx`
- Candidate: `/home/daniel/projects/matmul-inspector/artifacts/mi-research-2/evidence-v1/fc1-split2/B.ptx`
- Kernels: `fc1_kernel` → `reduce_partials`

## Compatibility

| Field | Baseline | Candidate | Match |
|---|---|---|---|
| PTX version | 8.7 | 8.7 | True |
| Target | sm_89 | sm_89 | True |
| Address size | 64 | 64 | True |
| Kernel interface | — | — | False |

## Full-module text comparison

These raw and normalized results compare the complete PTX modules, not only the selected entry point.

| Layer | Result | Changed lines |
|---|---|---:|
| Raw | different | 830 |
| Normalized | different | 754 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 255 | 36 | -219 |
| comparison_predication | 13 | 1 | -12 |
| control_flow | 2 | 1 | -1 |
| conversion | 34 | 4 | -30 |
| memory_global | 8 | 5 | -3 |
| memory_param | 3 | 2 | -1 |
| memory_shared | 16 | 0 | -16 |
| other | 38 | 0 | -38 |
| synchronization | 7 | 0 | -7 |
| tensor | 32 | 0 | -32 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.f32` | 0 | 8 | +8 |
| `add.s32` | 39 | 0 | -39 |
| `add.s64` | 33 | 3 | -30 |
| `and.b32` | 13 | 1 | -12 |
| `bar.sync` | 7 | 0 | -7 |
| `bfe.s32` | 1 | 0 | -1 |
| `bfe.u32` | 2 | 0 | -2 |
| `bra` | 1 | 0 | -1 |
| `cp.async.cg.shared.global` | 18 | 0 | -18 |
| `cp.async.commit_group` | 6 | 0 | -6 |
| `cp.async.wait_group` | 2 | 0 | -2 |
| `cvt.rn.f16x2.f32` | 32 | 4 | -28 |
| `cvt.s64.s32` | 1 | 0 | -1 |
| `cvt.u32.u64` | 1 | 0 | -1 |
| `div.s32` | 1 | 0 | -1 |
| `ld.global.v4.b32` | 0 | 4 | +4 |
| `ld.param.b64` | 3 | 2 | -1 |
| `ld.shared.v4.b32` | 8 | 0 | -8 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 0 | -8 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 0 | -4 |
| `mad.lo.s32` | 1 | 0 | -1 |
| `mad.wide.s32` | 10 | 2 | -8 |
| `mad.wide.u32` | 2 | 0 | -2 |
| `min.s32` | 1 | 0 | -1 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 32 | 0 | -32 |
| `mov.b32` | 69 | 1 | -68 |
| `mov.b64` | 1 | 0 | -1 |
| `mov.u32` | 2 | 18 | +16 |
| `mul.hi.u32` | 1 | 0 | -1 |
| `mul.lo.s32` | 14 | 0 | -14 |
| `mul.wide.s32` | 3 | 0 | -3 |
| `mul.wide.u32` | 2 | 0 | -2 |
| `or.b32` | 20 | 1 | -19 |
| `or.b64` | 2 | 0 | -2 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 6 | 0 | -6 |
| `setp.eq.b32` | 1 | 0 | -1 |
| `setp.gt.s32` | 2 | 0 | -2 |
| `setp.lt.s32` | 2 | 1 | -1 |
| `setp.lt.u32` | 1 | 0 | -1 |
| `setp.ne.b32` | 1 | 0 | -1 |
| `shl.b32` | 18 | 2 | -16 |
| `shl.b64` | 1 | 0 | -1 |
| `shr.u32` | 3 | 0 | -3 |
| `st.global.v4.b32` | 8 | 1 | -7 |
| `st.shared.v4.b32` | 8 | 0 | -8 |
| `sub.s32` | 3 | 0 | -3 |
| `xor.b32` | 13 | 0 | -13 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b32 | 318 | 36 | -282 |
| b64 | 56 | 8 | -48 |
| pred | 8 | 2 | -6 |

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
