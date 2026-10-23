# MI-Research-10 transcript recovery

This checkout reconstructs the source and documentation from the 17 decoded
Codex patches. It does not contain the original MI-Research-10 GPU validation,
paired timing, or final analysis artifacts. No GPU work was repeated.
The small `environment.json` came from an original patch and records the
historical device; it is not a new device measurement.

Each decoded patch matches its saved Codex tool call after removing only the
original deleted-checkout path prefix. Sequentially replaying all 17 patches
produced 11 files identical to the recovered patch targets before the two
historical documents received the recovery notices above. The
[patch audit](../recovery-evidence/mi-research-10/patch-audit.json) records the
session lines and patch hashes. The recovery notices and this page are the only
edits beyond the decoded patches.

## Frozen-plan provenance

| Item | SHA-256 |
| --- | --- |
| Original frozen plan, recorded in the Oct 21 Codex session | `89952d7ecf4a08764f768cadfd827940d64352f0921dd5b7c3b0db6e1490b55d` |
| Regenerated frozen plan in this checkout | `0063d03a5b7a5f9727a968a709a164072d060b62231c67b73c6d5fc37d4482cd` |
| Regenerated four-CTA Perminov-58 kernel, matching the recorded source hash | `fa0715e83b773816966ec70ea6f536d16d0f9ca72be0753ed16875bb504d0cd0` |

The original session records the original plan hash, both complete candidate
descriptor vectors and source hashes, and all five upstream evidence hashes. It
does not contain the original plan bytes or the exact `frozen_at_utc` value. A
search of the available recovery inputs and local checkouts found no second
copy of the original plan. Two independent runs of the recovered freeze
procedure produced plans that differ only in `frozen_at_utc`. After excluding
that field, their content matches exactly. Both complete descriptor vectors,
both source hashes, and all five upstream hashes match the original recorded
output. The candidate selection, rank-23 tensor identity, four-CTA workload,
and 4,096-byte positive candidate-minus-control live-byte delta also check.

This proves internal regeneration determinism for the checked content. It does
not prove that the lost original plan differed only in its timestamp. The
original SHA-256 remains an expected historical hash, not the hash of the
regenerated artifact. The original plan and its timestamp were not recreated
or edited to force a match. The original prose claim of 4,096 **fewer** bytes
remains in the regenerated plan, while its descriptors say 4,096 **more**.

The saved [comparison record](../recovery-evidence/mi-research-10/plan-comparison.json)
identifies the original session output lines. The adjacent files retain the
original freeze and descriptor output text and the local command logs.

## Validation boundary

The original `analysis.json` was generated from the original frozen plan,
retained MI-Research-7B compiler control, new GPU validation `raw.json`, and
new paired timing `raw.json`. It also embeds the original plan SHA-256. The
retained control is available, but the new GPU validation, paired timing,
analysis, and final hash manifest are missing. The saved session prints selected
summary fields and a file list, not the full raw records or the analysis file.
Those summaries cannot establish every analysis field or recreate the original
hash chain. Exact reconstruction of `analysis.json` from this session is
unverifiable. No substitute analysis or GPU measurements were created.

The current three-file MI-Research-9/10 CPU selection has 28 passes and one
skip. `test_evidence_and_frozen_prediction_audit` skips only when all four
historical outputs are absent: `analysis.json`, `hash_manifest.json`,
`gpu-validation-v1/`, and `graph-pair-s1/`. The recovered frozen plan, kernel
source, and small `environment.json` do not count as GPU evidence. A partial
bundle, corrupt analysis, or hash mismatch enters the original strict replay
and fails. CPU-only regression tests cover the missing, present, partial,
corrupt, and mismatched cases. The historical evidence test is **skipped, not
passed**, in this checkout.

The standalone `scripts/analyze_mi_research_10.py --verify` still fails at the
missing `artifacts/mi-research-10/analysis.json`. The GPU validation and
timing directories are also absent. Historical compiler, correctness, timing,
and final manifest claims remain unverified here. Restoring only the lost GPU
files may still fail their frozen-plan identity checks: the original plan hash
is `89952d7ecf4a08764f768cadfd827940d64352f0921dd5b7c3b0db6e1490b55d`,
while this checkout's regenerated plan hashes to
`0063d03a5b7a5f9727a968a709a164072d060b62231c67b73c6d5fc37d4482cd`.
The verification rules and original numerical tolerances were not changed.

The preregistration checker rejects the historical plan because it lacks
structured `descriptor_claims`. Its targeted CPU test also rejects a new
structured claim stating 4,096 fewer bytes when the descriptors say 4,096
more. The generator, recovered-plan replay, and MI-Research-9 replay pass.
See [the saved command results](../recovery-evidence/mi-research-10/commands.json)
for exact invocations, exit codes, and raw output files.
