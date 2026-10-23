# MI-Research-10 preregistration consistency gate

> **Recovery note:** The linked local frozen plan was regenerated and has a
> different timestamp and SHA-256 from the original plan described below.
> The original plan bytes are unavailable. See the
> [recovery checkpoint](mi_research_10_recovery.md) for the hashes and replay
> limits.

## Historical finding

The original MI-Research-10 [frozen plan](../artifacts/mi-research-10/frozen_plan.json) is evidence. Its SHA-256 is `89952d7ecf4a08764f768cadfd827940d64352f0921dd5b7c3b0db6e1490b55d`. It records 59,392 modeled live bytes per CTA for the four-CTA 55 circuit and 63,488 for the four-CTA 58 circuit. One prose hypothesis says the 58 circuit has 4,096 **fewer** bytes. The descriptor says 4,096 **more**. [The MI-Research-10 report](mi_research_10.md) classifies that prospective liveness claim as internally inconsistent. This safeguard does not alter the plan, source, or any measured evidence.

## New pre-compilation gate

The CPU-only [checker](../scripts/mi_research_10_preregistration.py) requires a **new** plan to carry `descriptor_claims`. Each claim names a numeric descriptor field, a direction (`more`, `fewer`, or `equal`), and the signed candidate-minus-control delta. The checker compares these with the two retained descriptor vectors. It walks nested numeric fields too, such as `per_grid_gates_by_phase.U`. All numeric fields must be covered, so a live-byte or dot claim cannot be omitted while other claims pass. It rejects missing, extra, malformed, and contradictory claims.

The exact historical error now produces an actionable failure:

```text
max_estimated_live_bytes_per_cta: expected more (candidate 63488 vs control 59392; delta +4096), stated fewer (delta -4096). Correct the new claim or its descriptor provenance before compilation; do not edit historical frozen evidence
```

The checker does not parse or rewrite prose. A new plan carrying the legacy `screening_hypotheses` prose field is rejected, even if it also has consistent structured claims. Keep independently authored compiler hypotheses in a separate field. Generate source comparison text from `summary(plan)` only after `validate(plan)` passes. `build_claims(plan, control_id, candidate_id)` can explicitly prepare a new claim block from the descriptors; it never changes the supplied plan or the original frozen file.

The [MI-Research-10 GPU runner](../scripts/run_mi_research_10.py) calls this gate before importing PyTorch or creating a GPU evidence directory. Its historical frozen plan lacks structured claims, so a fresh compilation from that old plan now stops. Historical `--verify` replay remains available. A future study must freeze a separate, consistent plan before compiling.

For a new plan JSON with structured claims, the standalone CPU check is:

```bash
PYTHONPATH=src:. python scripts/mi_research_10_preregistration.py NEW_PLAN.json
```

It prints deterministic candidate-versus-control summaries after validation. The offline tests construct a corrected plan fixture in memory and a temporary JSON file. They also reproduce the exact reversed 4,096-byte claim, check gates, dots, depth, nested phase counts, mapping-dependent direction, complete field coverage, nonmutation, and runner rejection before any GPU work:

```bash
PYTHONPATH=src:. python -m pytest -q tests/mi_research_10_preregistration_test.py
PYTHONPATH=src:. python scripts/analyze_mi_research_10.py --verify
```

This gate checks internal source-descriptor consistency only. It does not predict compiler allocation or latency, and it does not change the numerical reference, GPU kernels, or benchmark decision rule.
