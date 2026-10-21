# Paired FC1 A/B comparison

FP16 M=64, K=64, N=64; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 32, "block_n": 32, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness FAIL/UNAVAILABLE; max absolute error None.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness FAIL/UNAVAILABLE; max absolute error None.

**Decision: FAILED** — trusted local experiment failed at timeout; details omitted from artifacts

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: UNAVAILABLE; median RAW/perf rate ratio UNAVAILABLE; five 20 ms windows when RAW is available.
- Warnings: none.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=None bytes; registers/thread=None; spills=None; warps=None; stages=None; PTX=UNAVAILABLE; decoded SASS=UNAVAILABLE.
- B compiler: shared=None bytes; registers/thread=None; spills=None; warps=None; stages=None; PTX=UNAVAILABLE; decoded SASS=UNAVAILABLE.
- Normalized PTX equal: UNAVAILABLE; changed lines: UNAVAILABLE.
- A PTX SHA-256: `UNAVAILABLE`.
- B PTX SHA-256: `UNAVAILABLE`.
- PTX comparison: `UNAVAILABLE`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: UNAVAILABLE; driver: 616.92.
Software: {}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f2d6607653d04972bbc04757bd0ac5bf1864af6d` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4_n64_pair.json --output artifacts/mi-research-4/n64-eager-v2 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4/n64-eager-v2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
