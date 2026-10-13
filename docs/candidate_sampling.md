# Kernel candidate sampling

Candidate sampling is the CPU-only front end of a future kernel search loop. It
describes a workload and hardware target, validates a finite Triton GEMM search
space, and selects configurations without compiling or benchmarking kernels.

```python
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler,
)

space = TritonGemmSearchSpace(
    {
        "block_m": [32, 64, 128],
        "block_n": [64, 128, 256],
        "block_k": [16, 32, 64],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4],
    },
    SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
    identifier="fc1-v1",
)
candidates = UniformSampler(space, seed=42).sample(10, unique=True)
```

Uniform sampling gives every valid configuration equal probability. Weighted
sampling accepts a weight for every value of every parameter. It multiplies
the normalized parameter weights to obtain each candidate's weight. This is an
initial approximation. It does not model interactions between parameters and
does not learn from benchmark results. With unique sampling, only candidates
with strictly positive product weight are eligible. Selection is sequential
probability-proportional sampling without replacement: after each draw, the
selected candidate is removed and probabilities are recalculated over the
remaining eligible candidates. An impossible request is rejected before the
random generator or duplicate-tracking state changes.

Each sampler owns an isolated pseudorandom generator. The same search space,
constraints, sampler, seed, and sequence of calls produces the same candidate
sequence. Timestamps are only added when sampling records are created. They do
not affect candidate identity.

Individual search values must be positive integers. Candidate-level validity is
separate and explicit. Add a named `CandidateConstraint` for relationships such
as a tile shape requiring a particular warp count or a target-specific resource
limit. A value appearing in the search space does not establish hardware
validity. Constraints run before sampling, and a finite unique sample raises
`SearchSpaceExhausted` when too few eligible candidates remain.

The full Cartesian product is validated in deterministic parameter order on
first use and cached as an immutable tuple shared by uniform and weighted
samplers. For parameter-list lengths `d1` through `d5`, construction takes
`O(d1*d2*d3*d4*d5)` constraint evaluations and up to the same order of memory.
This implementation is intended for bounded discrete spaces. Constraint
predicates must be pure and stable because changes to external state after the
first enumeration do not invalidate the cache.

`make_sampling_records`, `write_jsonl`, and `read_jsonl` provide a versioned
JSONL record. It includes stable candidate identity, context, sampling order,
and an `extensions` object reserved for later compilation, correctness,
latency, profiler-path, and failure data. See
`examples/sample_kernel_candidates.py` for an executable round trip.
Version 1 has strict required and top-level fields. Optional future data belongs
inside `extensions`. Adding a required field or any new top-level field requires
a new schema version and a dedicated parser. Existing version-1 records retain
their version-1 parsing path.

The longer-term search loop will send valid candidates to the existing
experiment runner, attach measured outcomes to these records, and use that
evidence to guide later sampling. No feedback or automatic weight updates are
implemented yet.

The high-level inspiration is AlphaTensor's use of sampled actions and measured
hardware runtime. The scope is different: AlphaTensor searched for mathematical
matrix multiplication algorithms using tensor decompositions. This foundation
samples implementation configurations for conventional GEMM kernels. It is not
an AlphaTensor implementation and contains no reinforcement learning, neural
network, or tree search.
