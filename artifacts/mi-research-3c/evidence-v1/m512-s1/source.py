"""Trusted-local two-way Split-K FP16 GEMM for MI-Research-2.

Each call allocates FP32 partial storage, launches a partitioned GEMM, then
launches a deterministic reduction into the caller's FP16 output. Both
launches and the allocation remain inside Matmul Inspector's timed wrapper.
"""

import torch
import triton
import triton.language as tl


SPLIT_K = 2
REDUCTION_BLOCK = 1024


@triton.jit
def split_k_partial(
    a_ptr, b_ptr, partial_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, K_PART: tl.constexpr,
):
    output_tile = tl.program_id(0)
    partition = tl.program_id(1)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    rows = output_tile // num_tiles_n * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = output_tile % num_tiles_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rows[:, None] * stride_am + (
        partition * K_PART + k_offsets[None, :]) * stride_ak
    b_ptrs = b_ptr + (
        partition * K_PART + k_offsets[:, None]) * stride_bk + cols[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_PART, BLOCK_K):
        a = tl.load(a_ptrs, mask=rows[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    partial_ptrs = partial_ptr + partition * M * N + rows[:, None] * N + cols[None, :]
    tl.store(partial_ptrs, accumulator, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def reduce_partials(
    partial_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < M * N
    first = tl.load(partial_ptr + offsets, mask=valid, other=0.0)
    second = tl.load(partial_ptr + M * N + offsets, mask=valid, other=0.0)
    out_offsets = offsets // N * stride_cm + offsets % N * stride_cn
    tl.store(out_ptr + out_offsets, (first + second).to(tl.float16), mask=valid)


def matmul_into(a, b, out, parameters):
    """Write the complete two-launch operation into preallocated ``out``."""
    m, k = a.shape
    n = b.shape[1]
    block_m = parameters["block_m"]
    block_n = parameters["block_n"]
    block_k = parameters["block_k"]
    if k % (SPLIT_K * block_k):
        raise ValueError("K must be divisible by both the partition count and block_k")

    # Allocation is deliberately per call and inside the measured operation.
    partials = torch.empty((SPLIT_K, m, n), device=a.device, dtype=torch.float32)
    partial_grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), SPLIT_K)
    split_k_partial[partial_grid](
        a, b, partials, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        K_PART=k // SPLIT_K,
        num_warps=parameters["num_warps"], num_stages=parameters["num_stages"],
    )
    # The returned launch exposes reduction PTX/metadata to the current
    # one-result compiler capture. It does not represent the partial kernel.
    return reduce_partials[(triton.cdiv(m * n, REDUCTION_BLOCK),)](
        partials, out, m, n, out.stride(0), out.stride(1),
        BLOCK=REDUCTION_BLOCK, num_warps=4,
    )
