"""Standalone FC1 Triton GEMM for MI-Research-1 execution-organization trials.

The JIT kernel body matches the packaged FC1 kernel. Candidate specs change
CTA grouping or warp count. This file does not import Matmul Inspector or
register itself with the package.
"""

import triton
import triton.language as tl

@triton.jit
def fc1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M

    group_size_m = tl.minimum(
        num_pid_m - first_pid_m,
        GROUP_SIZE_M,
    )

    pid_m = first_pid_m + (
        (pid % num_pid_in_group) % group_size_m
    )

    pid_n = (
        pid % num_pid_in_group
    ) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=offs_m[:, None] < M,
            other=0.0,
        )

        b = tl.load(
            b_ptrs,
            mask=offs_n[None, :] < N,
            other=0.0,
        )

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(c_ptrs, c, mask=mask)



def matmul_into(a, b, out, parameters):
    """Write FP16 (M,K) @ (K,N) into preallocated ``out``.

    Returning the Triton launch result lets Matmul Inspector retain PTX and
    resource metadata. Returning None is also allowed by the paired contract.
    """
    m, k = a.shape
    n = b.shape[1]
    block_m = parameters["block_m"]
    block_n = parameters["block_n"]
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    return fc1_kernel[grid](
        a, b, out, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_K=parameters["block_k"], GROUP_SIZE_M=parameters["group_size_m"],
        num_warps=parameters["num_warps"], num_stages=parameters["num_stages"],
    )
