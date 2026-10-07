from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class TritonGemmConfig:
    block_m: int = 64
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3
    group_size_m: int = 1


# Keep the established FC1 configuration unchanged. QKV was selected by the
# controlled RTX 4060 Ti projection study; tuning remains hardware-specific.
# FC2 retains the original configuration because no candidate passed the
# repeatability rule.
FC1_CONFIG = TritonGemmConfig()
QKV_CONFIG = TritonGemmConfig(
    block_m=64,
    block_n=64,
    block_k=64,
    num_warps=4,
    num_stages=2,
    group_size_m=1,
)
FC2_CONFIG = FC1_CONFIG
DEFAULT_CONFIG = FC1_CONFIG


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


def triton_gemm(a, b, config=DEFAULT_CONFIG):
    """Compute a 2-D FP16 GEMM with the shared Transformer Triton kernel."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_gemm expects two 2-D tensors")

    m, k = a.shape
    b_k, n = b.shape

    if k != b_k:
        raise ValueError(f"incompatible GEMM dimensions: {a.shape} and {b.shape}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_gemm expects CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("triton_gemm currently supports FP16 inputs only")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("triton_gemm expects contiguous inputs")
    if k % config.block_k:
        raise ValueError(f"K={k} must be divisible by BLOCK_K={config.block_k}")

    c = torch.empty(
        (m, n),
        device=a.device,
        dtype=a.dtype,
    )

    grid = (
        triton.cdiv(m, config.block_m)
        * triton.cdiv(n, config.block_n),
    )

    fc1_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_SIZE_M=config.group_size_m,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    return c


def triton_fc1(a, b):
    """Backward-compatible name for the original FC1 integration."""
    return triton_gemm(a, b, FC1_CONFIG)
