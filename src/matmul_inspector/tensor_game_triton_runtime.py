"""Optional PyTorch/Triton runtime for validated Tensor Game lowerings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_triton_lowering import validate_dot_precision


@triton.jit
def _fp32_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    input_precision: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset in range(0, k, block_k):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (offset + offs_k[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(offset + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision=input_precision, out_dtype=tl.float32)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    output_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(output_ptrs, accumulator, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def fp32_gemm(left, right, precision: str = "ieee"):
    """Contiguous CUDA FP32 GEMM with an explicit Triton dot precision."""
    precision = validate_dot_precision(precision)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("fp32_gemm expects compatible 2-D matrices")
    if not left.is_cuda or not right.is_cuda:
        raise ValueError("fp32_gemm expects CUDA tensors")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError("fp32_gemm expects float32 tensors")
    if not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("fp32_gemm expects contiguous tensors")
    m, k = left.shape
    n = right.shape[1]
    output = torch.empty((m, n), device=left.device, dtype=torch.float32)
    block_m = block_n = 32
    block_k = 16
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fp32_gemm_kernel[grid](
        left, right, output, m, n, k,
        left.stride(0), left.stride(1), right.stride(0), right.stride(1),
        output.stride(0), output.stride(1),
        block_m=block_m, block_n=block_n, block_k=block_k,
        input_precision=precision,
        num_warps=4, num_stages=3,
    )
    return output


def _combination(blocks, coefficients):
    result = torch.zeros_like(blocks[0], memory_format=torch.contiguous_format)
    for coefficient, block in zip(coefficients, blocks):
        if coefficient:
            result.add_(block, alpha=float(coefficient))
    return result


def execute_lowering(lowering, left, right, precision: str = "ieee"):
    """Execute a validated lowering with sequential materialization."""
    precision = validate_dot_precision(precision)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("lowering expects two equally sized square 2-D matrices")
    if left.shape[0] != left.shape[1] or left.shape[0] % 2:
        raise ValueError("matrix dimensions must be square and even")
    if not left.is_cuda or not right.is_cuda:
        raise ValueError("lowering expects CUDA tensors")
    if left.device != right.device:
        raise ValueError("inputs must be on the same CUDA device")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError("lowering expects float32 inputs")
    if not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("lowering expects contiguous inputs")
    half = left.shape[0] // 2
    a_blocks = tuple(left[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
    b_blocks = tuple(right[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
    products = []
    for action in lowering.actions:
        products.append(fp32_gemm(_combination(a_blocks, action.u), _combination(b_blocks, action.v), precision))
    output_blocks = []
    for output_index in range(4):
        block = torch.zeros_like(products[0])
        for action, product in zip(lowering.actions, products):
            coefficient = action.w[output_index]
            if coefficient:
                block.add_(product, alpha=float(coefficient))
        output_blocks.append(block)
    output = torch.empty_like(left)
    for index, block in enumerate(output_blocks):
        i, j = divmod(index, 2)
        output[i * half:(i + 1) * half, j * half:(j + 1) * half].copy_(block)
    return output
