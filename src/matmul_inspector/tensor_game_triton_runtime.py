"""Optional PyTorch/Triton runtime for validated Tensor Game lowerings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_triton_lowering import (
    validate_dot_precision,
    validate_execution_strategy,
    validate_reconstruction_strategy,
)


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


@triton.jit
def _fused_input_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    half: tl.constexpr,
    stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    stride_b0: tl.constexpr, stride_b1: tl.constexpr,
    stride_c0: tl.constexpr, stride_c1: tl.constexpr,
    u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
    v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    input_precision: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset in range(0, half, block_k):
        a = tl.zeros((block_m, block_k), dtype=tl.float32)
        b = tl.zeros((block_k, block_n), dtype=tl.float32)
        a_mask = (offs_m[:, None] < half) & (offset + offs_k[None, :] < half)
        b_mask = (offset + offs_k[:, None] < half) & (offs_n[None, :] < half)
        if u0 != 0:
            a += u0 * tl.load(a_ptr + offs_m[:, None] * stride_a0 + (offset + offs_k[None, :]) * stride_a1, mask=a_mask, other=0.0)
        if u1 != 0:
            a += u1 * tl.load(a_ptr + offs_m[:, None] * stride_a0 + (half + offset + offs_k[None, :]) * stride_a1, mask=a_mask, other=0.0)
        if u2 != 0:
            a += u2 * tl.load(a_ptr + (half + offs_m[:, None]) * stride_a0 + (offset + offs_k[None, :]) * stride_a1, mask=a_mask, other=0.0)
        if u3 != 0:
            a += u3 * tl.load(a_ptr + (half + offs_m[:, None]) * stride_a0 + (half + offset + offs_k[None, :]) * stride_a1, mask=a_mask, other=0.0)
        if v0 != 0:
            b += v0 * tl.load(b_ptr + (offset + offs_k[:, None]) * stride_b0 + offs_n[None, :] * stride_b1, mask=b_mask, other=0.0)
        if v1 != 0:
            b += v1 * tl.load(b_ptr + (offset + offs_k[:, None]) * stride_b0 + (half + offs_n[None, :]) * stride_b1, mask=b_mask, other=0.0)
        if v2 != 0:
            b += v2 * tl.load(b_ptr + (half + offset + offs_k[:, None]) * stride_b0 + offs_n[None, :] * stride_b1, mask=b_mask, other=0.0)
        if v3 != 0:
            b += v3 * tl.load(b_ptr + (half + offset + offs_k[:, None]) * stride_b0 + (half + offs_n[None, :]) * stride_b1, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision=input_precision, out_dtype=tl.float32)
    output_ptrs = c_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1
    tl.store(output_ptrs, accumulator, mask=(offs_m[:, None] < half) & (offs_n[None, :] < half))


@triton.jit
def _fused_reconstruction_kernel(
    p0, p1, p2, p3, p4, p5, p6, p7, output,
    half: tl.constexpr, rank: tl.constexpr,
    w00: tl.constexpr, w01: tl.constexpr, w02: tl.constexpr, w03: tl.constexpr,
    w10: tl.constexpr, w11: tl.constexpr, w12: tl.constexpr, w13: tl.constexpr,
    w20: tl.constexpr, w21: tl.constexpr, w22: tl.constexpr, w23: tl.constexpr,
    w30: tl.constexpr, w31: tl.constexpr, w32: tl.constexpr, w33: tl.constexpr,
    w40: tl.constexpr, w41: tl.constexpr, w42: tl.constexpr, w43: tl.constexpr,
    w50: tl.constexpr, w51: tl.constexpr, w52: tl.constexpr, w53: tl.constexpr,
    w60: tl.constexpr, w61: tl.constexpr, w62: tl.constexpr, w63: tl.constexpr,
    w70: tl.constexpr, w71: tl.constexpr, w72: tl.constexpr, w73: tl.constexpr,
    block: tl.constexpr,
):
    output_block = tl.program_id(0)
    offs_m = tl.program_id(1) * block + tl.arange(0, block)
    offs_n = tl.program_id(2) * block + tl.arange(0, block)
    mask = (offs_m[:, None] < half) & (offs_n[None, :] < half)
    product_offsets = offs_m[:, None] * half + offs_n[None, :]
    accumulator = tl.zeros((block, block), dtype=tl.float32)
    pointers = (p0, p1, p2, p3, p4, p5, p6, p7)
    coefficients = (
        (w00, w01, w02, w03), (w10, w11, w12, w13),
        (w20, w21, w22, w23), (w30, w31, w32, w33),
        (w40, w41, w42, w43), (w50, w51, w52, w53),
        (w60, w61, w62, w63), (w70, w71, w72, w73),
    )
    for index in tl.static_range(0, rank):
        row = coefficients[index]
        coefficient = tl.where(output_block == 0, row[0], tl.where(output_block == 1, row[1], tl.where(output_block == 2, row[2], row[3])))
        accumulator += coefficient * tl.load(pointers[index] + product_offsets, mask=mask, other=0.0)
    output_row = (output_block // 2) * half + offs_m
    output_col = (output_block % 2) * half + offs_n
    tl.store(output + output_row[:, None] * (2 * half) + output_col[None, :], accumulator, mask=mask)


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


def fused_input_gemm(left, right, u, v, precision: str = "ieee"):
    """Form one pair of 2x2 block combinations inside the GEMM load path."""
    precision = validate_dot_precision(precision)
    half = left.shape[0] // 2
    output = torch.empty((half, half), device=left.device, dtype=torch.float32)
    grid = (triton.cdiv(half, 32), triton.cdiv(half, 32))
    _fused_input_gemm_kernel[grid](
        left, right, output, half,
        left.stride(0), left.stride(1), right.stride(0), right.stride(1),
        output.stride(0), output.stride(1),
        *tuple(int(value) for value in u), *tuple(int(value) for value in v),
        block_m=32, block_n=32, block_k=16, input_precision=precision,
        num_warps=4, num_stages=3,
    )
    return output


def _combination(blocks, coefficients):
    result = torch.zeros_like(blocks[0], memory_format=torch.contiguous_format)
    for coefficient, block in zip(coefficients, blocks):
        if coefficient:
            result.add_(block, alpha=float(coefficient))
    return result


def fused_reconstruction(products, actions, output_size):
    """Reconstruct all four output blocks in one Triton kernel launch."""
    half = output_size // 2
    output = torch.empty((output_size, output_size), device=products[0].device, dtype=torch.float32)
    padded = list(products) + [products[0]] * (8 - len(products))
    coefficients = [int(value) for action in actions for value in action.w]
    coefficients += [0] * (32 - len(coefficients))
    grid = (4, triton.cdiv(half, 32), triton.cdiv(half, 32))
    _fused_reconstruction_kernel[grid](
        *padded, output, half, len(products), *coefficients, block=32,
        num_warps=4, num_stages=3,
    )
    return output


def baseline_reconstruction(products, actions, output_size):
    """Existing eager reconstruction path, retained as the default reference."""
    half = output_size // 2
    output_blocks = []
    for output_index in range(4):
        block = torch.zeros_like(products[0])
        for action, product in zip(actions, products):
            coefficient = action.w[output_index]
            if coefficient:
                block.add_(product, alpha=float(coefficient))
        output_blocks.append(block)
    output = torch.empty((output_size, output_size), device=products[0].device, dtype=torch.float32)
    for index, block in enumerate(output_blocks):
        i, j = divmod(index, 2)
        output[i * half:(i + 1) * half, j * half:(j + 1) * half].copy_(block)
    return output


def execute_lowering(
    lowering, left, right, precision: str = "ieee", strategy: str = "unfused",
    reconstruction: str = "baseline",
):
    """Execute a validated lowering with sequential materialization."""
    precision = validate_dot_precision(precision)
    strategy = validate_execution_strategy(strategy)
    reconstruction = validate_reconstruction_strategy(reconstruction)
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
    products = []
    if strategy == "unfused":
        a_blocks = tuple(left[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
        b_blocks = tuple(right[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
        for action in lowering.actions:
            products.append(fp32_gemm(_combination(a_blocks, action.u), _combination(b_blocks, action.v), precision))
    else:
        for action in lowering.actions:
            products.append(fused_input_gemm(left, right, action.u, action.v, precision))
    if reconstruction == "fused":
        return fused_reconstruction(products, lowering.actions, left.shape[0])
    return baseline_reconstruction(products, lowering.actions, left.shape[0])
