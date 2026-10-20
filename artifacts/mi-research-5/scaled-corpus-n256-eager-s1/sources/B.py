"""Frozen MI-Research-5 rank-7 variant 0b88a39c7ca3237f69af292cf82e7ba7c4d3873d4dd4f032192ff22347da7263.
Term order: (0, 1, 2, 3, 4, 5, 6). Generated before GPU execution.
"""

import torch
import triton
import triton.language as tl

PARAMETERS = {"block_m": 16, "block_n": 16, "block_k": 32,
              "num_warps": 4, "num_stages": 2, "group_size_m": 1}

@triton.jit
def _product(
    a_ptr, b_ptr, rows, cols, inner,
    half: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
    v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
):
    product = tl.zeros((block_m, block_n), tl.float32)
    for k_base in range(0, half, block_k):
        k = k_base + inner
        a_mask = (rows[:, None] < half) & (k[None, :] < half)
        b_mask = (k[:, None] < half) & (cols[None, :] < half)
        left = tl.zeros((block_m, block_k), tl.float32)
        right = tl.zeros((block_k, block_n), tl.float32)
        if u0 != 0:
            left += u0 * tl.load(a_ptr + rows[:, None] * stride_am + k[None, :] * stride_ak,
                                 a_mask, other=0).to(tl.float32)
        if u1 != 0:
            left += u1 * tl.load(a_ptr + rows[:, None] * stride_am + (half + k[None, :]) * stride_ak,
                                 a_mask, other=0).to(tl.float32)
        if u2 != 0:
            left += u2 * tl.load(a_ptr + (half + rows[:, None]) * stride_am + k[None, :] * stride_ak,
                                 a_mask, other=0).to(tl.float32)
        if u3 != 0:
            left += u3 * tl.load(a_ptr + (half + rows[:, None]) * stride_am +
                                 (half + k[None, :]) * stride_ak, a_mask, other=0).to(tl.float32)
        if v0 != 0:
            right += v0 * tl.load(b_ptr + k[:, None] * stride_bk + cols[None, :] * stride_bn,
                                  b_mask, other=0).to(tl.float32)
        if v1 != 0:
            right += v1 * tl.load(b_ptr + k[:, None] * stride_bk + (half + cols[None, :]) * stride_bn,
                                  b_mask, other=0).to(tl.float32)
        if v2 != 0:
            right += v2 * tl.load(b_ptr + (half + k[:, None]) * stride_bk + cols[None, :] * stride_bn,
                                  b_mask, other=0).to(tl.float32)
        if v3 != 0:
            right += v3 * tl.load(b_ptr + (half + k[:, None]) * stride_bk +
                                  (half + cols[None, :]) * stride_bn, b_mask, other=0).to(tl.float32)
        product = tl.dot(left, right, product, input_precision="tf32x3", out_dtype=tl.float32)
    return product


@triton.jit
def _variant_kernel(
    a_ptr, b_ptr, out_ptr, half: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    inner = tl.arange(0, block_k)
    # Frozen corpus term 1.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 0, -1, 1, -1, 1, 0, 0, 1)
    c00 = -p
    c01 = -p
    c10 = p
    # Frozen corpus term 2.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 -1, -1, -1, -1, 1, 0, 0, 0)
    c00 += -p
    c01 += p
    # Frozen corpus term 3.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 0, 0, 1, 0, 0, 1, 0, -1)
    c01 += (-2 * p)
    c11 = p
    # Frozen corpus term 4.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 0, -1, 0, -1, -1, 0, 1, 0)
    c00 += (-2 * p)
    c10 += p
    # Frozen corpus term 5.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 0, 0, 1, 1, 0, 0, 0, 1)
    c00 += p
    c01 += -p
    c10 += -p
    c11 += p
    # Frozen corpus term 6.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 -1, 0, -2, 0, 1, 1, 0, 0)
    c01 += -p
    # Frozen corpus term 7.
    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,
                 stride_bk, stride_bn, block_m, block_n, block_k,
                 0, 1, 0, 2, 0, 0, 1, 1)
    c00 += -p
    c10 += p
    mask = (rows[:, None] < half) & (cols[None, :] < half)
    tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,
             c00.to(tl.float16), mask)
    tl.store(out_ptr + rows[:, None] * stride_om + (half + cols[None, :]) * stride_on,
             c01.to(tl.float16), mask)
    tl.store(out_ptr + (half + rows[:, None]) * stride_om + cols[None, :] * stride_on,
             c10.to(tl.float16), mask)
    tl.store(out_ptr + (half + rows[:, None]) * stride_om + (half + cols[None, :]) * stride_on,
             c11.to(tl.float16), mask)


def matmul_into(a, b, out, parameters):
    if parameters != PARAMETERS:
        raise ValueError("MI-Research-5 uses one frozen 16x16x32 configuration")
    if (a.ndim != 2 or b.ndim != 2 or out.ndim != 2 or a.shape != b.shape or
            a.shape != out.shape or a.shape[0] not in (128, 256) or
            a.dtype != torch.float16 or b.dtype != torch.float16 or
            out.dtype != torch.float16 or not a.is_cuda or not b.is_cuda or not out.is_cuda):
        raise ValueError("expected matching 128 or 256 square CUDA FP16 matrices")
    half = a.shape[0] // 2
    grid = (triton.cdiv(half, 16), triton.cdiv(half, 16))
    return _variant_kernel[grid] (
        a, b, out, half,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        16, 16, 32, num_warps=4, num_stages=2)
