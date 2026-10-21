"""Experimental 3x3 staged product/reconstruction GPU mapping.

The same kernels execute the conventional 27-product control and Laderman's
23-product decomposition. Coefficients are caller-owned int32 device values.
Products and output are caller-owned buffers. No allocation occurs in call().
"""

import torch
import triton
import triton.language as tl

SIZE = 96
HALF_TILE = 16
K_TILE = 32


@triton.jit
def _block_product(a_ptr, b_ptr, p_ptr, coeff_ptr, term,
                   n: tl.constexpr, h: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * 16 + tl.arange(0, 16)
    inner = tl.arange(0, 32)
    acc = tl.zeros((16, 16), tl.float32)
    for base in range(0, h, 32):
        a = tl.zeros((16, 32), tl.float32)
        b = tl.zeros((32, 16), tl.float32)
        for block in tl.static_range(0, 9):
            u = tl.load(coeff_ptr + term * 27 + block)
            if u != 0:
                a_part = tl.load(a_ptr + (block // 3 * h + row[:, None]) * n
                                 + block % 3 * h + base + inner[None, :],
                                 mask=(row[:, None] < h) & (base + inner[None, :] < h), other=0)
                a += u.to(tl.float32) * a_part.to(tl.float32)
            v = tl.load(coeff_ptr + term * 27 + 9 + block)
            if v != 0:
                b_part = tl.load(b_ptr + (block // 3 * h + base + inner[:, None]) * n
                                 + block % 3 * h + col[None, :],
                                 mask=(base + inner[:, None] < h) & (col[None, :] < h), other=0)
                b += v.to(tl.float32) * b_part.to(tl.float32)
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    offset = term * h * h + row[:, None] * h + col[None, :]
    tl.store(p_ptr + offset, acc, mask=(row[:, None] < h) & (col[None, :] < h))


@triton.jit
def _reconstruct(p_ptr, c_ptr, coeff_ptr, n: tl.constexpr, h: tl.constexpr,
                 rank: tl.constexpr):
    out = tl.program_id(0)
    row = tl.program_id(1) * 16 + tl.arange(0, 16)
    col = tl.program_id(2) * 16 + tl.arange(0, 16)
    offset = row[:, None] * h + col[None, :]
    mask = (row[:, None] < h) & (col[None, :] < h)
    acc = tl.zeros((16, 16), tl.float32)
    for term in tl.static_range(0, rank):
        coefficient = tl.load(coeff_ptr + term * 27 + 18 + out).to(tl.float32)
        product = tl.load(p_ptr + term * h * h + offset, mask=mask, other=0)
        acc += coefficient * product
    out_row = out // 3 * h + row
    out_col = out % 3 * h + col
    tl.store(c_ptr + out_row[:, None] * n + out_col[None, :], acc, mask=mask)


def buffers(actions):
    """Allocate fixed-address buffers outside a measured complete operation."""
    rank = len(actions)
    if rank not in (23, 27):
        raise ValueError("expected the frozen 23- or 27-product algorithm")
    coefficients = torch.tensor([list(a.u) + list(a.v) + list(a.w) for a in actions],
                                device="cuda", dtype=torch.int32)
    products = torch.empty((rank, SIZE // 3, SIZE // 3), device="cuda", dtype=torch.float32)
    output = torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)
    return coefficients, products, output


def call(left, right, coefficients, products, output, launches=None):
    if (left.shape != (SIZE, SIZE) or right.shape != (SIZE, SIZE)
            or output.shape != (SIZE, SIZE) or left.dtype != torch.float16
            or right.dtype != torch.float16 or output.dtype != torch.float16
            or not left.is_contiguous() or not right.is_contiguous() or not output.is_contiguous()
            or coefficients.shape != (products.shape[0], 27)
            or products.shape != (coefficients.shape[0], SIZE // 3, SIZE // 3)):
        raise ValueError("unsupported 3x3 staged probe contract")
    rank = int(products.shape[0])
    h = SIZE // 3
    product_launch = None
    for term in range(rank):
        product_launch = _block_product[(triton.cdiv(h, HALF_TILE), triton.cdiv(h, HALF_TILE))](
            left, right, products, coefficients, term, SIZE, h, num_warps=4, num_stages=2)
    reduction_launch = _reconstruct[(9, triton.cdiv(h, HALF_TILE), triton.cdiv(h, HALF_TILE))](
        products, output, coefficients, SIZE, h, rank, num_warps=4, num_stages=2)
    if launches is not None:
        launches.update(product=product_launch, reconstruction=reduction_launch)
    return output
