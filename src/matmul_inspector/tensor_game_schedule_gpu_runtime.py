"""Optional selected HA-Schedule-3 Triton lowerings. Import only with CUDA deps."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tensor_game_triton_runtime import execute_fused_preallocated, fused_input_gemm, fused_reconstruction


@triton.jit
def _materialize_u_form(
    a_ptr, form_ptr, half: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    row = offset // half
    col = offset % half
    mask = offset < half * half
    value = tl.zeros((block,), dtype=tl.float32)
    if u0 != 0:
        value += u0 * tl.load(a_ptr + row * stride_a0 + col * stride_a1, mask=mask, other=0).to(tl.float32)
    if u1 != 0:
        value += u1 * tl.load(a_ptr + row * stride_a0 + (half + col) * stride_a1, mask=mask, other=0).to(tl.float32)
    if u2 != 0:
        value += u2 * tl.load(a_ptr + (half + row) * stride_a0 + col * stride_a1, mask=mask, other=0).to(tl.float32)
    if u3 != 0:
        value += u3 * tl.load(a_ptr + (half + row) * stride_a0 + (half + col) * stride_a1, mask=mask, other=0).to(tl.float32)
    tl.store(form_ptr + offset, value.to(tl.float16), mask=mask)


@triton.jit
def _shared_form_gemm(
    a_ptr, b_ptr, form_ptr, product_ptr,
    half: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    stride_b0: tl.constexpr, stride_b1: tl.constexpr,
    u0: tl.constexpr, u1: tl.constexpr, u2: tl.constexpr, u3: tl.constexpr,
    v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
    use_prepared: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    row = tl.program_id(0) * block_m + tl.arange(0, block_m)
    col = tl.program_id(1) * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    accum = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start in range(0, half, block_k):
        a_mask = (row[:, None] < half) & (start + ks[None, :] < half)
        b_mask = (start + ks[:, None] < half) & (col[None, :] < half)
        if use_prepared:
            a = tl.load(form_ptr + row[:, None] * half + start + ks[None, :],
                        mask=a_mask, other=0).to(tl.float32)
        else:
            a = tl.zeros((block_m, block_k), dtype=tl.float32)
            if u0 != 0:
                a += u0 * tl.load(a_ptr + row[:, None] * stride_a0 + (start + ks[None, :]) * stride_a1,
                                  mask=a_mask, other=0).to(tl.float32)
            if u1 != 0:
                a += u1 * tl.load(a_ptr + row[:, None] * stride_a0 + (half + start + ks[None, :]) * stride_a1,
                                  mask=a_mask, other=0).to(tl.float32)
            if u2 != 0:
                a += u2 * tl.load(a_ptr + (half + row[:, None]) * stride_a0 + (start + ks[None, :]) * stride_a1,
                                  mask=a_mask, other=0).to(tl.float32)
            if u3 != 0:
                a += u3 * tl.load(a_ptr + (half + row[:, None]) * stride_a0 + (half + start + ks[None, :]) * stride_a1,
                                  mask=a_mask, other=0).to(tl.float32)
            # Match the FP16 materialized form's rounding in the paired path.
            a = a.to(tl.float16).to(tl.float32)
        b = tl.zeros((block_k, block_n), dtype=tl.float32)
        if v0 != 0:
            b += v0 * tl.load(b_ptr + (start + ks[:, None]) * stride_b0 + col[None, :] * stride_b1,
                              mask=b_mask, other=0).to(tl.float32)
        if v1 != 0:
            b += v1 * tl.load(b_ptr + (start + ks[:, None]) * stride_b0 + (half + col[None, :]) * stride_b1,
                              mask=b_mask, other=0).to(tl.float32)
        if v2 != 0:
            b += v2 * tl.load(b_ptr + (half + start + ks[:, None]) * stride_b0 + col[None, :] * stride_b1,
                              mask=b_mask, other=0).to(tl.float32)
        if v3 != 0:
            b += v3 * tl.load(b_ptr + (half + start + ks[:, None]) * stride_b0 + (half + col[None, :]) * stride_b1,
                              mask=b_mask, other=0).to(tl.float32)
        accum = tl.dot(a, b, accum, input_precision="ieee", out_dtype=tl.float32)
    offsets = row[:, None] * half + col[None, :]
    tl.store(product_ptr + offsets, accum,
             mask=(row[:, None] < half) & (col[None, :] < half))


def allocate_workspace(lowering, size: int, mode: str):
    if mode not in {"reference", "stored_form", "recomputed_form"}:
        raise ValueError(f"unsupported GPU schedule mode: {mode}")
    products = [torch.empty((size // 2, size // 2), device="cuda", dtype=torch.float32)
                for _ in lowering.actions]
    output = torch.empty((size, size), device="cuda", dtype=torch.float16)
    prepared = (torch.empty((size // 2, size // 2), device="cuda", dtype=torch.float16)
                if mode == "stored_form" else None)
    return {"products": products, "output": output, "prepared": prepared}


def execute_selected(lowering, left, right, workspace, mode: str, capture: dict | None = None):
    """Execute one complete operation into preallocated buffers."""
    if left.dtype != torch.float16 or right.dtype != torch.float16:
        raise ValueError("selected GPU schedules require FP16 inputs")
    if mode == "reference":
        return execute_fused_preallocated(lowering, left, right,
                                          workspace["products"], workspace["output"])
    if mode not in {"stored_form", "recomputed_form"}:
        raise ValueError("unknown selected schedule mode")
    if len(lowering.actions) != 8 or lowering.actions[0].u != lowering.actions[1].u:
        raise ValueError("selected frozen rank-8 lowering lacks the shared first-two-term u form")
    half = left.shape[0] // 2
    form = workspace["prepared"]
    if mode == "stored_form":
        if form is None:
            raise ValueError("stored-form mode requires a prepared buffer")
        compiled = _materialize_u_form[(triton.cdiv(half * half, 256),)](
            left, form, half, left.stride(0), left.stride(1),
            *lowering.actions[0].u, block=256, num_warps=4)
        if capture is not None:
            capture["materialize_shared_u"] = compiled
    for index, (action, product) in enumerate(zip(lowering.actions, workspace["products"])):
        if index < 2:
            compiled = _shared_form_gemm[(triton.cdiv(half, 32), triton.cdiv(half, 32))](
                left, right, form if form is not None else left, product,
                half, left.stride(0), left.stride(1), right.stride(0), right.stride(1),
                *action.u, *action.v, use_prepared=mode == "stored_form",
                block_m=32, block_n=32, block_k=16, num_warps=4, num_stages=3)
            if capture is not None and index == 0:
                capture["shared_form_gemm"] = compiled
        else:
            common_capture = {} if capture is not None else None
            fused_input_gemm(left, right, action.u, action.v, output=product,
                             capture=common_capture)
            if capture is not None:
                capture[f"common_gemm_{index}"] = common_capture["compiled"]
    reconstruction_capture = {} if capture is not None else None
    output = fused_reconstruction(workspace["products"], lowering.actions,
                                  left.shape[0], output=workspace["output"],
                                  capture=reconstruction_capture)
    if capture is not None:
        capture["reconstruction"] = reconstruction_capture["compiled"]
    return output


def expected_kernel_names(rank: int, mode: str) -> list[str]:
    if mode == "reference":
        return ["_fused_input_gemm_kernel"] * rank + ["_fused_reconstruction_kernel"]
    if mode in {"stored_form", "recomputed_form"}:
        return (["_materialize_u_form"] if mode == "stored_form" else []) + [
            "_shared_form_gemm"] * 2 + ["_fused_input_gemm_kernel"] * (rank - 2) + [
                "_fused_reconstruction_kernel"]
    raise ValueError(f"unknown schedule mode: {mode}")
