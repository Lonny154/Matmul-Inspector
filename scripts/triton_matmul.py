"""IEEE FP32 GEMM: correctness/benchmark by default, fixed launches for profiling.

Examples:
    python scripts/triton_matmul.py
    python scripts/triton_matmul.py --profile tuned
    python scripts/triton_matmul.py --profile weak

Profile modes launch once without autotuning or torch.matmul, so Nsight Compute
can target matmul_kernel_fixed. Existing callable wrappers remain available.
"""
import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_fixed(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

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
        k_offset = k * BLOCK_K

        a = tl.load(
            a_ptrs,
            mask=(
                (offs_m[:, None] < M)
                & (offs_k[None, :] + k_offset < K)
            ),
            other=0.0,
        )

        b = tl.load(
            b_ptrs,
            mask=(
                (offs_k[:, None] + k_offset < K)
                & (offs_n[None, :] < N)
            ),
            other=0.0,
        )

        accumulator += tl.dot(
            a,
            b,
            input_precision="ieee",
        )

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    c_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        c_ptrs,
        accumulator,
        mask=c_mask,
    )


# One arithmetic implementation; fixed launches bypass this autotuner entirely.
matmul_kernel = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)(matmul_kernel_fixed)

TUNED_PROFILE_CONFIG = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)
WEAK_PROFILE_CONFIG = dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4, num_stages=3)


def _launch_matmul(a: torch.Tensor, b: torch.Tensor, *, config=None) -> torch.Tensor:
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    kernel = matmul_kernel if config is None else matmul_kernel_fixed
    kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **({} if config is None else config),
    )
    return c


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Use the unchanged four-configuration, shape-keyed autotuner."""
    return _launch_matmul(a, b)


def triton_matmul_fixed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tuned 128x64x32, four-warps, three-stages profiling path."""
    return _launch_matmul(a, b, config=TUNED_PROFILE_CONFIG)


def triton_matmul_fixed_bad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Deliberately weak 32x32x32 profile; legacy callable name retained."""
    return _launch_matmul(a, b, config=WEAK_PROFILE_CONFIG)


def _profile_once(matmul) -> None:
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    c = matmul(a, b)
    torch.cuda.synchronize()
    print(c[0, 0].item())


def profile_triton_once() -> None:
    _profile_once(triton_matmul_fixed)


def profile_triton_bad_once() -> None:
    _profile_once(triton_matmul_fixed_bad)


def benchmark_matmul(M: int, N: int, K: int) -> None:
    a = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float32,
    )

    b = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float32,
    )

    # Compile / warm up both paths before benchmarking.
    triton_matmul(a, b)
    torch.matmul(a, b)
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        lambda: triton_matmul(a, b),
        warmup=100,
        rep=500,
        return_mode="median",
    )

    torch_ms = triton.testing.do_bench(
        lambda: torch.matmul(a, b),
        warmup=100,
        rep=500,
        return_mode="median",
    )

    flops = 2.0 * M * N * K

    triton_gflops = flops / (triton_ms * 1e6)
    torch_gflops = flops / (torch_ms * 1e6)

    print()
    print(f"Benchmark: {M}x{K} @ {K}x{N}")
    print(
        f"Triton: {triton_ms:.6f} ms "
        f"({triton_gflops:.2f} GFLOP/s)"
    )
    print(
        f"cuBLAS: {torch_ms:.6f} ms "
        f"({torch_gflops:.2f} GFLOP/s)"
    )
    print(
        f"Triton / cuBLAS performance: "
        f"{triton_gflops / torch_gflops:.3f}x"
    )


def check_correctness() -> None:
    torch.manual_seed(42)

    M = 256
    K = 256
    N = 256

    a = torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.float32,
    )

    b = torch.randn(
        (K, N),
        device="cuda",
        dtype=torch.float32,
    )

    triton_output = triton_matmul(a, b)
    torch_output = torch.matmul(a, b)

    max_abs_error = (
        triton_output - torch_output
    ).abs().max().item()

    mean_abs_error = (
        triton_output - torch_output
    ).abs().mean().item()

    print(f"shape: {M}x{K} @ {K}x{N}")
    print(f"max abs error:  {max_abs_error:.8e}")
    print(f"mean abs error: {mean_abs_error:.8e}")

    if torch.allclose(
        triton_output,
        torch_output,
        rtol=1e-4,
        atol=1e-4,
    ):
        print("PASS: Triton matches torch.matmul")
    else:
        print("FAIL: Triton differs from torch.matmul")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("tuned", "weak"),
                        help="launch one fixed 1024-square GEMM, without autotuning or benchmarking")
    args = parser.parse_args()
    if args.profile == "tuned":
        profile_triton_once()
    elif args.profile == "weak":
        profile_triton_bad_once()
    else:
        check_correctness()
        benchmark_matmul(1024, 1024, 1024)


if __name__ == "__main__":
    main()
