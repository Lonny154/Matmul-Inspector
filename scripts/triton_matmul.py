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

@triton.autotune(
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
)

@triton.jit
def matmul_kernel(
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
    # Which output tile is this Triton program responsible for?
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row and column indices for this output tile.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for the first K tile.
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

    # FP32 accumulator for this BLOCK_M x BLOCK_N output tile.
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # Walk across K one BLOCK_K chunk at a time.
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

        # Advance A across columns and B down rows.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Output pointers.
    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    c_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_fixed(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    M, K = a.shape
    _, N = b.shape

    c = torch.empty(
        (M, N),
        device=a.device,
        dtype=torch.float32,
    )

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_kernel_fixed[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return c

def triton_matmul(a: torch.Tensor, b: torch.Tensor,) -> torch.Tensor:
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    M, K = a.shape
    _, N = b.shape

    c = torch.empty(
        (M, N),
        device=a.device,
        dtype=torch.float32,
    )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c

def profile_triton_once() -> None:
    M = N = K = 1024

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

    # Use the already-autotuned wrapper once.
    c = triton_matmul_fixed(a, b)

    torch.cuda.synchronize()

    # Prevent Python from optimizing the result away conceptually
    print(c[0, 0].item())

def benchmark_stages(M: int, N: int, K: int) -> None:
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    block_m = 128
    block_n = 64
    block_k = 32
    warps = 4

    print()
    print(f"Stage sweep: {M}x{K} @ {K}x{N}")

    for stages in [1, 2, 3, 4, 5]:
        triton_matmul(
            a,
            b,
            block_m,
            block_n,
            block_k,
            warps,
            stages,
        )
        torch.cuda.synchronize()

        ms = triton.testing.do_bench(
            lambda: triton_matmul(
                a,
                b,
                block_m,
                block_n,
                block_k,
                warps,
                stages,
            ),
            warmup=50,
            rep=200,
            return_mode="median",
        )

        flops = 2.0 * M * N * K
        gflops = flops / (ms * 1e6)

        print(
            f"{stages} stages: "
            f"{ms:.6f} ms  "
            f"{gflops:.2f} GFLOP/s"
        )

def benchmark_warps(M: int, N: int, K: int) -> None:
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    block_m = 128
    block_n = 64
    block_k = 32

    print()
    print(f"Warp sweep: {M}x{K} @ {K}x{N}")

    for warps in [2, 4, 8]:
        triton_matmul(
            a,
            b,
            block_m,
            block_n,
            block_k,
            warps,
        )
        torch.cuda.synchronize()

        ms = triton.testing.do_bench(
            lambda: triton_matmul(
                a,
                b,
                block_m,
                block_n,
                block_k,
                warps,
            ),
            warmup=50,
            rep=200,
            return_mode="median",
        )

        flops = 2.0 * M * N * K
        gflops = flops / (ms * 1e6)

        print(
            f"{warps} warps: "
            f"{ms:.6f} ms  "
            f"{gflops:.2f} GFLOP/s"
        )

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

def benchmark_configs(M: int, N: int, K: int) -> None:
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    configs = [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 32),
        (64, 64, 64),
        (128, 64, 32),
        (64, 128, 32),
    ]

    print()
    print(f"Tile sweep: {M}x{K} @ {K}x{N}")

    for block_m, block_n, block_k in configs:
        triton_matmul(
            a,
            b,
            block_m,
            block_n,
            block_k,
        )
        torch.cuda.synchronize()

        ms = triton.testing.do_bench(
            lambda: triton_matmul(
                a,
                b,
                block_m,
                block_n,
                block_k,
            ),
            warmup=50,
            rep=200,
            return_mode="median",
        )

        flops = 2.0 * M * N * K
        gflops = flops / (ms * 1e6)

        print(
            f"{block_m:3d}x{block_n:3d}x{block_k:3d}: "
            f"{ms:.6f} ms  "
            f"{gflops:.2f} GFLOP/s"
        )

def main_old():
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

    #benchmark_matmul(1024, 1024, 1024)

    profile_triton_once()

   # for size in [256, 512, 1024, 2048]:
        #benchmark_matmul(size, size, size)

def main():
    profile_triton_once()


if __name__ == "__main__":
    main()