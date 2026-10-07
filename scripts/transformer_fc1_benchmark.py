import argparse

import torch
import triton
import triton.language as tl


def benchmark_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        torch.matmul(a, b)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        torch.matmul(a, b)

    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)

    return total_ms / iterations

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
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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

def triton_fc1(
    a,
    b,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
):
    m, k = a.shape
    _, n = b.shape

    c = torch.empty(
        (m, n),
        device=a.device,
        dtype=a.dtype,
    )

    grid = (
        triton.cdiv(m, block_m)
        * triton.cdiv(n, block_n),
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c

def benchmark_fn(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        fn()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--profile-64x64", action="store_true",)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")

    a = torch.randn(
        args.m,
        args.k,
        device=device,
        dtype=torch.float16,
    )

    b = torch.randn(
        args.k,
        args.n,
        device=device,
        dtype=torch.float16,
    )

    torch_output = torch.matmul(a, b)

    triton_output = triton_fc1(
        a,
        b,
        block_m=128,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=3,
    )

    max_error = (
        torch_output - triton_output
    ).abs().max().item()

    torch_ms = benchmark_fn(
        lambda: torch.matmul(a, b),
        args.warmup,
        args.iterations,
    )

    stage_configs = [1, 2, 3, 4]

    print()
    print("Stage sweep — 64x64x32, 4 warps")
    print("--------------------------------")

    for stages in stage_configs:
        ms = benchmark_fn(
            lambda s=stages: triton_fc1(
                a,
                b,
                block_m=64,
                block_n=64,
                block_k=32,
                num_warps=4,
                num_stages=s,
            ),
            args.warmup,
            args.iterations,
        )
        flops = 2 * args.m * args.k * args.n

        tflops = flops / (ms / 1000.0) / 1e12

        print(
            f"stages={stages}  "
            f"{ms:.6f} ms  "
            f"{tflops:.2f} TFLOP/s  "
            f"{ms / torch_ms:.3f}x torch"
        )

    if args.profile_64x64:
        with torch.cuda.nvtx.range("triton_fc1_64x64_s2"):
            triton_fc1(
                a,
                b,
                block_m=64,
                block_n=64,
                block_k=32,
                num_warps=4,
                num_stages=2,
            )

        torch.cuda.synchronize()
        return

    configs = [
        (128, 128),
        (128, 64),
        (64, 128),
        (64, 64),
    ]

    print()
    print("Tile sweep")
    print("----------")

    for block_m, block_n in configs:
        triton_ms = benchmark_fn(
            lambda bm=block_m, bn=block_n: triton_fc1(
                a,
                b,
                block_m=bm,
                block_n=bn,
                block_k=32,
                num_warps=4,
                num_stages=3,
            ),
            args.warmup,
            args.iterations,
        )
        flops = 2 * args.m * args.k * args.n
        triton_tflops = flops / (triton_ms / 1000.0) / 1e12

        print(
            f"{block_m:>3}x{block_n:<3}  "
            f"{triton_ms:.6f} ms  "
            f"{triton_tflops:.2f} TFLOP/s  "
            f"{triton_ms / torch_ms:.3f}x torch"
        )

    flops = 2 * args.m * args.k * args.n

    torch_tflops = flops / (torch_ms / 1000.0) / 1e12
    triton_tflops = flops / (triton_ms / 1000.0) / 1e12

    print()
    print("Performance")
    print("-----------")
    print(
        f"Torch/cuBLAS: {torch_ms:.6f} ms  "
        f"{torch_tflops:.2f} TFLOP/s"
    )
    print(
        f"Triton:      {triton_ms:.6f} ms  "
        f"{triton_tflops:.2f} TFLOP/s"
    )

    print(
        f"Triton / Torch latency: "
        f"{triton_ms / torch_ms:.3f}x"
    )

    print(f"Max abs error: {max_error}")
    print(
        "Allclose:",
        torch.allclose(
            torch_output,
            triton_output,
            atol=1e-2,
            rtol=1e-2,
        ),
    )

if __name__ == "__main__":
    main()