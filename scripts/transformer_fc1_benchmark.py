import argparse

import torch
import triton
import triton.language as tl
import statistics
import random

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
def fc1_kernel_flat_store(
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

    offs = (
        offs_m[:, None] * N
        + offs_n[None, :]
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        c_ptr + offs,
        c,
        mask=mask,
    )

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

@triton.jit
def fc1_kernel_transposed_b(
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

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_k = tl.arange(0, BLOCK_K)

    # A is loaded normally as [BLOCK_M, BLOCK_K].
    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    # B is deliberately loaded as [BLOCK_N, BLOCK_K]
    # instead of [BLOCK_K, BLOCK_N].
    b_ptrs = (
        b_ptr
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=offs_m[:, None] < M,
            other=0.0,
        )

        b_transposed = tl.load(
            b_ptrs,
            mask=offs_n[:, None] < N,
            other=0.0,
        )

        # Convert [BLOCK_N, BLOCK_K]
        # into [BLOCK_K, BLOCK_N] for tl.dot.
        b = tl.trans(b_transposed)

        accumulator += tl.dot(
            a,
            b,
        )

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    output = accumulator.to(tl.float16)

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    output_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        c_ptrs,
        output,
        mask=output_mask,
    )

def triton_fc1_transposed_b(
    a,
    b,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    group_size_m=1,
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

    fc1_kernel_transposed_b[grid](
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
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c

def triton_fc1(
    a,
    b,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    group_size_m=1,
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
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c

def triton_fc1_flat_store(
    a,
    b,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    group_size_m=1,
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

    fc1_kernel_flat_store[grid](
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
        GROUP_SIZE_M=group_size_m,
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
    parser.add_argument(
        "--profile-warps2",
        action="store_true",
    )
    parser.add_argument("--profile-64x128-4w-2s", action="store_true",)
    parser.add_argument(
        "--profile-64x128-4w-3s",
        action="store_true",
    )
    parser.add_argument(
        "--profile-flat-store",
        action="store_true",
    )
    parser.add_argument(
        "--profile-tile",
        choices=[
            "32x64",
            "32x128",
            "64x64",
            "64x128",
            "128x64",
            "128x128",
        ],
    )
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
        group_size_m=1,
    )

    max_error = (
        torch_output - triton_output
    ).abs().max().item()

    torch_ms = benchmark_fn(
        lambda: torch.matmul(a, b),
        args.warmup,
        args.iterations,
    )

    flops = 2 * args.m * args.k * args.n

    tile_configs = {
        "32x64": {
            "block_m": 32,
            "block_n": 64,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },
        "32x128": {
            "block_m": 32,
            "block_n": 128,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },

        "64x64": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },
        "64x128": {
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },
        "128x64": {
            "block_m": 128,
            "block_n": 64,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },
        "128x128": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 1,
        },
    }

    if args.profile_tile:
        cfg = tile_configs[args.profile_tile]

        with torch.cuda.nvtx.range(f"tile_{args.profile_tile}"):
            triton_fc1(
                a,
                b,
                block_m=cfg["block_m"],
                block_n=cfg["block_n"],
                block_k=cfg["block_k"],
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
                group_size_m=cfg["group_size_m"],
            )

        torch.cuda.synchronize()
        return

    if args.profile_flat_store:
        with torch.cuda.nvtx.range("triton_fc1_flat_store"):
            triton_fc1_flat_store(
                a,
                b,
                block_m=64,
                block_n=128,
                block_k=32,
                num_warps=4,
                num_stages=3,
                group_size_m=1,
            )

        torch.cuda.synchronize()
        return

    if args.profile_warps2:
        with torch.cuda.nvtx.range("triton_fc1_warps2"):
            triton_fc1(
                a,
                b,
                block_m=64,
                block_n=128,
                block_k=32,
                num_warps=2,
                num_stages=2,
                group_size_m=1,
            )

        torch.cuda.synchronize()
        return

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

    if args.profile_64x128_4w_3s:
        with torch.cuda.nvtx.range("triton_fc1_64x128_4w_s3"):
            triton_fc1(
                a,
                b,
                block_m=64,
                block_n=128,
                block_k=32,
                num_warps=4,
                num_stages=3,
                group_size_m=1,
            )

        torch.cuda.synchronize()
        return

    if args.profile_64x128_4w_2s:
        with torch.cuda.nvtx.range("triton_fc1_64x128_4w_s2"):
            triton_fc1(
                a,
                b,
                block_m=64,
                block_n=128,
                block_k=32,
                num_warps=4,
                num_stages=2,
                group_size_m=1,
            )

        torch.cuda.synchronize()
        return

    transposed_output = triton_fc1_transposed_b(
        a,
        b,
        block_m=64,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=3,
        group_size_m=1,
    )

    flat_output = triton_fc1_flat_store(
        a,
        b,
        block_m=64,
        block_n=128,
        block_k=32,
        num_warps=4,
        num_stages=3,
        group_size_m=1,
    )

    flat_max_error = (
        torch_output - flat_output
    ).abs().max().item()

    print()
    print("Flat-store correctness")
    print("----------------------")
    print(f"Max abs error: {flat_max_error}")
    print(
        "Allclose:",
        torch.allclose(
            torch_output,
            flat_output,
            atol=1e-2,
            rtol=1e-2,
        ),
    )

    transposed_max_error = (
        torch_output - transposed_output
    ).abs().max().item()

    print()
    print("Transposed-B correctness")
    print("------------------------")
    print(f"Max abs error: {transposed_max_error}")
    print(
        "Allclose:",
        torch.allclose(
            torch_output,
            transposed_output,
            atol=1e-2,
            rtol=1e-2,
        ),
    )

    benchmark_configs = {
        "torch": lambda: torch.matmul(a, b),

        "current": lambda: triton_fc1(
            a,
            b,
            block_m=64,
            block_n=128,
            block_k=32,
            num_warps=4,
            num_stages=3,
            group_size_m=1,
        ),

        "flat_store": lambda: triton_fc1_flat_store(
            a,
            b,
            block_m=64,
            block_n=128,
            block_k=32,
            num_warps=4,
            num_stages=3,
            group_size_m=1,
        ),
    }
    samples = {
        name: []
        for name in benchmark_configs
    }

    rounds = 100

    print()
    print("Interleaved benchmark")
    print("---------------------")

    for _ in range(rounds):
        names = list(benchmark_configs.keys())
        random.shuffle(names)

        for name in names:
            ms = benchmark_fn(
                benchmark_configs[name],
                args.warmup,
                args.iterations,
            )

            samples[name].append(ms)

    for name, values in samples.items():
        median_ms = statistics.median(values)
        mean_ms = statistics.mean(values)
        stdev_ms = statistics.stdev(values)

        tflops = flops / (median_ms / 1000.0) / 1e12

        print(
            f"{name:<12} "
            f"median={median_ms:.6f} ms  "
            f"mean={mean_ms:.6f} ms  "
            f"std={stdev_ms:.6f} ms  "
            f"{tflops:.2f} TFLOP/s"
        )

    print()
    print("Correctness")
    print("-----------")
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