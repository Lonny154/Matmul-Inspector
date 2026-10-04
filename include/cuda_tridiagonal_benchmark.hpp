#pragma once

#include "tridiagonal_benchmark.hpp"

#include <vector>

namespace matmul_inspector {

// These baseline benchmarks process systems sequentially from the host. They
// do not launch a kernel that exposes multiple systems as GPU parallelism.
TridiagonalBenchmarkResult benchmark_cuda_pcr_kernel_only(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cuda_pcr_end_to_end(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cuda_pcr_true_batched_kernel_only(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cuda_pcr_true_batched_end_to_end(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

// Returns allocation_setup, h2d, kernel_only, d2h, and reusable_end_to_end rows.
std::vector<TridiagonalBenchmarkResult> benchmark_cuda_pcr_true_batched_reuse(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

// Returns d2d_reset and device_resident rows. Input upload, output download,
// allocation, and correctness checks occur outside both timed regions.
std::vector<TridiagonalBenchmarkResult> benchmark_cuda_pcr_device_resident(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cuda_pcr_hybrid_device_resident(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cuda_pcr_fused_device_resident(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

}  // namespace matmul_inspector
