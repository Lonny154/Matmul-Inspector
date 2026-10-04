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

}  // namespace matmul_inspector
