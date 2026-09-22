#pragma once

#include "matrix.hpp"
#include "benchmark.hpp"
#include <map>

#include <string>

// Probe the current/default CUDA device and runtime. On failure, optionally
// return a diagnostic for demos and test skips; CUDA headers are not required.
bool cuda_available(std::string* reason = nullptr);

// Naive FP32 matmul on the current/default device. Input padding is copied;
// output padding is zero. Empty products return zeros without using CUDA.
// Invalid dimensions/strides throw std::invalid_argument; CUDA errors throw
// std::runtime_error. The two-argument overload returns tightly packed rows.
Matrix cuda_matmul(const Matrix& a, const Matrix& b);
Matrix cuda_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride);

enum class CudaMatmulKernel { naive, tiled, naive_fma, naive_no_fma, naive_reordered };
inline constexpr unsigned cuda_matmul_tile_size = 16;

Matrix cuda_matmul(const Matrix& a, const Matrix& b, CudaMatmulKernel kernel);
Matrix cuda_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride, CudaMatmulKernel kernel);

struct CudaMatmulBenchmark {
    Matrix naive;
    Matrix tiled;
    float naive_ms;
    float tiled_ms;
    benchmark::Statistics naive_stats, tiled_stats;
};

// Packed outputs, configurable warmups (default three), then repetitions
// CUDA-event measurements. Allocation, initialization and transfers are outside
// the timed regions. Statistics are computed afterwards. Empty products have zero elapsed time.
CudaMatmulBenchmark benchmark_cuda_matmul(const Matrix& a, const Matrix& b, int repetitions = 20, int warmups = 3);

struct CudaKernelMeasurement { Matrix output; benchmark::Statistics timing; };
CudaKernelMeasurement benchmark_cuda_kernel(const Matrix& a, const Matrix& b,
    CudaMatmulKernel kernel, int repetitions = 20, int warmups = 3);

// Best-effort device/runtime metadata; unavailable fields are "unknown".
std::map<std::string, std::string> cuda_metadata();
