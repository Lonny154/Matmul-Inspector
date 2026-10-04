#pragma once

#include "benchmark.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace matmul_inspector {

struct TridiagonalSystem {
    std::vector<double> lower, diag, upper, rhs, expected;
};

struct TridiagonalBenchmarkResult {
    std::string algorithm, backend, timing_scope, batch_execution;
    // Number of independent systems included in one timing sample. This does
    // not imply that the systems execute concurrently.
    std::size_t system_size = 0, batch_size = 0;
    int warmups = 0, iterations = 0;
    benchmark::Statistics timing;
};

std::vector<TridiagonalSystem> make_benchmark_systems(
    std::size_t system_size, std::size_t batch_size);

void verify_tridiagonal_solution(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    double tolerance = 1e-9);

TridiagonalBenchmarkResult benchmark_thomas_batch(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

TridiagonalBenchmarkResult benchmark_cpu_pcr_batch(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups);

std::string tridiagonal_benchmark_csv_header();
std::string tridiagonal_benchmark_csv_row(const TridiagonalBenchmarkResult& result);

}  // namespace matmul_inspector
