#pragma once

#include "benchmark.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace matmul_inspector {

struct TridiagonalSystem {
    std::vector<double> lower, diag, upper, rhs, expected;
};

struct TridiagonalBenchmarkResult {
    TridiagonalBenchmarkResult(
        std::string algorithm_value,std::string backend_value,
        std::string timing_scope_value,std::string batch_execution_value,
        std::size_t system_size_value,std::size_t batch_size_value,
        int warmups_value,int iterations_value,benchmark::Statistics timing_value,
        std::string selected_path_value={},std::string dispatch_rule_value={})
        : algorithm(std::move(algorithm_value)), backend(std::move(backend_value)),
          timing_scope(std::move(timing_scope_value)),
          batch_execution(std::move(batch_execution_value)),
          system_size(system_size_value), batch_size(batch_size_value),
          warmups(warmups_value), iterations(iterations_value),
          timing(std::move(timing_value)),
          selected_path(std::move(selected_path_value)),
          dispatch_rule(std::move(dispatch_rule_value)) {}

    std::string algorithm, backend, timing_scope, batch_execution;
    // Number of independent systems included in one timing sample. This does
    // not imply that the systems execute concurrently.
    std::size_t system_size = 0, batch_size = 0;
    int warmups = 0, iterations = 0;
    benchmark::Statistics timing;
    // Populated by policy-driven rows. Appended to the CSV so existing column
    // positions and baseline row semantics remain unchanged.
    std::string selected_path, dispatch_rule;
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
