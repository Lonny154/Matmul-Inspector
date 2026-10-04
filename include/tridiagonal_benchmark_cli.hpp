#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace matmul_inspector {

enum class TridiagonalBenchmarkMode {
    all,
    cpu,
    serial_host_loop,
    true_batched_gpu,
    true_batched_reuse,
    true_batched_device_resident
};

struct TridiagonalBenchmarkConfig {
    std::vector<std::size_t> system_sizes{32,64,128,256,512,1024,2048,4096};
    std::vector<std::size_t> batch_sizes{1,8,32,128,512};
    TridiagonalBenchmarkMode mode=TridiagonalBenchmarkMode::all;
    int warmups=3;
    int iterations=20;
    std::string output="results/tridiagonal-baseline";
    bool show_help=false;
};

TridiagonalBenchmarkConfig parse_tridiagonal_benchmark_cli(
    const std::vector<std::string>& arguments);

const char* tridiagonal_benchmark_mode_name(TridiagonalBenchmarkMode mode);
bool tridiagonal_benchmark_mode_includes(
    TridiagonalBenchmarkMode selected,TridiagonalBenchmarkMode path);
std::string tridiagonal_benchmark_help();

}  // namespace matmul_inspector
