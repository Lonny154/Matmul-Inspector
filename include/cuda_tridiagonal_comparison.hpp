#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace matmul_inspector {

struct TridiagonalComparisonConfig {
    std::vector<std::size_t> system_sizes{128,256,512,1024};
    std::vector<std::size_t> batch_sizes{1,32,1024};
    int warmups=5;
    int rounds=5;
    int iterations=20;
    std::uint32_t seed=20261012;
    std::string output="results/thomas-vs-pcr";
    bool show_help=false;
};

TridiagonalComparisonConfig parse_tridiagonal_comparison_cli(
    const std::vector<std::string>& arguments);
std::string tridiagonal_comparison_help();
int run_cuda_tridiagonal_comparison(const TridiagonalComparisonConfig& config);

}  // namespace matmul_inspector
