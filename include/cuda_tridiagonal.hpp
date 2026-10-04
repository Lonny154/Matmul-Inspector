#pragma once

#include <vector>

namespace matmul_inspector {

std::vector<double> solve_pcr_cuda(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs);

std::vector<std::vector<double>> solve_pcr_cuda_batched(
    const std::vector<std::vector<double>>& lower,
    const std::vector<std::vector<double>>& diag,
    const std::vector<std::vector<double>>& upper,
    const std::vector<std::vector<double>>& rhs);

}  // namespace matmul_inspector
