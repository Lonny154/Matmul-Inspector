#pragma once

#include <vector>

namespace matmul_inspector {

std::vector<double> solve_thomas(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs);

std::vector<double> solve_pcr(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs);

}  // namespace matmul_inspector