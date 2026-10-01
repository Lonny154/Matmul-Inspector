#pragma once
#include "operation.hpp"
#include <optional>
#include <cstdint>
#include <string>

namespace summation {
bool supported(const std::string& method);
// Existing cpu/cpu-reverse are accepted aliases. All returned outputs are FP32;
// fp64_accumulation computes the existing FP64 reference then casts once.
Matrix execute(const std::string& method, operation::Kind kind, const Matrix& a, const Matrix& b);
const char* fp_policy();
struct Analysis {
    float rounded_target = 0;
    bool target_bitwise_match = false;
    std::optional<std::uint32_t> ulp_to_target;
    std::optional<double> signed_error, absolute_error, relative_error;
    std::optional<double> baseline_error, error_reduction, reduction_factor;
    std::string factor_status = "unavailable", classification = "unavailable";
};
// Improvement baseline is always forward FP32, independent of the selected
// compare-mode reference. Nonfinite error metrics remain unavailable.
Analysis analyze(float value, float forward, double fp64);
}
