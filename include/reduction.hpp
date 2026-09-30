#pragma once
#include "operation.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace reduction {
bool fixture_supported(const std::string& fixture);
// Structured dot fixtures use B=1, so A explicitly describes the product terms.
void fill_structured(Matrix& a, Matrix& b, operation::Kind kind,
                     const std::string& fixture, std::uint32_t seed);
double reference_fp64(operation::Kind kind, const Matrix& a, const Matrix& b);
inline constexpr double relative_floor = 1e-12;
struct Observation {
    float value = 0, baseline = 0;
    double fp64_reference = 0;
    std::optional<double> absolute_error, relative_error;
    std::optional<std::uint32_t> ulp;
    bool tolerance_pass = false;
};
Observation observe(float value, float baseline, double fp64, float atol, float rtol);
std::string classification(float value);
// One repeat is insufficient evidence. Numerical repeat agreement uses the
// existing tolerance helper against the first execution, not the CPU baseline.
std::string determinism(const std::vector<Observation>& values, float atol, float rtol);
}
