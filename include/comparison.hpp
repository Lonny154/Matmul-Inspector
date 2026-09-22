#pragma once

#include "matrix.hpp"
#include <optional>
#include <array>
#include <cstdint>
#include <vector>

namespace comparison {

struct Divergence {
    std::size_t row, col;
    float reference, candidate;
};

struct Result {
    std::size_t divergent_count = 0;
    bool tolerance_pass = true;
    float atol = 1e-6f, rtol = 1e-5f;
    std::optional<Divergence> first_bitwise;
    std::optional<Divergence> first_numeric;
    std::size_t total_count = 0, tolerance_failures = 0;
    // Error/ULP aggregates exclude pairs containing NaN or infinity. Errors use
    // the numeric helpers (FP32 arithmetic, relative error zero for zero reference).
    std::size_t finite_pairs = 0, finite_divergent_count = 0;
    std::size_t nan_pairs = 0, infinity_pairs = 0, zero_reference_nonzero = 0;
    std::uint32_t max_ulp = 0;
    double mean_divergent_ulp = 0, max_absolute_error = 0, max_relative_error = 0;
    double divergent_percent = 0, tolerance_failure_percent = 0;
    // Bitwise-divergent finite pairs: 0 (signed zeros), 1, 2, 3-4, 5-8, >8.
    std::array<std::size_t, 6> ulp_bins{};
    std::size_t mismatch_count = 0;
    std::vector<Divergence> mismatches;
    bool mismatches_truncated = false;
};

Result compare(const Matrix& reference, const Matrix& candidate,
               float atol = 1e-6f, float rtol = 1e-5f, std::size_t max_mismatches = 100);

}  // namespace comparison
