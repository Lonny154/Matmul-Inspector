#include "comparison.hpp"
#include "numeric.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace comparison {

Result compare(const Matrix& reference, const Matrix& candidate, float atol, float rtol, std::size_t max_mismatches) {
    if (reference.rows() != candidate.rows() || reference.cols() != candidate.cols()) {
        throw std::invalid_argument("Cannot compare results with different dimensions");
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) {
        throw std::invalid_argument("Tolerances must be finite and nonnegative");
    }
    Result result;
    result.atol = atol;
    result.rtol = rtol;
    result.total_count = reference.rows() * reference.cols();
    long double ulp_sum = 0;
    for (std::size_t row = 0; row < reference.rows(); ++row) {
        for (std::size_t col = 0; col < reference.cols(); ++col) {
            Divergence point{row, col, reference(row, col), candidate(row, col)};
            const bool divergent = !numeric::bitwise_equal(point.reference, point.candidate);
            const bool passes = numeric::nearly_equal(point.candidate, point.reference, atol, rtol);
            if (divergent) {
                ++result.divergent_count;
                if (!result.first_bitwise) result.first_bitwise = point;
            }
            if (!passes) {
                ++result.tolerance_failures;
                result.tolerance_pass = false;
                if (!result.first_numeric) result.first_numeric = point;
            }
            if (divergent || !passes) {
                ++result.mismatch_count;
                if (result.mismatches.size() < max_mismatches) result.mismatches.push_back(point);
            }
            if (std::isnan(point.reference) || std::isnan(point.candidate)) {
                ++result.nan_pairs;
            } else if (!std::isfinite(point.reference) || !std::isfinite(point.candidate)) {
                ++result.infinity_pairs;
            } else {
                ++result.finite_pairs;
                const auto ulp = numeric::ulp_distance(point.reference, point.candidate);
                result.max_ulp = std::max(result.max_ulp, ulp);
                result.max_absolute_error = std::max(result.max_absolute_error,
                    static_cast<double>(numeric::absolute_error(point.candidate, point.reference)));
                result.max_relative_error = std::max(result.max_relative_error,
                    static_cast<double>(numeric::relative_error(point.candidate, point.reference)));
                if (point.reference == 0 && point.candidate != 0) ++result.zero_reference_nonzero;
                if (divergent) {
                    ++result.finite_divergent_count;
                    ulp_sum += ulp;
                    const unsigned bin = ulp <= 2 ? ulp : (ulp <= 4 ? 3 : (ulp <= 8 ? 4 : 5));
                    ++result.ulp_bins[bin];
                }
            }
        }
    }
    if (result.finite_divergent_count) result.mean_divergent_ulp = static_cast<double>(ulp_sum / result.finite_divergent_count);
    if (result.total_count) {
        result.divergent_percent = 100.0 * result.divergent_count / result.total_count;
        result.tolerance_failure_percent = 100.0 * result.tolerance_failures / result.total_count;
    }
    result.mismatches_truncated = result.mismatch_count > result.mismatches.size();
    return result;
}

}  // namespace comparison
