#include "tridiagonal.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool approximately_equal(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

bool test_known_system() {
    const std::vector<double> lower{-1.0, -1.0, -1.0};
    const std::vector<double> diag{2.0, 2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0, -1.0};
    const std::vector<double> rhs{0.0, 0.0, 0.0, 5.0};

    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};

    const auto actual =
        matmul_inspector::solve_thomas(lower, diag, upper, rhs);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approximately_equal(actual[i], expected[i])) {
            std::cerr
                << "Known-system mismatch at index " << i
                << ": expected " << expected[i]
                << ", got " << actual[i] << '\n';
            return false;
        }
    }

    return true;
}

bool test_single_equation() {
    const std::vector<double> lower{};
    const std::vector<double> diag{4.0};
    const std::vector<double> upper{};
    const std::vector<double> rhs{12.0};

    const auto actual =
        matmul_inspector::solve_thomas(lower, diag, upper, rhs);

    if (actual.size() != 1 || !approximately_equal(actual[0], 3.0)) {
        std::cerr << "1x1 system failed\n";
        return false;
    }

    return true;
}

bool test_invalid_dimensions() {
    try {
        const std::vector<double> lower{-1.0};
        const std::vector<double> diag{2.0, 2.0, 2.0};
        const std::vector<double> upper{-1.0, -1.0};
        const std::vector<double> rhs{1.0, 2.0, 3.0};

        (void)matmul_inspector::solve_thomas(
            lower, diag, upper, rhs);
    } catch (const std::invalid_argument&) {
        return true;
    }

    std::cerr << "Invalid dimensions did not throw\n";
    return false;
}

bool test_zero_pivot() {
    try {
        const std::vector<double> lower{1.0};
        const std::vector<double> diag{0.0, 2.0};
        const std::vector<double> upper{1.0};
        const std::vector<double> rhs{1.0, 1.0};

        (void)matmul_inspector::solve_thomas(
            lower, diag, upper, rhs);
    } catch (const std::runtime_error&) {
        return true;
    }

    std::cerr << "Zero pivot did not throw\n";
    return false;
}

}  // namespace

int main() {
    if (!test_known_system()) {
        return 1;
    }

    if (!test_single_equation()) {
        return 1;
    }

    if (!test_invalid_dimensions()) {
        return 1;
    }

    if (!test_zero_pivot()) {
        return 1;
    }

    std::cout << "All Thomas solver tests passed\n";
    return 0;
}