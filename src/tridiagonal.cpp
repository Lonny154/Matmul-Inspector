#include "tridiagonal.hpp"

#include <stdexcept>
#include <vector>

namespace matmul_inspector {

std::vector<double> solve_thomas(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs) {

    const std::size_t n = diag.size();

    if (n == 0) {
        throw std::invalid_argument("Diagonal must not be empty");
    }

    if (rhs.size() != n ||
        lower.size() != n - 1 ||
        upper.size() != n - 1) {
        throw std::invalid_argument("Invalid tridiagonal dimensions");
    }

    // Working copies because Thomas elimination modifies these values.
    std::vector<double> b = diag;
    std::vector<double> d = rhs;

    // Forward elimination.
    for (std::size_t i = 1; i < n; ++i) {
        if (b[i - 1] == 0.0) {
            throw std::runtime_error("Zero pivot in Thomas algorithm");
        }

        const double m = lower[i - 1] / b[i - 1];

        b[i] -= m * upper[i - 1];
        d[i] -= m * d[i - 1];
    }

    if (b[n - 1] == 0.0) {
        throw std::runtime_error("Zero pivot in Thomas algorithm");
    }

    // Back substitution.
    std::vector<double> x(n);

    x[n - 1] = d[n - 1] / b[n - 1];

    for (std::size_t i = n - 1; i-- > 0;) {
        if (b[i] == 0.0) {
            throw std::runtime_error("Zero pivot in Thomas algorithm");
        }

        x[i] = (d[i] - upper[i] * x[i + 1]) / b[i];
    }

    return x;
}

}  // namespace matmul_inspector