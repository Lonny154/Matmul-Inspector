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

std::vector<double> solve_pcr(
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

    std::vector<double> a(n, 0.0);
    std::vector<double> b = diag;
    std::vector<double> c(n, 0.0);
    std::vector<double> d = rhs;

    for (std::size_t i = 1; i < n; ++i) {
        a[i] = lower[i - 1];
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        c[i] = upper[i];
    }

    // PCR
    std::vector<double> next_a(n);
    std::vector<double> next_b(n);
    std::vector<double> next_c(n);
    std::vector<double> next_d(n);

    for (std::size_t offset = 1; offset < n; offset *= 2) {

        for (std::size_t i = 0; i < n; ++i) {

            next_a[i] = 0.0;
            next_b[i] = b[i];
            next_c[i] = 0.0;
            next_d[i] = d[i];

            if (i >= offset) {
                const std::size_t left = i - offset;
                const double alpha = -a[i] / b[left];

                next_b[i] += alpha * c[left];
                next_d[i] += alpha * d[left];
                next_a[i] = alpha * a[left];

            }

            if (i + offset < n) {
                const std::size_t right = i + offset;
                const double beta = -c[i] / b[right];

                next_b[i] += beta * a[right];
                next_d[i] += beta * d[right];
                next_c[i] = beta * c[right];
            }
        }

        a.swap(next_a);
        b.swap(next_b);
        c.swap(next_c);
        d.swap(next_d);

    }
    
    std::vector<double> x(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = d[i] / b[i];
    }

    return x;
}



}  // namespace matmul_inspector