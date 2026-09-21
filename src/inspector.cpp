#include "inspector.hpp"
#include "numeric.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

void Inspector::find_first_mismatch(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c
) {
    if (a.cols() != b.rows()) {
        std::cerr << "Incompatible matrix dimensions\n";
        return;
    }

    if (c.rows() != a.rows() || c.cols() != b.cols()) {
        std::cerr << "Output matrix has incorrect dimensions\n";
        return;
    }

    constexpr float abs_tolerance = 1e-6f;
    constexpr float rel_tolerance = 1e-5f;

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t col = 0; col < c.cols(); ++col) {
            float expected = 0.0f;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                expected += a(row, k) * b(k, col);
            }

            float actual = c(row, col);

            if (!numeric::nearly_equal(
                    actual,
                    expected,
                    abs_tolerance,
                    rel_tolerance
                )) {
                float difference = std::fabs(actual - expected);

                float tolerance = numeric::comparison_tolerance(
                    actual,
                    expected,
                    abs_tolerance,
                    rel_tolerance
                );

                std::cout << std::setprecision(10);
                std::cout << "FIRST NUMERIC MISMATCH\n";
                std::cout << "----------------------\n";
                std::cout << "C[" << row << "," << col << "]\n";
                std::cout << "actual:     " << actual << '\n';
                std::cout << "expected:   " << expected << '\n';
                std::cout << "difference: " << difference << '\n';
                std::cout << "tolerance:  " << tolerance << "\n\n";

                trace_matmul(a, b, c, row, col);
                return;
            }
        }
    }

    std::cout << "No numeric mismatches found\n";
}

void Inspector::trace_matmul(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c,
    std::size_t row,
    std::size_t col
) {
    if (a.cols() != b.rows()) {
        std::cerr << "Incompatible matrix dimensions\n";
        return;
    }

    if (row >= a.rows() || col >= b.cols()) {
        std::cerr << "Output coordinate out of bounds\n";
        return;
    }

    float sum = 0.0f;

    std::cout << "Trace C[" << row << "," << col << "]\n";
    std::cout << "----------------\n";

    for (std::size_t k = 0; k < a.cols(); ++k) {
        const float* a_ptr = &a(row, k);
        const float* b_ptr = &b(k, col);

        float product = (*a_ptr) * (*b_ptr);
        sum += product;

        std::cout
            << "A[" << row << "," << k << "] "
            << static_cast<const void*>(a_ptr)
            << " = " << *a_ptr
            << "  *  "
            << "B[" << k << "," << col << "] "
            << static_cast<const void*>(b_ptr)
            << " = " << *b_ptr
            << "  ->  "
            << product
            << '\n';
    }

    const float* c_ptr = &c(row, col);

    float difference = std::fabs(*c_ptr - sum);

    constexpr float abs_tolerance = 1e-6f;
    constexpr float rel_tolerance = 1e-5f;

    bool numeric_match = numeric::nearly_equal(
        *c_ptr,
        sum,
        abs_tolerance,
        rel_tolerance
    );

    bool bitwise_match =
        numeric::float_to_bits(*c_ptr) == numeric::float_to_bits(sum);

    float relative_error = 0.0f;

    if (sum != 0.0f) {
        relative_error = difference / std::fabs(sum);
    }

    std::cout << std::setprecision(10);

    std::cout << "\nresult = " << sum << '\n';

    std::cout
        << "\nC[" << row << "," << col << "] "
        << static_cast<const void*>(c_ptr)
        << " = " << *c_ptr
        << '\n';

    std::cout << "recomputed result = " << sum << '\n';

    std::cout << "actual bits:     "
              << numeric::float_bits(*c_ptr)
              << '\n';

    std::cout << "expected bits:   "
              << numeric::float_bits(sum)
              << '\n';

    std::cout
        << "actual hex:      0x"
        << std::hex
        << std::setw(8)
        << std::setfill('0')
        << numeric::float_to_bits(*c_ptr)
        << '\n';

    std::cout
        << "expected hex:    0x"
        << std::setw(8)
        << numeric::float_to_bits(sum)
        << std::dec
        << std::setfill(' ')
        << '\n';

    std::cout << "ULP distance:    "
              << numeric::ulp_distance(*c_ptr, sum)
              << '\n';

    std::cout << "absolute error:  "
              << difference
              << '\n';

    std::cout << "relative error:  "
              << relative_error
              << '\n';

    std::cout << "numeric:         "
              << (numeric_match ? "MATCH" : "MISMATCH")
              << '\n';

    std::cout << "bitwise:         "
              << (bitwise_match ? "MATCH" : "MISMATCH")
              << '\n';
}

void Inspector::dump_memory(const Matrix& matrix) {
    std::cout << "Memory dump\n";
    std::cout << "-----------\n";

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            const float* ptr = &matrix(row, col);

            std::cout
                << "[" << row << "," << col << "]  "
                << static_cast<const void*>(ptr)
                << "  "
                << std::fixed
                << std::setprecision(3)
                << *ptr
                << '\n';
        }
    }
}

void Inspector::find_first_bitwise_mismatch(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c
) {
    if (a.cols() != b.rows()) {
        std::cerr << "Incompatible matrix dimensions\n";
        return;
    }

    if (c.rows() != a.rows() || c.cols() != b.cols()) {
        std::cerr << "Output matrix has incorrect dimensions\n";
        return;
    }

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t col = 0; col < c.cols(); ++col) {
            float expected = 0.0f;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                expected += a(row, k) * b(k, col);
            }

            float actual = c(row, col);

            bool bitwise_match =
                numeric::float_to_bits(actual) == numeric::float_to_bits(expected);

            if (!bitwise_match) {
                std::cout << std::setprecision(10);
                std::cout << "FIRST BITWISE MISMATCH\n";
                std::cout << "----------------------\n";
                std::cout << "C[" << row << "," << col << "]\n";
                std::cout << "actual:     " << actual << '\n';
                std::cout << "expected:   " << expected << '\n';
                std::cout << "difference: "
                          << std::fabs(actual - expected)
                          << "\n\n";

                trace_matmul(a, b, c, row, col);
                return;
            }
        }
    }

    std::cout << "No bitwise mismatches found\n";
}
