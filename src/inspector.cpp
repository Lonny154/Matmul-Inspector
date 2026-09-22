#include "inspector.hpp"
#include "numeric.hpp"

#include <iomanip>
#include <iostream>

void Inspector::compare_results(const Matrix& expected, const Matrix& actual, float atol, float rtol) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        std::cerr << "Cannot compare results with different dimensions\n";
        return;
    }

    report_comparison(comparison::compare(expected, actual, atol, rtol));
}

void Inspector::report_comparison(const comparison::Result& result) {
    const bool numeric_match = result.tolerance_pass;
    const bool bitwise_match = result.divergent_count == 0;
    const float abs_tolerance = result.atol, rel_tolerance = result.rtol;
    std::cout << "numeric: " << (numeric_match ? "MATCH" : "MISMATCH") << '\n';
    std::cout << "bitwise: " << (bitwise_match ? "MATCH" : "MISMATCH") << '\n';
    std::cout << "bitwise equal: " << (bitwise_match ? "yes" : "no") << '\n';
    std::cout << "divergent elements: " << result.divergent_count << '\n';
    std::cout << "numerical tolerance: " << (numeric_match ? "PASS" : "FAIL") << '\n';
    std::cout << std::setprecision(10)
              << "bitwise divergent percent: " << result.divergent_percent << '\n'
              << "tolerance failures: " << result.tolerance_failures << " (" << result.tolerance_failure_percent << "%)\n"
              << "max ULP (finite pairs): " << result.max_ulp << '\n'
              << "mean ULP (divergent finite pairs): " << result.mean_divergent_ulp << '\n'
              << "max absolute error (finite pairs): " << result.max_absolute_error << '\n'
              << "max relative error (finite pairs): " << result.max_relative_error << '\n'
              << "ULP bins [0,1,2,3-4,5-8,>8]: ";
    for (auto count : result.ulp_bins) std::cout << count << ' ';
    std::cout << "\nNaN pairs: " << result.nan_pairs << "; infinity pairs: " << result.infinity_pairs
              << "; zero-reference/nonzero-candidate pairs: " << result.zero_reference_nonzero << '\n';
    auto report = [&](const char* heading, const comparison::Divergence& point) {
        const auto row = point.row, col = point.col;
        float a = point.candidate, e = point.reference;
        std::cout << std::setprecision(10)
                  << heading << "\nC[" << row << ',' << col << "]\n"
                  << "actual:         " << a << '\n'
                  << "expected:       " << e << '\n'
                  << "actual bits:    " << numeric::float_bits(a) << '\n'
                  << "expected bits:  " << numeric::float_bits(e) << '\n'
                  << "ULP distance:   " << numeric::ulp_distance(a, e) << '\n'
                  << "absolute error: " << numeric::absolute_error(a, e) << '\n'
                  << "relative error: " << numeric::relative_error(a, e) << '\n'
                  << "tolerance:      " << numeric::comparison_tolerance(a, e, abs_tolerance, rel_tolerance)
                  << '\n'
                  << "element tolerance: "
                  << (numeric::nearly_equal(a, e, abs_tolerance, rel_tolerance) ? "PASS" : "FAIL") << '\n';
    };
    if (numeric_match) {
        std::cout << "No numeric mismatches found\n";
    } else {
        report("FIRST NUMERIC MISMATCH", *result.first_numeric);
    }
    if (bitwise_match) {
        std::cout << "No bitwise mismatches found\n";
    } else {
        report("FIRST BITWISE MISMATCH", *result.first_bitwise);
        if (numeric_match) {
            std::cout << "Bitwise differences are within numerical tolerance.\n";
        }
    }
}

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
                float difference = numeric::absolute_error(actual, expected);

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

    float difference = numeric::absolute_error(*c_ptr, sum);

    constexpr float abs_tolerance = 1e-6f;
    constexpr float rel_tolerance = 1e-5f;

    bool numeric_match = numeric::nearly_equal(
        *c_ptr,
        sum,
        abs_tolerance,
        rel_tolerance
    );

    bool bitwise_match = numeric::bitwise_equal(*c_ptr, sum);
    float relative_error = numeric::relative_error(*c_ptr, sum);

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

            bool bitwise_match = numeric::bitwise_equal(actual, expected);

            if (!bitwise_match) {
                std::cout << std::setprecision(10);
                std::cout << "FIRST BITWISE MISMATCH\n";
                std::cout << "----------------------\n";
                std::cout << "C[" << row << "," << col << "]\n";
                std::cout << "actual:     " << actual << '\n';
                std::cout << "expected:   " << expected << '\n';
                std::cout << "difference: "
                          << numeric::absolute_error(actual, expected)
                          << "\n\n";

                trace_matmul(a, b, c, row, col);
                return;
            }
        }
    }

    std::cout << "No bitwise mismatches found\n";
}
