#include "inspector.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <bitset>
#include <cstdint>

static std::bitset<32> float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    return std::bitset<32>(bits);
}

static std::uint32_t float_hex(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

static std::uint32_t float_to_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

static std::uint32_t ulp_distance(float a, float b) {
    std::uint32_t a_bits = float_to_bits(a);
    std::uint32_t b_bits = float_to_bits(b);

    return a_bits > b_bits
        ? a_bits - b_bits
        : b_bits - a_bits;
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

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t col = 0; col < c.cols(); ++col) {
            float expected = 0.0f;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                expected += a(row, k) * b(k, col);
            }

            float actual = c(row, col);

            float difference = std::fabs(actual - expected);
            float tolerance = 1e-5f;

            if (difference > tolerance) {
                std::cout << std::setprecision(10);
                std::cout << "FIRST MISMATCH\n";
                std::cout << "--------------\n";
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

    std::cout << "No mismatches found\n";
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
        const float* a_ptr = a.data() + (row * a.cols() + k);
        const float* b_ptr = b.data() + (k * b.cols() + col);

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

    std::cout << "\nresult = " << sum << '\n';

    const float* c_ptr = c.data() + (row * c.cols() + col);

    std::cout
        << "\nC[" << row << "," << col << "] "
        << static_cast<const void*>(c_ptr)
        << " = " << *c_ptr
        << '\n';

    std::cout << "recomputed result = " << sum << '\n';

    std::cout << "actual bits:     " << float_bits(*c_ptr) << '\n';
    std::cout << "expected bits:   " << float_bits(sum) << '\n';

    std::cout
    << "actual hex:      0x"
    << std::hex
    << std::setw(8)
    << std::setfill('0')
    << float_hex(*c_ptr)
    << '\n';

    std::cout
        << "expected hex:    0x"
        << std::setw(8)
        << float_hex(sum)
        << std::dec
        << std::setfill(' ')
        << '\n';

    std::cout << "ULP distance:     "
          << ulp_distance(*c_ptr, sum)
          << '\n';

    float difference = std::fabs(*c_ptr - sum);
    float tolerance = 1e-5f;

    bool numeric_match = difference <= tolerance;
    bool bitwise_match = std::memcmp(c_ptr, &sum, sizeof(float)) == 0;

    std::cout << "numeric: "
            << (numeric_match ? "MATCH" : "MISMATCH")
            << '\n';

    std::cout << "bitwise: "
            << (bitwise_match ? "MATCH" : "MISMATCH")
            << '\n';

    std::cout << "difference = " << difference << '\n';
    
    }

void Inspector::dump_memory(const Matrix& matrix) {
    const float* ptr = matrix.data();

    std::cout << "Memory dump\n";
    std::cout << "-----------\n";

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            std::size_t index = row * matrix.cols() + col;

            std::cout
                << "[" << row << "," << col << "]  "
                << static_cast<const void*>(ptr + index)
                << "  "
                << std::fixed
                << std::setprecision(3)
                << *(ptr + index)
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

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t col = 0; col < c.cols(); ++col) {
            float expected = 0.0f;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                expected += a(row, k) * b(k, col);
            }

            float actual = c(row, col);

            bool bitwise_match =
                std::memcmp(&actual, &expected, sizeof(float)) == 0;

            if (!bitwise_match) {
                std::cout << std::setprecision(10);
                std::cout << "FIRST BITWISE MISMATCH\n";
                std::cout << "----------------------\n";
                std::cout << "C[" << row << "," << col << "]\n";
                std::cout << "actual:   " << actual << '\n';
                std::cout << "expected: " << expected << '\n';
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