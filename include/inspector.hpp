#pragma once

#include "matrix.hpp"

class Inspector {
public:
    // Compare logical output elements directly, independent of their strides.
    // Reports numerical agreement and bitwise agreement separately, including
    // diagnostics for the first mismatch of each kind. Expected is the reference.
    static void compare_results(const Matrix& expected, const Matrix& actual);

    static void dump_memory(const Matrix& matrix);

    static void trace_matmul(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c,
    std::size_t row,
    std::size_t col);

    static void find_first_mismatch(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c);

    static void find_first_bitwise_mismatch(
    const Matrix& a,
    const Matrix& b,
    const Matrix& c
    );
};
