#pragma once

#include "matrix.hpp"

class Inspector {
public:
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

