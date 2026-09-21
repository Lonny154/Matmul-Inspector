#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);
    // Row stride (leading dimension) is measured in floats, not bytes, and
    // must be at least cols. Storage includes padding after every row.
    Matrix(std::size_t rows, std::size_t cols, std::size_t row_stride);

    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t row_stride() const;

    float& operator()(std::size_t row, std::size_t col);
    const float& operator()(std::size_t row, std::size_t col) const;

    // Raw storage, including padding. Element (row, col) is at
    // data() + row * row_stride() + col; logical indexing is unchecked.
    float* data();
    const float* data() const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t row_stride_;
    std::vector<float> data_;
};

Matrix matmul(const Matrix& a, const Matrix& b);
// Select the output row stride; the two-argument overload returns packed rows.
Matrix matmul(const Matrix& a, const Matrix& b, std::size_t row_stride);
