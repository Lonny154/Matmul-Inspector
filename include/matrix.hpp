#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);

    std::size_t rows() const;
    std::size_t cols() const;

    float& operator()(std::size_t row, std::size_t col);
    const float& operator()(std::size_t row, std::size_t col) const;

    float* data();
    const float* data() const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};

Matrix matmul(const Matrix& a, const Matrix& b);