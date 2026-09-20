#include "matrix.hpp"
#include <stdexcept>

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(a.rows(), b.cols());

    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t col = 0; col < b.cols(); ++col) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(row, k) * b(k, col);
            }

            result(row, col) = sum;
        }
    }

    return result;
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, 0.0f) {}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

float& Matrix::operator()(std::size_t row, std::size_t col) {
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(std::size_t row, std::size_t col) const {
    return data_[row * cols_ + col];
}

float* Matrix::data() {
    return data_.data();
}

const float* Matrix::data() const {
    return data_.data();
}