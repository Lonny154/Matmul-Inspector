#include "matrix.hpp"
#include <limits>
#include <stdexcept>

namespace {

std::size_t storage_size(std::size_t rows, std::size_t cols, std::size_t row_stride) {
    if (row_stride < cols) {
        throw std::invalid_argument("Row stride must be at least the column count");
    }
    if (row_stride != 0 && rows > std::numeric_limits<std::size_t>::max() / row_stride) {
        throw std::length_error("Matrix storage size overflow");
    }
    return rows * row_stride;
}

}  // namespace

Matrix matmul(const Matrix& a, const Matrix& b) {
    return matmul(a, b, b.cols());
}

Matrix matmul(const Matrix& a, const Matrix& b, std::size_t row_stride) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(a.rows(), b.cols(), row_stride);

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
    : Matrix(rows, cols, cols) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::size_t row_stride)
    : rows_(rows),
      cols_(cols),
      row_stride_(row_stride),
      data_(storage_size(rows, cols, row_stride), 0.0f) {}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

std::size_t Matrix::row_stride() const {
    return row_stride_;
}

float& Matrix::operator()(std::size_t row, std::size_t col) {
    return data_[row * row_stride_ + col];
}

const float& Matrix::operator()(std::size_t row, std::size_t col) const {
    return data_[row * row_stride_ + col];
}

float* Matrix::data() {
    return data_.data();
}

const float* Matrix::data() const {
    return data_.data();
}
