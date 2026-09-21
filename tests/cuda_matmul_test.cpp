#include "cuda_matmul.hpp"
#include "numeric.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

template <typename Function>
void check_invalid(Function function, const char* description) {
    try {
        function();
        check(false, description);
    } catch (const std::invalid_argument&) {
    }
}

void fill(Matrix& matrix, unsigned seed) {
    // NaN padding exposes any accidental use of padding in the dot products.
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.row_stride(); ++col) {
            matrix.data()[row * matrix.row_stride() + col] = std::numeric_limits<float>::quiet_NaN();
        }
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            seed = 1664525u * seed + 1013904223u;
            // Non-binary fractions, with signs, exercise floating-point rounding.
            matrix(row, col) = static_cast<float>(static_cast<int>((seed >> 16) % 101) - 50) / 100.0f;
        }
    }
}

void check_agreement(const Matrix& cpu, const Matrix& gpu) {
    check(cpu.rows() == gpu.rows() && cpu.cols() == gpu.cols(), "CPU/GPU result dimensions");
    for (std::size_t row = 0; row < cpu.rows(); ++row) {
        for (std::size_t col = 0; col < cpu.cols(); ++col) {
            check(numeric::nearly_equal(gpu(row, col), cpu(row, col), 1e-6f, 1e-5f),
                  "CPU/GPU numerical agreement");
        }
        for (std::size_t col = gpu.cols(); col < gpu.row_stride(); ++col) {
            check(gpu.data()[row * gpu.row_stride() + col] == 0.0f, "GPU output padding is initialized");
        }
    }
}

void test_example() {
    Matrix a(2, 3), b(3, 2);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            a(row, col) = static_cast<float>(row * 3 + col + 1);
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 2; ++col) {
            b(row, col) = static_cast<float>(row * 2 + col + 7);
        }
    }
    Matrix gpu = cuda_matmul(a, b);
    check(gpu.row_stride() == 2, "default GPU output is contiguous");
    check(gpu(0, 0) == 58.0f && gpu(0, 1) == 64.0f
          && gpu(1, 0) == 139.0f && gpu(1, 1) == 154.0f, "small known GPU product");
    check_agreement(matmul(a, b), gpu);
}

void test_layouts() {
    struct Shape { std::size_t rows, inner, cols; };
    // 17 x 19 gives multiple blocks with an incomplete final block.
    for (const Shape shape : {Shape{1, 1, 1}, Shape{3, 5, 2}, Shape{17, 7, 19}, Shape{4, 31, 6}}) {
        for (std::size_t a_padding : {0u, 3u}) {
            for (std::size_t b_padding : {0u, 5u}) {
                Matrix a(shape.rows, shape.inner, shape.inner + a_padding);
                Matrix b(shape.inner, shape.cols, shape.cols + b_padding);
                fill(a, 42);
                fill(b, 123);
                Matrix a_before = a, b_before = b;
                Matrix cpu = matmul(a, b, shape.cols + 1);
                for (std::size_t c_padding : {0u, 7u}) {
                    Matrix gpu = c_padding == 0 ? cuda_matmul(a, b)
                        : cuda_matmul(a, b, shape.cols + c_padding);
                    check(gpu.row_stride() == shape.cols + c_padding, "GPU output row stride");
                    check_agreement(cpu, gpu);
                }
                for (std::size_t i = 0; i < a.rows() * a.row_stride(); ++i) {
                    check(numeric::bitwise_equal(a.data()[i], a_before.data()[i]), "A storage unchanged");
                }
                for (std::size_t i = 0; i < b.rows() * b.row_stride(); ++i) {
                    check(numeric::bitwise_equal(b.data()[i], b_before.data()[i]), "B storage unchanged");
                }
            }
        }
    }
}

void test_validation_and_empty_products() {
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(2, 4)); }, "reject incompatible dimensions");
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(3, 4), 3); }, "reject short output stride");
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(3, 4), 0); }, "reject zero output stride");
    Matrix zero = cuda_matmul(Matrix(2, 0, 3), Matrix(0, 4, 6), 7);
    check_agreement(matmul(Matrix(2, 0), Matrix(0, 4)), zero);
    Matrix no_rows = cuda_matmul(Matrix(0, 3, 4), Matrix(3, 2, 5), 6);
    check(no_rows.rows() == 0 && no_rows.cols() == 2 && no_rows.row_stride() == 6, "zero output rows");
    Matrix no_cols = cuda_matmul(Matrix(2, 3, 4), Matrix(3, 0, 2));
    check(no_cols.rows() == 2 && no_cols.cols() == 0 && no_cols.row_stride() == 0, "zero output columns");
}

}  // namespace

int main() {
    try {
        // Argument validation and empty products need no runtime or device.
        test_validation_and_empty_products();
        if (failures != 0) {
            return 1;
        }
        std::string reason;
        if (!cuda_available(&reason)) {
            std::cout << "SKIP: " << reason << '\n';
            // A real, nonempty operation must report runtime failure to callers.
            try {
                cuda_matmul(Matrix(1, 1), Matrix(1, 1));
                std::cerr << "FAIL: unavailable CUDA did not throw\n";
                return 1;
            } catch (const std::runtime_error&) {
            }
            return 77;
        }
        test_example();
        test_layouts();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
