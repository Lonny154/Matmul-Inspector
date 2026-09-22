#include "experiment.hpp"
#include "cuda_matmul.hpp"
#include "numeric.hpp"

#include <iostream>
#include <cmath>
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
    for (const Shape shape : {Shape{1, 1, 1}, Shape{3, 5, 2}, Shape{17, 33, 19}, Shape{4, 31, 6}}) {
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
                    Matrix tiled = cuda_matmul(a, b, shape.cols + c_padding, CudaMatmulKernel::tiled);
                    check(tiled.row_stride() == shape.cols + c_padding, "tiled output row stride");
                    check_agreement(cpu, tiled);
                    check_agreement(gpu, tiled);
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
    for (auto kernel : {CudaMatmulKernel::naive_fma, CudaMatmulKernel::naive_no_fma, CudaMatmulKernel::naive_reordered}) {
        check_invalid([kernel] { cuda_matmul(Matrix(2,3), Matrix(2,4), kernel); }, "controlled invalid dimensions");
        check_invalid([kernel] { cuda_matmul(Matrix(2,3), Matrix(3,4), 3, kernel); }, "controlled short stride");
        check_invalid([kernel] { benchmark_cuda_kernel(Matrix(1,1), Matrix(1,1), kernel, 0); }, "controlled zero iterations");
        check_invalid([kernel] { benchmark_cuda_kernel(Matrix(1,1), Matrix(1,1), kernel, 1, -1); }, "controlled negative warmups");
        auto empty = benchmark_cuda_kernel(Matrix(2,0), Matrix(0,4), kernel, 1);
        check(empty.output.rows() == 2 && empty.output.cols() == 4 && empty.timing.mean_ms == 0, "controlled empty product");
    }
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(2, 4)); }, "reject incompatible dimensions");
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(3, 4), 3); }, "reject short output stride");
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(3, 4), 0); }, "reject zero output stride");
    Matrix zero = cuda_matmul(Matrix(2, 0, 3), Matrix(0, 4, 6), 7);
    check_agreement(matmul(Matrix(2, 0), Matrix(0, 4)), zero);
    Matrix no_rows = cuda_matmul(Matrix(0, 3, 4), Matrix(3, 2, 5), 6);
    check(no_rows.rows() == 0 && no_rows.cols() == 2 && no_rows.row_stride() == 6, "zero output rows");
    Matrix no_cols = cuda_matmul(Matrix(2, 3, 4), Matrix(3, 0, 2));
    check(no_cols.rows() == 2 && no_cols.cols() == 0 && no_cols.row_stride() == 0, "zero output columns");

    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(2, 4), CudaMatmulKernel::tiled); },
                  "tiled rejects incompatible dimensions");
    check_invalid([] { cuda_matmul(Matrix(2, 3), Matrix(3, 4), 3, CudaMatmulKernel::tiled); },
                  "tiled rejects short output stride");
    check_agreement(zero, cuda_matmul(Matrix(2, 0, 3), Matrix(0, 4, 6), 7, CudaMatmulKernel::tiled));
    check(cuda_matmul(Matrix(0, 3), Matrix(3, 2), CudaMatmulKernel::tiled).rows() == 0,
          "tiled zero rows");
    check(cuda_matmul(Matrix(2, 3), Matrix(3, 0), CudaMatmulKernel::tiled).cols() == 0,
          "tiled zero columns");
    check_invalid([] { benchmark_cuda_matmul(Matrix(1, 1), Matrix(1, 1), 0); }, "reject zero repetitions");
    check_invalid([] { benchmark_cuda_matmul(Matrix(1, 1), Matrix(1, 1), -1); }, "reject negative repetitions");
    check_invalid([] { benchmark_cuda_matmul(Matrix(1, 1), Matrix(1, 1), 1, -1); }, "reject negative warmups");
    auto empty = benchmark_cuda_matmul(Matrix(2, 0), Matrix(0, 4), 1);
    check(empty.naive_ms == 0.0f && empty.tiled_ms == 0.0f, "empty products have no kernel time");
    check_agreement(empty.naive, empty.tiled);
}

void test_tiled_sizes_and_timing() {
    for (std::size_t size : {4u, 256u, 257u, 1024u}) {
        Matrix a(size, size), b(size, size);
        fill(a, 42);
        fill(b, 123);
        Matrix naive = cuda_matmul(a, b);
        Matrix tiled = cuda_matmul(a, b, CudaMatmulKernel::tiled);
        // Numerical agreement is required; bitwise agreement is measured, not assumed.
        check_agreement(naive, tiled);
        std::size_t divergent = 0;
        for (std::size_t row = 0; row < size; ++row) {
            for (std::size_t col = 0; col < size; ++col) {
                divergent += !numeric::bitwise_equal(naive(row, col), tiled(row, col));
            }
        }
        std::cout << "Size " << size << ": " << divergent << " bitwise divergent elements\n";
        if (size == 4) {
            check_agreement(matmul(a, b), tiled);
            auto measured = benchmark_cuda_matmul(a, b, 2, 0);
            check_agreement(naive, measured.naive);
            check_agreement(tiled, measured.tiled);
            check(std::isfinite(measured.naive_ms) && measured.naive_ms > 0.0f, "valid naive GPU timing");
            check(std::isfinite(measured.tiled_ms) && measured.tiled_ms > 0.0f, "valid tiled GPU timing");
            check(measured.naive_stats.min_ms <= measured.naive_stats.mean_ms && measured.naive_stats.stddev_ms >= 0,
                  "naive timing statistics");
            check(measured.tiled_stats.min_ms <= measured.tiled_stats.median_ms && measured.tiled_stats.stddev_ms >= 0,
                  "tiled timing statistics");
        }
    }
}

void test_controlled() {
    for (std::size_t k : {4u, 16u, 17u, 257u}) {
        Matrix a(3,k,k+2), b(k,5,8);
        experiment::Config config;
        experiment::fill_inputs(a,b,config);
        for (auto kernel : {CudaMatmulKernel::naive_fma, CudaMatmulKernel::naive_no_fma, CudaMatmulKernel::naive_reordered}) {
            auto result = cuda_matmul(a,b,9,kernel);
            auto repeat = cuda_matmul(a,b,kernel);
            check(result.rows() == 3 && result.cols() == 5 && result.row_stride() == 9, "controlled dimensions and stride");
            check(comparison::compare(result,repeat).divergent_count == 0, "controlled repeat deterministic");
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t col = 5; col < 9; ++col) check(result.data()[row*9+col] == 0, "controlled output padding");
                for (std::size_t col = 0; col < 5; ++col) {
                    float expected = 0, odd = 0;
                    for (std::size_t term = 0; term < k; ++term) {
                        if (kernel == CudaMatmulKernel::naive_no_fma) {
                            volatile float product = a(row,term) * b(term,col);
                            expected = expected + product;
                        } else if (kernel == CudaMatmulKernel::naive_reordered && term % 2)
                            odd = std::fma(a(row,term), b(term,col), odd);
                        else expected = std::fma(a(row,term), b(term,col), expected);
                    }
                    if (kernel == CudaMatmulKernel::naive_reordered) expected += odd;
                    check(numeric::bitwise_equal(expected,result(row,col)), "controlled arithmetic matches explicit host model");
                }
            }
        }
    }
    Matrix a(4,4), b(4,4);
    experiment::Config c; c.input = "cancellation";
    experiment::fill_inputs(a,b,c);
    auto fused = cuda_matmul(a,b,CudaMatmulKernel::naive_fma);
    auto reordered = cuda_matmul(a,b,CudaMatmulKernel::naive_reordered);
    check(fused(0,0) == 1 && reordered(0,0) == 2, "cancellation changes association only");
    c.input = "fma-sensitive"; experiment::fill_inputs(a,b,c);
    fused = cuda_matmul(a,b,CudaMatmulKernel::naive_fma);
    auto separate = cuda_matmul(a,b,CudaMatmulKernel::naive_no_fma);
    check(fused(0,0) == -0x1p-46f && separate(0,0) == 0, "FMA single rounding residual");
    auto timed = benchmark_cuda_kernel(a,b,CudaMatmulKernel::naive_reordered,2,1);
    check(timed.timing.mean_ms > 0 && timed.timing.stddev_ms >= 0, "controlled event timing");
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
        test_controlled();
        test_example();
        test_layouts();
        test_tiled_sizes_and_timing();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
