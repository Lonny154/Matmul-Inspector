#include <iostream>
#include <exception>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "inspector.hpp"
#include "matrix.hpp"
#include "experiment.hpp"

#ifdef MATMUL_INSPECTOR_HAS_CUDA
#include "cuda_matmul.hpp"
#endif

namespace {

void print_matrix(const char* label, const Matrix& matrix) {
    std::cout << label << '\n';
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            std::cout << matrix(row, col) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) return experiment::run_cli(std::vector<std::string>(argv + 1, argv + argc));

    Matrix a(2, 3);
    Matrix b(3, 2);

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;

    b(0, 0) = 7.0f;
    b(0, 1) = 8.0f;
    b(1, 0) = 9.0f;
    b(1, 1) = 10.0f;
    b(2, 0) = 11.0f;
    b(2, 1) = 12.0f;

    Matrix cpu = matmul(a, b);
    print_matrix("CPU C", cpu);

#ifdef MATMUL_INSPECTOR_HAS_CUDA
    std::string reason;
    if (!cuda_available(&reason)) {
        std::cout << "GPU comparison skipped: " << reason << '\n';
        return 0;
    }
    try {
        Matrix gpu = cuda_matmul(a, b);
        print_matrix("GPU C", gpu);
        std::cout << "CPU vs GPU (expected: CPU, actual: GPU)\n";
        Inspector::compare_results(cpu, gpu);
    } catch (const std::exception& error) {
        std::cerr << "GPU matmul failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "GPU comparison skipped: CUDA support was not built\n";
#endif

    return 0;
}
