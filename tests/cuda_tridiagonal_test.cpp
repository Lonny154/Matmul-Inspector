#include "cuda_tridiagonal.hpp"
#include "cuda_tridiagonal_benchmark.hpp"
#include "tridiagonal.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<double> apply_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& x) {

    const std::size_t n = diag.size();
    std::vector<double> rhs(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] += diag[i] * x[i];

        if (i > 0) {
            rhs[i] += lower[i - 1] * x[i - 1];
        }

        if (i + 1 < n) {
            rhs[i] += upper[i] * x[i + 1];
        }
    }

    return rhs;
}

bool test_size(std::size_t n) {
    std::vector<double> lower(n > 0 ? n - 1 : 0, -1.0);
    std::vector<double> diag(n, 4.0);
    std::vector<double> upper(n > 0 ? n - 1 : 0, -1.0);

    std::vector<double> expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = static_cast<double>(i + 1);
    }

    const auto rhs =
        apply_tridiagonal(lower, diag, upper, expected);

    const auto cpu =
        matmul_inspector::solve_pcr(
            lower, diag, upper, rhs);

    const auto gpu =
        matmul_inspector::solve_pcr_cuda(
            lower, diag, upper, rhs);

    constexpr double tolerance = 1e-10;

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(gpu[i] - expected[i]) > tolerance) {
            std::cerr
                << "CUDA PCR failed for n=" << n
                << " at index " << i
                << ": expected " << expected[i]
                << ", got " << gpu[i] << '\n';
            return false;
        }

        if (std::abs(gpu[i] - cpu[i]) > tolerance) {
            std::cerr
                << "CPU/CUDA PCR mismatch for n=" << n
                << " at index " << i
                << ": CPU " << cpu[i]
                << ", CUDA " << gpu[i] << '\n';
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    const std::vector<std::size_t> sizes{
        1, 2, 3, 4,
        7, 8,
        15, 16,
        31, 32,
        63, 64,
        100,
        127, 128,
        255, 256, 257,
        511, 512,
        1000
    };

    for (const auto n : sizes) {
        if (!test_size(n)) {
            return 1;
        }
    }

    const auto systems=matmul_inspector::make_benchmark_systems(33,3);
    const auto kernel=matmul_inspector::benchmark_cuda_pcr_kernel_only(systems,3,1);
    const auto end_to_end=matmul_inspector::benchmark_cuda_pcr_end_to_end(systems,2,1);
    if (kernel.backend!="cuda" || kernel.algorithm!="pcr" ||
        kernel.timing_scope!="kernel_only" || kernel.system_size!=33 || kernel.batch_size!=3 ||
        kernel.batch_execution!="serial_host_loop" ||
        kernel.timing.samples.size()!=4 || kernel.timing.median_ms<=0 ||
        end_to_end.timing_scope!="end_to_end" || end_to_end.timing.samples.size()!=3 ||
        end_to_end.timing.median_ms<=0) {
        std::cerr << "CUDA PCR benchmark timing/metadata failed\n";
        return 1;
    }

    std::cout << "All CUDA PCR tests passed\n";
    return 0;
}
