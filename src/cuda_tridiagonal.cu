#include "cuda_tridiagonal.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status));
    }
}

class DeviceDoubleBuffer {
public:
    explicit DeviceDoubleBuffer(std::size_t count) {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&ptr_),
                count * sizeof(double)),
            "cudaMalloc");
    }

    DeviceDoubleBuffer(const DeviceDoubleBuffer&) = delete;
    DeviceDoubleBuffer& operator=(const DeviceDoubleBuffer&) = delete;

    ~DeviceDoubleBuffer() {
        if (ptr_) {
            const cudaError_t status = cudaFree(ptr_);

            if (status != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "cudaFree during cleanup: %s\n",
                    cudaGetErrorString(status));
            }
        }
    }

    double* data() const {
        return ptr_;
    }

private:
    double* ptr_ = nullptr;
};

}  // namespace

namespace matmul_inspector {

__global__ void pcr_stage_kernel(
    const double* a,
    const double* b,
    const double* c,
    const double* d,
    double* next_a,
    double* next_b,
    double* next_c,
    double* next_d,
    std::size_t n,
    std::size_t offset) {

    const std::size_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    next_a[i] = 0.0;
    next_b[i] = b[i];
    next_c[i] = 0.0;
    next_d[i] = d[i];

    if (i >= offset) {
        const std::size_t left = i - offset;
        const double alpha = -a[i] / b[left];

        next_a[i] = alpha * a[left];
        next_b[i] += alpha * c[left];
        next_d[i] += alpha * d[left];
    }

    if (i + offset < n) {
        const std::size_t right = i + offset;
        const double beta = -c[i] / b[right];

        next_b[i] += beta * a[right];
        next_c[i] = beta * c[right];
        next_d[i] += beta * d[right];
    }
}

__global__ void pcr_solve_kernel(
    const double* b,
    const double* d,
    double* x,
    std::size_t n) {

    const std::size_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    x[i] = d[i] / b[i];
}

std::vector<double> solve_pcr_cuda(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs) {

    const std::size_t n = diag.size();

    if (n == 0) {
        throw std::invalid_argument("Diagonal must not be empty");
    }

    if (rhs.size() != n ||
        lower.size() != n - 1 ||
        upper.size() != n - 1) {
        throw std::invalid_argument("Invalid tridiagonal dimensions");
    }

    std::vector<double> a(n, 0.0);
    std::vector<double> b = diag;
    std::vector<double> c(n, 0.0);
    std::vector<double> d = rhs;

    for (std::size_t i = 1; i < n; ++i) {
        a[i] = lower[i - 1];
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        c[i] = upper[i];
    }

    DeviceDoubleBuffer device_a(n);
    DeviceDoubleBuffer device_b(n);
    DeviceDoubleBuffer device_c(n);
    DeviceDoubleBuffer device_d(n);

    DeviceDoubleBuffer device_next_a(n);
    DeviceDoubleBuffer device_next_b(n);
    DeviceDoubleBuffer device_next_c(n);
    DeviceDoubleBuffer device_next_d(n);

    const std::size_t bytes = n * sizeof(double);

    check_cuda(
        cudaMemcpy(
            device_a.data(),
            a.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "copy PCR a to device");

    check_cuda(
        cudaMemcpy(
            device_b.data(),
            b.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "copy PCR b to device");

    check_cuda(
        cudaMemcpy(
            device_c.data(),
            c.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "copy PCR c to device");

    check_cuda(
        cudaMemcpy(
            device_d.data(),
            d.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "copy PCR d to device");

    DeviceDoubleBuffer device_x(n);

    double* current_a = device_a.data();
    double* current_b = device_b.data();
    double* current_c = device_c.data();
    double* current_d = device_d.data();

    double* next_a = device_next_a.data();
    double* next_b = device_next_b.data();
    double* next_c = device_next_c.data();
    double* next_d = device_next_d.data();

    constexpr unsigned int threads_per_block = 256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (n + threads_per_block - 1) /
            threads_per_block);

    for (std::size_t offset = 1; offset < n; offset *= 2) {
        pcr_stage_kernel<<<blocks, threads_per_block>>>(
            current_a,
            current_b,
            current_c,
            current_d,
            next_a,
            next_b,
            next_c,
            next_d,
            n,
            offset);

        check_cuda(
            cudaGetLastError(),
            "PCR stage kernel launch");

        std::swap(current_a, next_a);
        std::swap(current_b, next_b);
        std::swap(current_c, next_c);
        std::swap(current_d, next_d);
    }

    pcr_solve_kernel<<<blocks, threads_per_block>>>(
        current_b,
        current_d,
        device_x.data(),
        n);

    check_cuda(
        cudaGetLastError(),
        "PCR solve kernel launch");

    std::vector<double> result(n);

    check_cuda(
        cudaDeviceSynchronize(),
        "PCR synchronization");

    check_cuda(
        cudaMemcpy(
            result.data(),
            device_x.data(),
            bytes,
            cudaMemcpyDeviceToHost),
        "copy PCR result to host");

    return result;
}

}  // namespace matmul_inspector