#include "cuda_matmul.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_) {
            // During exception unwinding, report cleanup errors without masking
            // the original error. Normal-path frees use release() and throw.
            cudaError_t status = cudaFree(ptr_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cudaFree during cleanup: %s\n", cudaGetErrorString(status));
            }
        }
    }

    float* data() const { return ptr_; }
    void release() {
        float* ptr = std::exchange(ptr_, nullptr);
        check_cuda(cudaFree(ptr), "cudaFree");
    }

private:
    float* ptr_ = nullptr;
};

std::size_t storage_bytes(const Matrix& matrix) {
    // Matrix already validates the rows * stride multiplication.
    std::size_t elements = matrix.rows() * matrix.row_stride();
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("Matrix byte size overflow");
    }
    return elements * sizeof(float);
}

__global__ void matmul_kernel(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    std::size_t row = index / cols;
    std::size_t col = index % cols;
    float sum = 0.0f;
    for (std::size_t k = 0; k < inner; ++k) {
        sum += a[row * a_stride + k] * b[k * b_stride + col];
    }
    c[row * c_stride + col] = sum;
}

}  // namespace

bool cuda_available(std::string* reason) {
    if (reason) {
        reason->clear();
    }
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        if (reason) {
            *reason = status == cudaSuccess ? "No CUDA devices found"
                : std::string("cudaGetDeviceCount: ") + cudaGetErrorString(status);
        }
        return false;
    }
    // Force runtime/context initialization, even when a driver enumerates a GPU
    // that the process cannot actually access.
    status = cudaFree(nullptr);
    if (status != cudaSuccess) {
        if (reason) {
            *reason = std::string("CUDA runtime initialization: ") + cudaGetErrorString(status);
        }
        return false;
    }
    return true;
}

Matrix cuda_matmul(const Matrix& a, const Matrix& b) {
    return cuda_matmul(a, b, b.cols());
}

Matrix cuda_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }
    Matrix result(a.rows(), b.cols(), row_stride);
    if (result.rows() == 0 || result.cols() == 0 || a.cols() == 0) {
        return result;
    }

    constexpr unsigned threads = 256;
    std::size_t elements = result.rows() * result.cols();
    std::size_t blocks = elements / threads + (elements % threads != 0);
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("Matrix exceeds the naive CUDA kernel's grid limit");
    }

    const auto a_bytes = storage_bytes(a);
    const auto b_bytes = storage_bytes(b);
    const auto c_bytes = storage_bytes(result);
    DeviceBuffer device_a(a_bytes);
    DeviceBuffer device_b(b_bytes);
    DeviceBuffer device_c(c_bytes);
    check_cuda(cudaMemcpy(device_a.data(), a.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A to device");
    check_cuda(cudaMemcpy(device_b.data(), b.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy B to device");
    check_cuda(cudaMemset(device_c.data(), 0, c_bytes), "cudaMemset C");

    matmul_kernel<<<static_cast<unsigned>(blocks), threads>>>(
        device_a.data(), device_b.data(), device_c.data(),
        a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride()
    );
    check_cuda(cudaGetLastError(), "matmul kernel launch");
    check_cuda(cudaDeviceSynchronize(), "matmul kernel synchronization");
    check_cuda(cudaMemcpy(result.data(), device_c.data(), c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C to host");

    device_c.release();
    device_b.release();
    device_a.release();
    return result;
}
