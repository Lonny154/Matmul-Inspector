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

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() {
        if (event_) {
            cudaError_t status = cudaEventDestroy(event_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cudaEventDestroy during cleanup: %s\n", cudaGetErrorString(status));
            }
        }
    }
    cudaEvent_t get() const { return event_; }
    void release() {
        check_cuda(cudaEventDestroy(std::exchange(event_, nullptr)), "cudaEventDestroy");
    }
private:
    cudaEvent_t event_ = nullptr;
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

// The controlled sequential variants instantiate the same body and differ only
// in the rounding primitive. Intrinsic operations cannot be contracted/reassociated.
template<bool Fused>
__device__ __forceinline__ float controlled_madd(float a, float b, float sum) {
    if constexpr (Fused) return __fmaf_rn(a, b, sum);
    else return __fadd_rn(__fmul_rn(a, b), sum);
}

template<bool Fused, bool Reordered>
__device__ __forceinline__ void controlled_matmul(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const std::size_t row = index / cols, col = index % cols;
    float sum = 0.0f;
    if constexpr (Reordered) {
        float odd = 0.0f;
        for (std::size_t k = 0; k < inner; k += 2) {
            sum = controlled_madd<Fused>(a[row * a_stride + k], b[k * b_stride + col], sum);
            if (k + 1 < inner)
                odd = controlled_madd<Fused>(a[row * a_stride + k + 1], b[(k + 1) * b_stride + col], odd);
        }
        sum = __fadd_rn(sum, odd);
    } else {
        for (std::size_t k = 0; k < inner; ++k)
            sum = controlled_madd<Fused>(a[row * a_stride + k], b[k * b_stride + col], sum);
    }
    c[row * c_stride + col] = sum;
}

extern "C" __global__ void mi_naive_fma(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    controlled_matmul<true, false>(a, b, c, rows, cols, inner, a_stride, b_stride, c_stride);
}

extern "C" __global__ void mi_naive_no_fma(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    controlled_matmul<false, false>(a, b, c, rows, cols, inner, a_stride, b_stride, c_stride);
}

extern "C" __global__ void mi_naive_reordered(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    controlled_matmul<true, true>(a, b, c, rows, cols, inner, a_stride, b_stride, c_stride);
}

__global__ void tiled_matmul_kernel(
    const float* a, const float* b, float* c,
    std::size_t rows, std::size_t cols, std::size_t inner,
    std::size_t a_stride, std::size_t b_stride, std::size_t c_stride
) {
    constexpr unsigned tile = cuda_matmul_tile_size;
    __shared__ float a_tile[tile][tile];
    __shared__ float b_tile[tile][tile];
    const unsigned x = threadIdx.x;
    const unsigned y = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * tile + y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * tile + x;
    float sum = 0.0f;
    for (std::size_t base = 0; base < inner; base += tile) {
        a_tile[y][x] = row < rows && base + x < inner ? a[row * a_stride + base + x] : 0.0f;
        b_tile[y][x] = base + y < inner && col < cols ? b[(base + y) * b_stride + col] : 0.0f;
        __syncthreads();

        // Keep one accumulator and increasing K, including the partial tile.
        // Do not add artificial zero products beyond K to the accumulator.
        for (unsigned k = 0; k < tile && base + k < inner; ++k) {
            sum += a_tile[y][k] * b_tile[k][x];
        }
        // Every thread must finish reading before the next tile overwrites it.
        __syncthreads();
    }
    if (row < rows && col < cols) {
        c[row * c_stride + col] = sum;
    }
}

Matrix run_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride,
                  CudaMatmulKernel kernel, int repetitions, int warmups, benchmark::Statistics* stats);

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
    return cuda_matmul(a, b, row_stride, CudaMatmulKernel::naive);
}

Matrix cuda_matmul(const Matrix& a, const Matrix& b, CudaMatmulKernel kernel) {
    return cuda_matmul(a, b, b.cols(), kernel);
}

Matrix cuda_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride, CudaMatmulKernel kernel) {
    return run_matmul(a, b, row_stride, kernel, 1, 0, nullptr);
}

CudaMatmulBenchmark benchmark_cuda_matmul(const Matrix& a, const Matrix& b, int repetitions, int warmups) {
    if (repetitions <= 0) {
        throw std::invalid_argument("Benchmark repetitions must be positive");
    }
    if (warmups < 0) throw std::invalid_argument("Warmups must be nonnegative");
    benchmark::Statistics naive_stats, tiled_stats;
    Matrix naive = run_matmul(a, b, b.cols(), CudaMatmulKernel::naive, repetitions, warmups, &naive_stats);
    Matrix tiled = run_matmul(a, b, b.cols(), CudaMatmulKernel::tiled, repetitions, warmups, &tiled_stats);
    return {std::move(naive), std::move(tiled), static_cast<float>(naive_stats.mean_ms),
            static_cast<float>(tiled_stats.mean_ms), naive_stats, tiled_stats};
}

CudaKernelMeasurement benchmark_cuda_kernel(const Matrix& a, const Matrix& b,
    CudaMatmulKernel kernel, int repetitions, int warmups) {
    if (repetitions <= 0 || warmups < 0) throw std::invalid_argument("Invalid benchmark counts");
    benchmark::Statistics timing;
    Matrix output = run_matmul(a, b, b.cols(), kernel, repetitions, warmups, &timing);
    return {std::move(output), timing};
}

namespace {

Matrix run_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride,
                  CudaMatmulKernel kernel, int repetitions, int warmups, benchmark::Statistics* stats) {
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
    constexpr unsigned tile = cuda_matmul_tile_size;
    std::size_t blocks_x = result.cols() / tile + (result.cols() % tile != 0);
    std::size_t blocks_y = result.rows() / tile + (result.rows() % tile != 0);
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (kernel != CudaMatmulKernel::tiled && blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("Matrix exceeds the naive CUDA kernel's grid limit");
    }
    if (kernel == CudaMatmulKernel::tiled
        && (blocks_x > static_cast<std::size_t>(properties.maxGridSize[0])
            || blocks_y > static_cast<std::size_t>(properties.maxGridSize[1]))) {
        throw std::length_error("Matrix exceeds the tiled CUDA kernel's grid limit");
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

    auto launch = [&] {
        if (kernel == CudaMatmulKernel::naive) {
            matmul_kernel<<<static_cast<unsigned>(blocks), threads>>>(
                device_a.data(), device_b.data(), device_c.data(),
                a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride()
            );
        } else if (kernel == CudaMatmulKernel::tiled) {
            tiled_matmul_kernel<<<dim3(static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y)), dim3(tile, tile)>>>(
                device_a.data(), device_b.data(), device_c.data(),
                a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride()
            );
        } else if (kernel == CudaMatmulKernel::naive_fma) {
            mi_naive_fma<<<static_cast<unsigned>(blocks), threads>>>(
                device_a.data(), device_b.data(), device_c.data(),
                a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride());
        } else if (kernel == CudaMatmulKernel::naive_no_fma) {
            mi_naive_no_fma<<<static_cast<unsigned>(blocks), threads>>>(
                device_a.data(), device_b.data(), device_c.data(),
                a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride());
        } else if (kernel == CudaMatmulKernel::naive_reordered) {
            mi_naive_reordered<<<static_cast<unsigned>(blocks), threads>>>(
                device_a.data(), device_b.data(), device_c.data(),
                a.rows(), b.cols(), a.cols(), a.row_stride(), b.row_stride(), result.row_stride());
        } else {
            throw std::invalid_argument("Unknown CUDA kernel");
        }
        check_cuda(cudaGetLastError(), "matmul kernel launch");
    };
    if (stats) {
        for (int i = 0; i < warmups; ++i) {
            launch();
        }
        check_cuda(cudaDeviceSynchronize(), "matmul warmup synchronization");
        CudaEvent start, stop;
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(repetitions));
        for (int i = 0; i < repetitions; ++i) {
            check_cuda(cudaEventRecord(start.get()), "cudaEventRecord start");
            launch();
            check_cuda(cudaEventRecord(stop.get()), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start.get(), stop.get()), "cudaEventElapsedTime");
            samples.push_back(ms);
        }
        *stats = benchmark::statistics(samples);
        stop.release();
        start.release();
    } else {
        launch();
    }
    check_cuda(cudaDeviceSynchronize(), "matmul kernel synchronization");
    check_cuda(cudaMemcpy(result.data(), device_c.data(), c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C to host");

    device_c.release();
    device_b.release();
    device_a.release();
    return result;
}

}  // namespace

std::map<std::string, std::string> cuda_metadata() {
    std::map<std::string, std::string> values{
        {"gpu_name", "unknown"}, {"gpu_compute_capability", "unknown"},
        {"cuda_runtime_version", "unknown"}, {"cuda_driver_api_version", "unknown"}};
    int runtime = 0, driver = 0, device = 0;
    if (cudaRuntimeGetVersion(&runtime) == cudaSuccess) values["cuda_runtime_version"] = std::to_string(runtime);
    if (cudaDriverGetVersion(&driver) == cudaSuccess && driver != 0) values["cuda_driver_api_version"] = std::to_string(driver);
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        values["gpu_name"] = properties.name;
        values["gpu_compute_capability"] = std::to_string(properties.major) + "." + std::to_string(properties.minor);
    }
    return values;
}
