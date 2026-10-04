#include "cuda_tridiagonal.hpp"
#include "cuda_tridiagonal_benchmark.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <memory>
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

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
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

namespace {

struct PreparedPcr {
    explicit PreparedPcr(const TridiagonalSystem& system)
        : n(system.diag.size()), a(n,0.0), b(system.diag), c(n,0.0), d(system.rhs),
          device_a(n), device_b(n), device_c(n), device_d(n),
          device_next_a(n), device_next_b(n), device_next_c(n), device_next_d(n), device_x(n) {
        for (std::size_t i=1; i<n; ++i) a[i]=system.lower[i-1];
        for (std::size_t i=0; i+1<n; ++i) c[i]=system.upper[i];
        const auto bytes=n*sizeof(double);
        check_cuda(cudaMemcpy(device_a.data(),a.data(),bytes,cudaMemcpyHostToDevice),"copy benchmark a");
        check_cuda(cudaMemcpy(device_b.data(),b.data(),bytes,cudaMemcpyHostToDevice),"copy benchmark b");
        check_cuda(cudaMemcpy(device_c.data(),c.data(),bytes,cudaMemcpyHostToDevice),"copy benchmark c");
        check_cuda(cudaMemcpy(device_d.data(),d.data(),bytes,cudaMemcpyHostToDevice),"copy benchmark d");
    }
    void reset() {
        const auto bytes=n*sizeof(double);
        check_cuda(cudaMemcpy(device_a.data(),a.data(),bytes,cudaMemcpyHostToDevice),"reset benchmark a");
        check_cuda(cudaMemcpy(device_b.data(),b.data(),bytes,cudaMemcpyHostToDevice),"reset benchmark b");
        check_cuda(cudaMemcpy(device_c.data(),c.data(),bytes,cudaMemcpyHostToDevice),"reset benchmark c");
        check_cuda(cudaMemcpy(device_d.data(),d.data(),bytes,cudaMemcpyHostToDevice),"reset benchmark d");
    }
    std::size_t n;
    std::vector<double> a,b,c,d;
    DeviceDoubleBuffer device_a,device_b,device_c,device_d;
    DeviceDoubleBuffer device_next_a,device_next_b,device_next_c,device_next_d,device_x;
};

void launch_prepared(PreparedPcr& p) {
    double *current_a=p.device_a.data(), *current_b=p.device_b.data();
    double *current_c=p.device_c.data(), *current_d=p.device_d.data();
    double *next_a=p.device_next_a.data(), *next_b=p.device_next_b.data();
    double *next_c=p.device_next_c.data(), *next_d=p.device_next_d.data();
    constexpr unsigned threads=256;
    const auto blocks=static_cast<unsigned>((p.n+threads-1)/threads);
    for (std::size_t offset=1; offset<p.n; offset*=2) {
        pcr_stage_kernel<<<blocks,threads>>>(current_a,current_b,current_c,current_d,
            next_a,next_b,next_c,next_d,p.n,offset);
        check_cuda(cudaGetLastError(),"benchmark PCR stage launch");
        std::swap(current_a,next_a); std::swap(current_b,next_b);
        std::swap(current_c,next_c); std::swap(current_d,next_d);
    }
    pcr_solve_kernel<<<blocks,threads>>>(current_b,current_d,p.device_x.data(),p.n);
    check_cuda(cudaGetLastError(),"benchmark PCR solve launch");
}

benchmark::Statistics analyze_samples(const std::vector<benchmark::Sample>& samples) {
    benchmark::AnalysisOptions options;
    options.bootstrap_samples=0;
    return benchmark::analyze(samples,options);
}

}  // namespace

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

TridiagonalBenchmarkResult benchmark_cuda_pcr_kernel_only(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid CUDA PCR benchmark configuration");
    const auto n=systems.front().diag.size();
    std::vector<std::unique_ptr<PreparedPcr>> prepared;
    prepared.reserve(systems.size());
    for (const auto& system : systems) {
        if (system.diag.size()!=n) throw std::invalid_argument("Mixed system sizes");
        prepared.push_back(std::make_unique<PreparedPcr>(system));
    }
    // Untimed correctness launch before collecting any samples.
    std::vector<double> result(n);
    for (std::size_t i=0; i<prepared.size(); ++i) {
        launch_prepared(*prepared[i]);
        check_cuda(cudaDeviceSynchronize(),"verify benchmark PCR");
        check_cuda(cudaMemcpy(result.data(),prepared[i]->device_x.data(),n*sizeof(double),cudaMemcpyDeviceToHost),
                   "copy benchmark verification result");
        verify_tridiagonal_solution(result,systems[i].expected);
    }
    std::vector<benchmark::Sample> samples;
    CudaEvent start,stop;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            // Restore coefficients before the event: copies are deliberately
            // excluded from kernel-only timing.
            for (auto& system : prepared) system->reset();
            check_cuda(cudaEventRecord(start.get()),"record PCR start");
            for (auto& system : prepared) launch_prepared(*system);
            check_cuda(cudaEventRecord(stop.get()),"record PCR stop");
            check_cuda(cudaEventSynchronize(stop.get()),"synchronize PCR event");
            float elapsed=0;
            check_cuda(cudaEventElapsedTime(&elapsed,start.get(),stop.get()),"elapsed PCR time");
            samples.push_back({0,iteration,warmup,elapsed});
        }
    }
    for (std::size_t i=0; i<prepared.size(); ++i) {
        check_cuda(cudaMemcpy(result.data(),prepared[i]->device_x.data(),n*sizeof(double),cudaMemcpyDeviceToHost),
                   "copy benchmark result");
        verify_tridiagonal_solution(result,systems[i].expected);
    }
    return {"pcr","cuda","kernel_only","serial_host_loop",n,systems.size(),
        warmups,iterations,analyze_samples(samples)};
}

TridiagonalBenchmarkResult benchmark_cuda_pcr_end_to_end(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid CUDA PCR benchmark configuration");
    // Correctness is checked before host timing begins.
    for (const auto& system : systems)
        verify_tridiagonal_solution(
            solve_pcr_cuda(system.lower,system.diag,system.upper,system.rhs),system.expected);
    std::vector<benchmark::Sample> samples;
    std::vector<std::vector<double>> output;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            const auto start=std::chrono::steady_clock::now();
            output.clear(); output.reserve(systems.size());
            for (const auto& system : systems)
                output.push_back(solve_pcr_cuda(system.lower,system.diag,system.upper,system.rhs));
            const auto stop=std::chrono::steady_clock::now();
            samples.push_back({0,iteration,warmup,
                std::chrono::duration<float,std::milli>(stop-start).count()});
        }
    }
    for (std::size_t i=0; i<systems.size(); ++i)
        verify_tridiagonal_solution(output[i],systems[i].expected);
    return {"pcr","cuda","end_to_end","serial_host_loop",
        systems.front().diag.size(),systems.size(),warmups,iterations,analyze_samples(samples)};
}

}  // namespace matmul_inspector
