#include "cuda_tridiagonal.hpp"
#include "cuda_tridiagonal_benchmark.hpp"
#include "cuda_tridiagonal_workspace.hpp"
#include "tridiagonal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <limits>
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

__global__ void pcr_batched_stage_kernel(
    const double* a,
    const double* b,
    const double* c,
    const double* d,
    double* next_a,
    double* next_b,
    double* next_c,
    double* next_d,
    std::size_t n,
    std::size_t total,
    std::size_t offset) {

    const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
    if (global >= total) return;

    const std::size_t system = global / n;
    const std::size_t equation = global % n;
    const std::size_t system_base = system * n;

    next_a[global] = 0.0;
    next_b[global] = b[global];
    next_c[global] = 0.0;
    next_d[global] = d[global];

    if (equation >= offset) {
        const std::size_t left = system_base + equation - offset;
        const double alpha = -a[global] / b[left];
        next_a[global] = alpha * a[left];
        next_b[global] += alpha * c[left];
        next_d[global] += alpha * d[left];
    }

    if (equation + offset < n) {
        const std::size_t right = system_base + equation + offset;
        const double beta = -c[global] / b[right];
        next_b[global] += beta * a[right];
        next_c[global] = beta * c[right];
        next_d[global] += beta * d[right];
    }
}

__global__ void pcr_batched_solve_kernel(
    const double* b,
    const double* d,
    double* x,
    std::size_t total) {

    const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
    if (global < total) x[global] = d[global] / b[global];
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

struct FlatBatch {
    std::size_t n = 0;
    std::size_t batch = 0;
    std::vector<double> a, b, c, d;
};

FlatBatch flatten_batch(
    const std::vector<std::vector<double>>& lower,
    const std::vector<std::vector<double>>& diag,
    const std::vector<std::vector<double>>& upper,
    const std::vector<std::vector<double>>& rhs) {
    const auto batch=diag.size();
    if (!batch || lower.size()!=batch || upper.size()!=batch || rhs.size()!=batch)
        throw std::invalid_argument("Invalid batched tridiagonal dimensions");
    const auto n=diag.front().size();
    if (!n) throw std::invalid_argument("Diagonal must not be empty");
    FlatBatch flat;
    flat.n=n; flat.batch=batch;
    const auto total=n*batch;
    flat.a.assign(total,0.0); flat.b.resize(total);
    flat.c.assign(total,0.0); flat.d.resize(total);
    for (std::size_t system=0; system<batch; ++system) {
        if (diag[system].size()!=n || rhs[system].size()!=n ||
            lower[system].size()!=n-1 || upper[system].size()!=n-1)
            throw std::invalid_argument("Invalid batched tridiagonal dimensions");
        const auto base=system*n;
        for (std::size_t i=0; i<n; ++i) {
            flat.b[base+i]=diag[system][i];
            flat.d[base+i]=rhs[system][i];
            if (i) flat.a[base+i]=lower[system][i-1];
            if (i+1<n) flat.c[base+i]=upper[system][i];
        }
    }
    return flat;
}

FlatBatch flatten_systems(const std::vector<TridiagonalSystem>& systems) {
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    lower.reserve(systems.size()); diag.reserve(systems.size());
    upper.reserve(systems.size()); rhs.reserve(systems.size());
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }
    return flatten_batch(lower,diag,upper,rhs);
}

struct PreparedBatchedPcr {
    PreparedBatchedPcr(std::size_t n,std::size_t batch)
        : flat{n,batch,{}, {}, {}, {}}, total(n*batch),
          device_a(total), device_b(total), device_c(total), device_d(total),
          device_next_a(total), device_next_b(total), device_next_c(total),
          device_next_d(total), device_x(total) {}
    explicit PreparedBatchedPcr(FlatBatch input)
        : PreparedBatchedPcr(input.n,input.batch) {
        upload(std::move(input));
    }
    void upload(FlatBatch input) {
        if (input.n!=flat.n || input.batch!=flat.batch)
            throw std::invalid_argument("Batched PCR workspace dimensions do not match input");
        flat=std::move(input);
        reset();
    }
    void reset() {
        const auto bytes=total*sizeof(double);
        check_cuda(cudaMemcpy(device_a.data(),flat.a.data(),bytes,cudaMemcpyHostToDevice),
                   "reset batched PCR a");
        check_cuda(cudaMemcpy(device_b.data(),flat.b.data(),bytes,cudaMemcpyHostToDevice),
                   "reset batched PCR b");
        check_cuda(cudaMemcpy(device_c.data(),flat.c.data(),bytes,cudaMemcpyHostToDevice),
                   "reset batched PCR c");
        check_cuda(cudaMemcpy(device_d.data(),flat.d.data(),bytes,cudaMemcpyHostToDevice),
                   "reset batched PCR d");
    }
    FlatBatch flat;
    std::size_t total;
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

void launch_prepared_batched(PreparedBatchedPcr& p) {
    double *current_a=p.device_a.data(), *current_b=p.device_b.data();
    double *current_c=p.device_c.data(), *current_d=p.device_d.data();
    double *next_a=p.device_next_a.data(), *next_b=p.device_next_b.data();
    double *next_c=p.device_next_c.data(), *next_d=p.device_next_d.data();
    constexpr unsigned threads=256;
    const auto blocks=static_cast<unsigned>((p.total+threads-1)/threads);
    for (std::size_t offset=1; offset<p.flat.n; offset*=2) {
        pcr_batched_stage_kernel<<<blocks,threads>>>(current_a,current_b,current_c,current_d,
            next_a,next_b,next_c,next_d,p.flat.n,p.total,offset);
        check_cuda(cudaGetLastError(),"batched PCR stage launch");
        std::swap(current_a,next_a); std::swap(current_b,next_b);
        std::swap(current_c,next_c); std::swap(current_d,next_d);
    }
    pcr_batched_solve_kernel<<<blocks,threads>>>(
        current_b,current_d,p.device_x.data(),p.total);
    check_cuda(cudaGetLastError(),"batched PCR solve launch");
}

std::vector<double> copy_batched_result(const PreparedBatchedPcr& p) {
    std::vector<double> result(p.total);
    check_cuda(cudaMemcpy(result.data(),p.device_x.data(),p.total*sizeof(double),
                          cudaMemcpyDeviceToHost),"copy batched PCR result");
    return result;
}

void verify_batched_result(const std::vector<double>& result,
                           const std::vector<TridiagonalSystem>& systems) {
    const auto n=systems.front().diag.size();
    if (result.size()!=n*systems.size())
        throw std::runtime_error("Invalid batched PCR result size");
    for (std::size_t system=0; system<systems.size(); ++system) {
        const auto begin=result.begin()+static_cast<std::ptrdiff_t>(system*n);
        const std::vector<double> actual(begin,begin+static_cast<std::ptrdiff_t>(n));
        verify_tridiagonal_solution(actual,systems[system].expected);
        verify_tridiagonal_solution(actual,solve_pcr(
            systems[system].lower,systems[system].diag,
            systems[system].upper,systems[system].rhs));
    }
}

benchmark::Statistics analyze_samples(const std::vector<benchmark::Sample>& samples) {
    benchmark::AnalysisOptions options;
    options.bootstrap_samples=0;
    return benchmark::analyze(samples,options);
}

std::size_t checked_workspace_size(std::size_t n,std::size_t batch) {
    if (!n || !batch || n>std::numeric_limits<std::size_t>::max()/batch)
        throw std::invalid_argument("Invalid batched PCR workspace dimensions");
    return n;
}

}  // namespace

struct CudaPcrBatchedWorkspace::Impl {
    Impl(std::size_t n,std::size_t batch)
        : prepared(checked_workspace_size(n,batch),batch) {}

    PreparedBatchedPcr prepared;
    std::unique_ptr<DeviceDoubleBuffer> resident_a,resident_b,resident_c,resident_d;
    bool uploaded=false;
    bool executed=false;
    bool resident=false;
};

CudaPcrBatchedWorkspace::CudaPcrBatchedWorkspace(
    std::size_t system_size,std::size_t batch_size)
    : impl_(std::make_unique<Impl>(system_size,batch_size)) {}

CudaPcrBatchedWorkspace::~CudaPcrBatchedWorkspace() = default;
CudaPcrBatchedWorkspace::CudaPcrBatchedWorkspace(CudaPcrBatchedWorkspace&&) noexcept = default;
CudaPcrBatchedWorkspace& CudaPcrBatchedWorkspace::operator=(
    CudaPcrBatchedWorkspace&&) noexcept = default;

std::size_t CudaPcrBatchedWorkspace::system_size() const noexcept {
    return impl_ ? impl_->prepared.flat.n : 0;
}

std::size_t CudaPcrBatchedWorkspace::batch_size() const noexcept {
    return impl_ ? impl_->prepared.flat.batch : 0;
}

void CudaPcrBatchedWorkspace::upload(
    const std::vector<std::vector<double>>& lower,
    const std::vector<std::vector<double>>& diag,
    const std::vector<std::vector<double>>& upper,
    const std::vector<std::vector<double>>& rhs) {
    impl_->prepared.upload(flatten_batch(lower,diag,upper,rhs));
    impl_->uploaded=true;
    impl_->executed=false;
    impl_->resident=false;
}

void CudaPcrBatchedWorkspace::reset() {
    if (!impl_->uploaded)
        throw std::logic_error("Batched PCR workspace has no uploaded coefficients");
    impl_->prepared.reset();
    impl_->executed=false;
}

void CudaPcrBatchedWorkspace::make_device_resident() {
    if (!impl_->uploaded)
        throw std::logic_error("Batched PCR workspace has no uploaded coefficients");
    if (!impl_->resident_a) {
        impl_->resident_a=std::make_unique<DeviceDoubleBuffer>(impl_->prepared.total);
        impl_->resident_b=std::make_unique<DeviceDoubleBuffer>(impl_->prepared.total);
        impl_->resident_c=std::make_unique<DeviceDoubleBuffer>(impl_->prepared.total);
        impl_->resident_d=std::make_unique<DeviceDoubleBuffer>(impl_->prepared.total);
    }
    const auto bytes=impl_->prepared.total*sizeof(double);
    check_cuda(cudaMemcpy(impl_->resident_a->data(),impl_->prepared.device_a.data(),bytes,
                          cudaMemcpyDeviceToDevice),"snapshot resident PCR a");
    check_cuda(cudaMemcpy(impl_->resident_b->data(),impl_->prepared.device_b.data(),bytes,
                          cudaMemcpyDeviceToDevice),"snapshot resident PCR b");
    check_cuda(cudaMemcpy(impl_->resident_c->data(),impl_->prepared.device_c.data(),bytes,
                          cudaMemcpyDeviceToDevice),"snapshot resident PCR c");
    check_cuda(cudaMemcpy(impl_->resident_d->data(),impl_->prepared.device_d.data(),bytes,
                          cudaMemcpyDeviceToDevice),"snapshot resident PCR d");
    impl_->resident=true;
    impl_->executed=false;
}

void CudaPcrBatchedWorkspace::reset_from_device() {
    if (!impl_->resident)
        throw std::logic_error("Batched PCR workspace has no resident coefficient snapshot");
    const auto bytes=impl_->prepared.total*sizeof(double);
    check_cuda(cudaMemcpy(impl_->prepared.device_a.data(),impl_->resident_a->data(),bytes,
                          cudaMemcpyDeviceToDevice),"restore resident PCR a");
    check_cuda(cudaMemcpy(impl_->prepared.device_b.data(),impl_->resident_b->data(),bytes,
                          cudaMemcpyDeviceToDevice),"restore resident PCR b");
    check_cuda(cudaMemcpy(impl_->prepared.device_c.data(),impl_->resident_c->data(),bytes,
                          cudaMemcpyDeviceToDevice),"restore resident PCR c");
    check_cuda(cudaMemcpy(impl_->prepared.device_d.data(),impl_->resident_d->data(),bytes,
                          cudaMemcpyDeviceToDevice),"restore resident PCR d");
    impl_->executed=false;
}

void CudaPcrBatchedWorkspace::execute() {
    if (!impl_->uploaded)
        throw std::logic_error("Batched PCR workspace has no uploaded coefficients");
    launch_prepared_batched(impl_->prepared);
    impl_->executed=true;
}

void CudaPcrBatchedWorkspace::download(std::vector<double>& result) const {
    if (!impl_->executed)
        throw std::logic_error("Batched PCR workspace has not executed");
    if (result.size()!=impl_->prepared.total)
        throw std::invalid_argument("Batched PCR result buffer has the wrong size");
    check_cuda(cudaMemcpy(result.data(),impl_->prepared.device_x.data(),
                          result.size()*sizeof(double),cudaMemcpyDeviceToHost),
               "download reusable batched PCR result");
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

std::vector<std::vector<double>> solve_pcr_cuda_batched(
    const std::vector<std::vector<double>>& lower,
    const std::vector<std::vector<double>>& diag,
    const std::vector<std::vector<double>>& upper,
    const std::vector<std::vector<double>>& rhs) {
    PreparedBatchedPcr prepared(flatten_batch(lower,diag,upper,rhs));
    launch_prepared_batched(prepared);
    check_cuda(cudaDeviceSynchronize(),"batched PCR synchronization");
    const auto flat_result=copy_batched_result(prepared);
    std::vector<std::vector<double>> result(
        prepared.flat.batch,std::vector<double>(prepared.flat.n));
    for (std::size_t system=0; system<prepared.flat.batch; ++system)
        std::copy_n(flat_result.begin()+static_cast<std::ptrdiff_t>(system*prepared.flat.n),
                    prepared.flat.n,result[system].begin());
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

TridiagonalBenchmarkResult benchmark_cuda_pcr_true_batched_kernel_only(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid true-batched CUDA PCR benchmark configuration");
    PreparedBatchedPcr prepared(flatten_systems(systems));

    // Correctness and CPU-PCR agreement are established before sampling.
    launch_prepared_batched(prepared);
    check_cuda(cudaDeviceSynchronize(),"verify true-batched PCR");
    verify_batched_result(copy_batched_result(prepared),systems);

    std::vector<benchmark::Sample> samples;
    CudaEvent start,stop;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            prepared.reset();
            check_cuda(cudaEventRecord(start.get()),"record true-batched PCR start");
            launch_prepared_batched(prepared);
            check_cuda(cudaEventRecord(stop.get()),"record true-batched PCR stop");
            check_cuda(cudaEventSynchronize(stop.get()),"synchronize true-batched PCR event");
            float elapsed=0;
            check_cuda(cudaEventElapsedTime(&elapsed,start.get(),stop.get()),
                       "elapsed true-batched PCR time");
            samples.push_back({0,iteration,warmup,elapsed});
        }
    }
    verify_batched_result(copy_batched_result(prepared),systems);
    return {"pcr","cuda","kernel_only","true_batched_gpu",prepared.flat.n,
        prepared.flat.batch,warmups,iterations,analyze_samples(samples)};
}

TridiagonalBenchmarkResult benchmark_cuda_pcr_true_batched_end_to_end(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid true-batched CUDA PCR benchmark configuration");
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    lower.reserve(systems.size()); diag.reserve(systems.size());
    upper.reserve(systems.size()); rhs.reserve(systems.size());
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }

    auto output=solve_pcr_cuda_batched(lower,diag,upper,rhs);
    std::vector<double> flat_output;
    flat_output.reserve(systems.size()*systems.front().diag.size());
    for (const auto& system : output)
        flat_output.insert(flat_output.end(),system.begin(),system.end());
    verify_batched_result(flat_output,systems);

    std::vector<benchmark::Sample> samples;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            const auto start=std::chrono::steady_clock::now();
            output=solve_pcr_cuda_batched(lower,diag,upper,rhs);
            const auto stop=std::chrono::steady_clock::now();
            samples.push_back({0,iteration,warmup,
                std::chrono::duration<float,std::milli>(stop-start).count()});
        }
    }
    flat_output.clear();
    for (const auto& system : output)
        flat_output.insert(flat_output.end(),system.begin(),system.end());
    verify_batched_result(flat_output,systems);
    return {"pcr","cuda","end_to_end","true_batched_gpu",
        systems.front().diag.size(),systems.size(),warmups,iterations,
        analyze_samples(samples)};
}

std::vector<TridiagonalBenchmarkResult> benchmark_cuda_pcr_true_batched_reuse(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid reusable CUDA PCR benchmark configuration");
    const auto n=systems.front().diag.size();
    const auto batch=systems.size();
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    lower.reserve(batch); diag.reserve(batch); upper.reserve(batch); rhs.reserve(batch);
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }

    CudaPcrBatchedWorkspace workspace(n,batch);
    workspace.upload(lower,diag,upper,rhs);
    workspace.execute();
    std::vector<double> output(n*batch);
    workspace.download(output);
    verify_batched_result(output,systems);

    const auto one_shot=solve_pcr_cuda_batched(lower,diag,upper,rhs);
    for (std::size_t system=0; system<batch; ++system) {
        const auto begin=output.begin()+static_cast<std::ptrdiff_t>(system*n);
        verify_tridiagonal_solution(
            std::vector<double>(begin,begin+static_cast<std::ptrdiff_t>(n)),one_shot[system]);
    }

    auto event_samples = [&](auto&& operation) {
        std::vector<benchmark::Sample> samples;
        CudaEvent start,stop;
        for (int phase=0; phase<2; ++phase) {
            const bool warmup=phase==0;
            const int count=warmup?warmups:iterations;
            for (int iteration=0; iteration<count; ++iteration) {
                check_cuda(cudaEventRecord(start.get()),"record reusable PCR component start");
                operation();
                check_cuda(cudaEventRecord(stop.get()),"record reusable PCR component stop");
                check_cuda(cudaEventSynchronize(stop.get()),
                           "synchronize reusable PCR component event");
                float elapsed=0;
                check_cuda(cudaEventElapsedTime(&elapsed,start.get(),stop.get()),
                           "elapsed reusable PCR component");
                samples.push_back({0,iteration,warmup,elapsed});
            }
        }
        return samples;
    };

    std::vector<benchmark::Sample> setup_samples;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            const auto start=std::chrono::steady_clock::now();
            auto temporary=std::make_unique<CudaPcrBatchedWorkspace>(n,batch);
            const auto stop=std::chrono::steady_clock::now();
            setup_samples.push_back({0,iteration,warmup,
                std::chrono::duration<float,std::milli>(stop-start).count()});
            temporary.reset();
        }
    }

    const auto h2d_samples=event_samples([&] { workspace.reset(); });

    std::vector<benchmark::Sample> kernel_samples;
    CudaEvent kernel_start,kernel_stop;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            workspace.reset();
            check_cuda(cudaEventRecord(kernel_start.get()),"record reusable PCR kernel start");
            workspace.execute();
            check_cuda(cudaEventRecord(kernel_stop.get()),"record reusable PCR kernel stop");
            check_cuda(cudaEventSynchronize(kernel_stop.get()),
                       "synchronize reusable PCR kernel event");
            float elapsed=0;
            check_cuda(cudaEventElapsedTime(&elapsed,kernel_start.get(),kernel_stop.get()),
                       "elapsed reusable PCR kernel");
            kernel_samples.push_back({0,iteration,warmup,elapsed});
        }
    }

    const auto d2h_samples=event_samples([&] { workspace.download(output); });

    std::vector<benchmark::Sample> total_samples;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            const auto start=std::chrono::steady_clock::now();
            workspace.reset();
            workspace.execute();
            workspace.download(output);
            const auto stop=std::chrono::steady_clock::now();
            total_samples.push_back({0,iteration,warmup,
                std::chrono::duration<float,std::milli>(stop-start).count()});
        }
    }
    verify_batched_result(output,systems);

    auto result = [&](const char* scope,const std::vector<benchmark::Sample>& samples) {
        return TridiagonalBenchmarkResult{"pcr","cuda",scope,"true_batched_reuse",
            n,batch,warmups,iterations,analyze_samples(samples)};
    };
    return {
        result("allocation_setup",setup_samples),
        result("h2d",h2d_samples),
        result("kernel_only",kernel_samples),
        result("d2h",d2h_samples),
        result("reusable_end_to_end",total_samples)
    };
}

std::vector<TridiagonalBenchmarkResult> benchmark_cuda_pcr_device_resident(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    if (systems.empty() || iterations<=0 || warmups<0)
        throw std::invalid_argument("Invalid device-resident CUDA PCR benchmark configuration");
    const auto n=systems.front().diag.size();
    const auto batch=systems.size();
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    lower.reserve(batch); diag.reserve(batch); upper.reserve(batch); rhs.reserve(batch);
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }

    CudaPcrBatchedWorkspace workspace(n,batch);
    workspace.upload(lower,diag,upper,rhs);
    workspace.make_device_resident();

    // Establish correctness against known solutions, CPU PCR, and the existing
    // one-shot true-batched CUDA path before collecting samples.
    workspace.execute();
    std::vector<double> output(n*batch);
    workspace.download(output);
    verify_batched_result(output,systems);
    const auto one_shot=solve_pcr_cuda_batched(lower,diag,upper,rhs);
    for (std::size_t system=0; system<batch; ++system) {
        const auto begin=output.begin()+static_cast<std::ptrdiff_t>(system*n);
        verify_tridiagonal_solution(
            std::vector<double>(begin,begin+static_cast<std::ptrdiff_t>(n)),one_shot[system]);
    }

    auto event_samples = [&](auto&& operation,const char* description) {
        std::vector<benchmark::Sample> samples;
        CudaEvent start,stop;
        for (int phase=0; phase<2; ++phase) {
            const bool warmup=phase==0;
            const int count=warmup?warmups:iterations;
            for (int iteration=0; iteration<count; ++iteration) {
                check_cuda(cudaEventRecord(start.get()),"record device-resident PCR start");
                operation();
                check_cuda(cudaEventRecord(stop.get()),"record device-resident PCR stop");
                check_cuda(cudaEventSynchronize(stop.get()),description);
                float elapsed=0;
                check_cuda(cudaEventElapsedTime(&elapsed,start.get(),stop.get()),
                           "elapsed device-resident PCR operation");
                samples.push_back({0,iteration,warmup,elapsed});
            }
        }
        return samples;
    };

    const auto reset_samples=event_samples(
        [&] { workspace.reset_from_device(); },
        "synchronize resident PCR D2D reset");

    std::vector<benchmark::Sample> solve_samples;
    CudaEvent start,stop;
    for (int phase=0; phase<2; ++phase) {
        const bool warmup=phase==0;
        const int count=warmup?warmups:iterations;
        for (int iteration=0; iteration<count; ++iteration) {
            // Restore inputs before the start event. Default-stream ordering
            // ensures the start timestamp follows the D2D copies.
            workspace.reset_from_device();
            check_cuda(cudaEventRecord(start.get()),"record resident solve start");
            workspace.execute();
            check_cuda(cudaEventRecord(stop.get()),"record resident solve stop");
            check_cuda(cudaEventSynchronize(stop.get()),"synchronize resident solve");
            float elapsed=0;
            check_cuda(cudaEventElapsedTime(&elapsed,start.get(),stop.get()),
                       "elapsed resident solve");
            solve_samples.push_back({0,iteration,warmup,elapsed});
        }
    }

    workspace.download(output);
    verify_batched_result(output,systems);

    auto result = [&](const char* scope,const std::vector<benchmark::Sample>& samples) {
        return TridiagonalBenchmarkResult{"pcr","cuda",scope,
            "true_batched_device_resident",n,batch,warmups,iterations,
            analyze_samples(samples)};
    };
    return {
        result("d2d_reset",reset_samples),
        result("device_resident",solve_samples)
    };
}

}  // namespace matmul_inspector
