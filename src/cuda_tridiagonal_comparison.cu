#include "cuda_tridiagonal_comparison.hpp"
#include "tridiagonal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef MATMUL_INSPECTOR_SOURCE_REVISION
#define MATMUL_INSPECTOR_SOURCE_REVISION "unknown"
#endif
#ifndef MATMUL_INSPECTOR_BUILD_TYPE
#define MATMUL_INSPECTOR_BUILD_TYPE "unknown"
#endif
#ifndef MATMUL_INSPECTOR_CUDA_ARCHITECTURES
#define MATMUL_INSPECTOR_CUDA_ARCHITECTURES "default"
#endif

namespace matmul_inspector {
namespace {

void check_cuda(cudaError_t status,const char* operation) {
    if (status!=cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&value_),"cudaEventCreate"); }
    ~Event() { if (value_) cudaEventDestroy(value_); }
    Event(const Event&)=delete;
    Event& operator=(const Event&)=delete;
    cudaEvent_t get() const { return value_; }
private:
    cudaEvent_t value_=nullptr;
};

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),count*sizeof(T)),"cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    T* data() const { return data_; }
    std::size_t bytes() const { return count_*sizeof(T); }
private:
    std::size_t count_;
    T* data_=nullptr;
};

__global__ void thomas_batched_kernel(
    const float* lower,const float* diag,const float* upper,const float* rhs,
    float* work_diag,float* work_rhs,float* output,std::size_t n,std::size_t batch) {
    const std::size_t system=blockIdx.x*blockDim.x+threadIdx.x;
    if (system>=batch) return;
    const std::size_t base=system*n;
    work_diag[base]=diag[base];
    work_rhs[base]=rhs[base];
    for (std::size_t i=1;i<n;++i) {
        const std::size_t at=base+i;
        const float factor=lower[at]/work_diag[at-1];
        work_diag[at]=diag[at]-factor*upper[at-1];
        work_rhs[at]=rhs[at]-factor*work_rhs[at-1];
    }
    output[base+n-1]=work_rhs[base+n-1]/work_diag[base+n-1];
    for (std::size_t i=n-1;i>0;--i) {
        const std::size_t at=base+i-1;
        output[at]=(work_rhs[at]-upper[at]*output[at+1])/work_diag[at];
    }
}

__global__ void pcr_float_stage_kernel(
    const float* a,const float* b,const float* c,const float* d,
    float* next_a,float* next_b,float* next_c,float* next_d,
    std::size_t n,std::size_t total,std::size_t offset) {
    const std::size_t global=blockIdx.x*blockDim.x+threadIdx.x;
    if (global>=total) return;
    const std::size_t equation=global%n;
    const std::size_t base=global-equation;
    float na=0.0f,nb=b[global],nc=0.0f,nd=d[global];
    if (equation>=offset) {
        const std::size_t left=base+equation-offset;
        const float alpha=-a[global]/b[left];
        na=alpha*a[left]; nb+=alpha*c[left]; nd+=alpha*d[left];
    }
    if (equation+offset<n) {
        const std::size_t right=base+equation+offset;
        const float beta=-c[global]/b[right];
        nb+=beta*a[right]; nc=beta*c[right]; nd+=beta*d[right];
    }
    next_a[global]=na; next_b[global]=nb; next_c[global]=nc; next_d[global]=nd;
}

__global__ void pcr_float_solve_kernel(
    const float* b,const float* d,float* output,std::size_t total) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<total) output[i]=d[i]/b[i];
}

struct HostProblem {
    std::size_t n=0,batch=0;
    std::vector<float> lower,diag,upper,rhs;
    std::vector<double> reference;
};

HostProblem make_problem(std::size_t n,std::size_t batch,std::uint32_t seed) {
    if (!n || !batch || n>std::numeric_limits<std::size_t>::max()/batch)
        throw std::invalid_argument("Invalid problem dimensions");
    HostProblem p; p.n=n; p.batch=batch;
    const auto total=n*batch;
    p.lower.assign(total,0.0f); p.diag.resize(total);
    p.upper.assign(total,0.0f); p.rhs.resize(total); p.reference.resize(total);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> off(-0.75f,0.75f);
    std::uniform_real_distribution<float> solution(-2.0f,2.0f);
    for (std::size_t s=0;s<batch;++s) {
        const auto base=s*n;
        std::vector<double> lo(n ? n-1 : 0),di(n),up(n ? n-1 : 0),right(n);
        for (std::size_t i=0;i<n;++i) {
            const float a=i ? off(rng) : 0.0f;
            const float c=i+1<n ? off(rng) : 0.0f;
            const float b=2.0f+std::abs(a)+std::abs(c);
            p.lower[base+i]=a; p.diag[base+i]=b; p.upper[base+i]=c;
            p.reference[base+i]=solution(rng);
            if (i) lo[i-1]=a;
            di[i]=b;
            if (i+1<n) up[i]=c;
        }
        for (std::size_t i=0;i<n;++i) {
            double value=di[i]*p.reference[base+i];
            if (i) value+=lo[i-1]*p.reference[base+i-1];
            if (i+1<n) value+=up[i]*p.reference[base+i+1];
            p.rhs[base+i]=static_cast<float>(value);
            right[i]=p.rhs[base+i];
        }
        const auto solved=solve_thomas(lo,di,up,right);
        std::copy(solved.begin(),solved.end(),p.reference.begin()+static_cast<std::ptrdiff_t>(base));
    }
    return p;
}

struct Accuracy { double max_abs=0,max_rel=0,max_residual=0; bool passed=true; };

Accuracy assess(const HostProblem& p,const std::vector<float>& actual) {
    if (actual.size()!=p.reference.size()) throw std::invalid_argument("Output size mismatch");
    Accuracy result;
    for (std::size_t s=0;s<p.batch;++s) {
        const auto base=s*p.n;
        double residual_sq=0,matrix_fro_sq=0,x_sq=0,rhs_sq=0;
        for (std::size_t i=0;i<p.n;++i) {
            const auto at=base+i;
            const double x=actual[at],ref=p.reference[at];
            const double abs_error=std::abs(x-ref);
            result.max_abs=std::max(result.max_abs,abs_error);
            result.max_rel=std::max(result.max_rel,abs_error/std::max(std::abs(ref),1e-30));
            result.passed=result.passed && std::isfinite(x) &&
                abs_error<=1e-5+1e-4*std::abs(ref);
            double ax=double(p.diag[at])*x;
            matrix_fro_sq+=double(p.diag[at])*p.diag[at];
            if (i) {
                ax+=double(p.lower[at])*actual[at-1];
                matrix_fro_sq+=double(p.lower[at])*p.lower[at];
            }
            if (i+1<p.n) {
                ax+=double(p.upper[at])*actual[at+1];
                matrix_fro_sq+=double(p.upper[at])*p.upper[at];
            }
            const double residual=ax-p.rhs[at];
            residual_sq+=residual*residual; x_sq+=x*x;
            rhs_sq+=double(p.rhs[at])*p.rhs[at];
        }
        const double denominator=std::sqrt(matrix_fro_sq)*std::sqrt(x_sq)+std::sqrt(rhs_sq);
        const double normalized=denominator ? std::sqrt(residual_sq)/denominator : 0;
        result.max_residual=std::max(result.max_residual,normalized);
    }
    return result;
}

enum class Algorithm { thomas,pcr };
const char* name(Algorithm value) { return value==Algorithm::thomas ? "cuda_thomas" : "cuda_pcr"; }

class Workspace {
public:
    explicit Workspace(const HostProblem& p)
        : p_(p),total_(p.n*p.batch),lower_(total_),diag_(total_),upper_(total_),rhs_(total_),
          work_a_(total_),work_b_(total_),work_c_(total_),work_d_(total_),
          next_a_(total_),next_b_(total_),next_c_(total_),next_d_(total_),output_(total_) {
        copy(lower_,p.lower); copy(diag_,p.diag); copy(upper_,p.upper); copy(rhs_,p.rhs);
    }
    void launch(Algorithm algorithm) {
        constexpr unsigned threads=256;
        if (algorithm==Algorithm::thomas) {
            const auto blocks=static_cast<unsigned>((p_.batch+threads-1)/threads);
            thomas_batched_kernel<<<blocks,threads>>>(lower_.data(),diag_.data(),upper_.data(),rhs_.data(),
                work_b_.data(),work_d_.data(),output_.data(),p_.n,p_.batch);
            check_cuda(cudaGetLastError(),"CUDA Thomas launch");
            return;
        }
        check_cuda(cudaMemcpy(work_a_.data(),lower_.data(),lower_.bytes(),cudaMemcpyDeviceToDevice),"reset PCR a");
        check_cuda(cudaMemcpy(work_b_.data(),diag_.data(),diag_.bytes(),cudaMemcpyDeviceToDevice),"reset PCR b");
        check_cuda(cudaMemcpy(work_c_.data(),upper_.data(),upper_.bytes(),cudaMemcpyDeviceToDevice),"reset PCR c");
        check_cuda(cudaMemcpy(work_d_.data(),rhs_.data(),rhs_.bytes(),cudaMemcpyDeviceToDevice),"reset PCR d");
        float *a=work_a_.data(),*b=work_b_.data(),*c=work_c_.data(),*d=work_d_.data();
        float *na=next_a_.data(),*nb=next_b_.data(),*nc=next_c_.data(),*nd=next_d_.data();
        const auto blocks=static_cast<unsigned>((total_+threads-1)/threads);
        for (std::size_t offset=1;offset<p_.n;offset*=2) {
            pcr_float_stage_kernel<<<blocks,threads>>>(a,b,c,d,na,nb,nc,nd,p_.n,total_,offset);
            check_cuda(cudaGetLastError(),"CUDA PCR stage launch");
            std::swap(a,na); std::swap(b,nb); std::swap(c,nc); std::swap(d,nd);
        }
        pcr_float_solve_kernel<<<blocks,threads>>>(b,d,output_.data(),total_);
        check_cuda(cudaGetLastError(),"CUDA PCR solve launch");
    }
    std::vector<float> download() const {
        std::vector<float> result(total_);
        check_cuda(cudaMemcpy(result.data(),output_.data(),output_.bytes(),cudaMemcpyDeviceToHost),"download output");
        return result;
    }
private:
    static void copy(DeviceBuffer<float>& target,const std::vector<float>& source) {
        check_cuda(cudaMemcpy(target.data(),source.data(),target.bytes(),cudaMemcpyHostToDevice),"upload input");
    }
    const HostProblem& p_; std::size_t total_;
    DeviceBuffer<float> lower_,diag_,upper_,rhs_,work_a_,work_b_,work_c_,work_d_;
    DeviceBuffer<float> next_a_,next_b_,next_c_,next_d_,output_;
};

struct Sample { int round=0,iteration=0; Algorithm algorithm; float event_ms=0; double host_ms=0; };

double percentile(std::vector<double> values,double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(),values.end());
    const double index=(values.size()-1)*p/100;
    const auto lo=static_cast<std::size_t>(index),hi=std::min(lo+1,values.size()-1);
    return values[lo]+(values[hi]-values[lo])*(index-lo);
}

std::string csv_escape(const std::string& value) {
    std::string result="\"";
    for (char c:value) { if (c=='\"') result+='\"'; result+=c; }
    return result+'\"';
}

std::string device_metadata() {
    int device=0; check_cuda(cudaGetDevice(&device),"cudaGetDevice");
    cudaDeviceProp prop{}; check_cuda(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties");
    int driver=0,runtime=0; check_cuda(cudaDriverGetVersion(&driver),"cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime),"cudaRuntimeGetVersion");
    std::ostringstream out;
    out << "{\n  \"gpu_name\": " << csv_escape(prop.name)
        << ",\n  \"compute_capability\": \"" << prop.major << '.' << prop.minor
        << "\",\n  \"multiprocessors\": " << prop.multiProcessorCount
        << ",\n  \"cuda_driver_version\": " << driver
        << ",\n  \"cuda_runtime_version\": " << runtime
        << ",\n  \"compiler\": \"nvcc " << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
        << "\",\n  \"source_revision\": \"" << MATMUL_INSPECTOR_SOURCE_REVISION
        << "\",\n  \"build_type\": \"" << MATMUL_INSPECTOR_BUILD_TYPE
        << "\",\n  \"cuda_architectures\": \"" << MATMUL_INSPECTOR_CUDA_ARCHITECTURES
        << "\",\n  \"dtype\": \"float32\",\n  \"threads_per_block\": 256,\n"
        << "  \"build_flags\": \"CMake CUDA target defaults; inspect compile_commands.json for the exact command\"\n}\n";
    return out.str();
}

}  // namespace

int run_cuda_tridiagonal_comparison(const TridiagonalComparisonConfig& config) {
    const std::filesystem::path output=config.output;
    if (!std::filesystem::create_directories(output) || !std::filesystem::is_empty(output))
        throw std::runtime_error("Output directory exists or cannot be created: "+output.string());
    std::ofstream raw(output/"raw_samples.csv"),summary(output/"summary.csv"),correctness(output/"correctness.csv"),failures(output/"failures.csv");
    if (!raw || !summary || !correctness || !failures) throw std::runtime_error("Cannot create result files");
    raw << "algorithm,system_size,batch_size,round,iteration,event_ms,host_synchronized_ms\n";
    summary << "algorithm,system_size,batch_size,dtype,threads_per_block,rounds,iterations,median_ms,p10_ms,p90_ms,systems_per_second\n";
    correctness << "algorithm,system_size,batch_size,seed,passed,max_abs_error,max_relative_error,max_normalized_residual,rtol,atol,norm\n";
    failures << "algorithm,system_size,batch_size,seed,phase,reason\n";
    std::ofstream(output/"environment.json") << device_metadata();
    std::ofstream methodology(output/"methodology.md");
    methodology << "# Thomas versus PCR experiment methodology\n\n"
        << "Inputs are deterministic FP32 strictly diagonally dominant systems. CPU Thomas in FP64 is the reference. "
        << "CUDA Thomas maps one independent system to one thread. CUDA PCR maps one equation to one thread and launches one global kernel per doubling offset. "
        << "Both use 256 threads per block. Allocation, upload, reference generation, validation, and download are outside kernel samples. "
        << "PCR's required device-to-device working-state reset is inside its event interval because it is required for each solve. "
        << "Each sample records CUDA-event time and host wall time through event synchronization. Candidate order is shuffled deterministically per round. "
        << "The residual is `||Ax-d||_2 / (||A||_F ||x||_2 + ||d||_2)`. Pass criteria are elementwise `rtol=1e-4`, `atol=1e-5`.\n";

    for (auto n:config.system_sizes) for (auto batch:config.batch_sizes) {
        const std::uint32_t case_seed=config.seed ^ std::uint32_t(n*2654435761ULL) ^ std::uint32_t(batch);
        try {
            const auto problem=make_problem(n,batch,case_seed);
            Workspace workspace(problem);
            for (Algorithm algorithm:{Algorithm::thomas,Algorithm::pcr}) {
                workspace.launch(algorithm); check_cuda(cudaDeviceSynchronize(),"correctness synchronization");
                const auto accuracy=assess(problem,workspace.download());
                correctness << name(algorithm) << ',' << n << ',' << batch << ',' << case_seed << ','
                    << (accuracy.passed ? "true" : "false") << ',' << std::setprecision(17)
                    << accuracy.max_abs << ',' << accuracy.max_rel << ',' << accuracy.max_residual
                    << ",0.0001,0.00001,l2_frobenius\n";
                if (!accuracy.passed) {
                    failures << name(algorithm) << ',' << n << ',' << batch << ',' << case_seed
                        << ",correctness," << csv_escape("rtol/atol check failed") << '\n';
                    throw std::runtime_error(std::string(name(algorithm))+" correctness failed");
                }
                for (int i=0;i<config.warmups;++i) workspace.launch(algorithm);
                check_cuda(cudaDeviceSynchronize(),"warmup synchronization");
            }
            std::vector<Sample> samples;
            std::mt19937 order_rng(case_seed);
            Event start,stop;
            for (int round=0;round<config.rounds;++round) {
                for (int iteration=0;iteration<config.iterations;++iteration) {
                    std::vector<Algorithm> order{Algorithm::thomas,Algorithm::pcr};
                    std::shuffle(order.begin(),order.end(),order_rng);
                    for (Algorithm algorithm:order) {
                    const auto host_start=std::chrono::steady_clock::now();
                    check_cuda(cudaEventRecord(start.get()),"record start");
                    workspace.launch(algorithm);
                    check_cuda(cudaEventRecord(stop.get()),"record stop");
                    check_cuda(cudaEventSynchronize(stop.get()),"synchronize stop");
                    const auto host_stop=std::chrono::steady_clock::now();
                    float event_ms=0; check_cuda(cudaEventElapsedTime(&event_ms,start.get(),stop.get()),"event elapsed time");
                    const double host_ms=std::chrono::duration<double,std::milli>(host_stop-host_start).count();
                    samples.push_back({round,iteration,algorithm,event_ms,host_ms});
                    raw << name(algorithm) << ',' << n << ',' << batch << ',' << round << ',' << iteration
                        << ',' << std::setprecision(9) << event_ms << ',' << std::setprecision(17) << host_ms << '\n';
                    }
                }
            }
            for (Algorithm algorithm:{Algorithm::thomas,Algorithm::pcr}) {
                std::vector<double> values;
                for (const auto& sample:samples) if (sample.algorithm==algorithm) values.push_back(sample.event_ms);
                const double median=percentile(values,50),p10=percentile(values,10),p90=percentile(values,90);
                summary << name(algorithm) << ',' << n << ',' << batch << ",float32,256,"
                    << config.rounds << ',' << config.iterations << ',' << std::setprecision(17)
                    << median << ',' << p10 << ',' << p90 << ','
                    << (median ? 1000.0*batch/median : 0) << '\n';
            }
        } catch (const std::exception& error) {
            failures << "configuration," << n << ',' << batch << ',' << case_seed
                << ",execution," << csv_escape(error.what()) << '\n';
            throw;
        }
    }
    return 0;
}

}  // namespace matmul_inspector
