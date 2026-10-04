#include "tridiagonal_benchmark.hpp"
#ifdef MATMUL_INSPECTOR_HAS_CUDA
#include "cuda_tridiagonal_benchmark.hpp"
#endif

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::size_t> list(const std::string& text) {
    std::vector<std::size_t> values;
    std::size_t start=0;
    while (start<text.size()) {
        const auto comma=text.find(',',start);
        const auto token=text.substr(start,comma==std::string::npos ? text.size()-start : comma-start);
        std::size_t value=0;
        const auto parsed=std::from_chars(token.data(),token.data()+token.size(),value);
        if (parsed.ec!=std::errc{} || parsed.ptr!=token.data()+token.size() || value==0)
            throw std::invalid_argument("Sizes must be comma-separated positive integers");
        values.push_back(value);
        if (comma==std::string::npos) break;
        start=comma+1;
    }
    if (values.empty()) throw std::invalid_argument("Size list must not be empty");
    return values;
}
int count(const std::string& text) {
    int value=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::invalid_argument("Invalid count");
    return value;
}
}  // namespace

int main(int argc,char** argv) {
    try {
        std::vector<std::size_t> sizes{32,64,128,256,512,1024,2048,4096};
        std::vector<std::size_t> batches{1,8,32,128,512};
        int warmups=3,iterations=20;
        std::filesystem::path output="results/tridiagonal-baseline";
        for (int i=1;i<argc;++i) {
            const std::string option=argv[i];
            if (option=="--help") {
                std::cout << "Usage: tridiagonal-benchmark [--sizes LIST] [--batches LIST] "
                             "[--warmups N] [--iterations N] [--output DIRECTORY]\n"
                             "  --batches is the number of systems processed serially per sample; "
                             "it is not concurrent GPU batching.\n";
                return 0;
            }
            if (i+1==argc) throw std::invalid_argument("Missing option value");
            const std::string value=argv[++i];
            if (option=="--sizes") sizes=list(value);
            else if (option=="--batches") batches=list(value);
            else if (option=="--warmups") warmups=count(value);
            else if (option=="--iterations") iterations=count(value);
            else if (option=="--output") output=value;
            else throw std::invalid_argument("Unknown option: "+option);
        }
        if (warmups<0 || iterations<=0) throw std::invalid_argument("Invalid benchmark counts");
        if (!std::filesystem::create_directories(output) || !std::filesystem::is_empty(output))
            throw std::runtime_error("Output directory exists or cannot be created: "+output.string());
        std::ofstream csv(output/"summary.csv");
        if (!csv) throw std::runtime_error("Cannot create summary.csv");
        csv << matmul_inspector::tridiagonal_benchmark_csv_header();
        for (const auto n : sizes) for (const auto batch : batches) {
            const auto systems=matmul_inspector::make_benchmark_systems(n,batch);
            csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                       matmul_inspector::benchmark_thomas_batch(systems,iterations,warmups))
                << matmul_inspector::tridiagonal_benchmark_csv_row(
                       matmul_inspector::benchmark_cpu_pcr_batch(systems,iterations,warmups));
#ifdef MATMUL_INSPECTOR_HAS_CUDA
            csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                       matmul_inspector::benchmark_cuda_pcr_kernel_only(systems,iterations,warmups))
                << matmul_inspector::tridiagonal_benchmark_csv_row(
                       matmul_inspector::benchmark_cuda_pcr_end_to_end(systems,iterations,warmups));
#endif
            std::cout << "N=" << n << " B=" << batch << " complete\n";
        }
        std::ofstream methodology(output/"methodology.txt");
        methodology << "CPU: steady_clock around a full batch solve, including output/work allocations.\n"
                    << "batch_size: number of independent systems processed sequentially by a host loop; it does not mean simultaneous GPU execution.\n"
                    << "CUDA kernel_only: CUDA events around the existing PCR stage and solve kernels for the full batch; allocation, coefficient reset copies, and result copies excluded.\n"
                    << "CUDA end_to_end: steady_clock around the existing solve_pcr_cuda call for every system, including allocation, H2D, kernels, synchronization, D2H, and cleanup.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tridiagonal-benchmark: " << error.what() << '\n';
        return 2;
    }
}
