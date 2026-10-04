#include "tridiagonal_benchmark_cli.hpp"

#include <charconv>
#include <stdexcept>

namespace matmul_inspector {
namespace {

std::size_t positive_size(const std::string& text,const char* option) {
    std::size_t value=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || value==0)
        throw std::invalid_argument(std::string(option)+" must be a positive integer");
    return value;
}

std::vector<std::size_t> size_list(const std::string& text,const char* option) {
    std::vector<std::size_t> values;
    std::size_t start=0;
    while (start<text.size()) {
        const auto comma=text.find(',',start);
        values.push_back(positive_size(
            text.substr(start,comma==std::string::npos ? text.size()-start : comma-start),
            option));
        if (comma==std::string::npos) break;
        start=comma+1;
    }
    if (values.empty()) throw std::invalid_argument(std::string(option)+" must not be empty");
    return values;
}

int count(const std::string& text,const char* option) {
    int value=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::invalid_argument(std::string("Invalid value for ")+option);
    return value;
}

TridiagonalBenchmarkMode mode(const std::string& value) {
    if (value=="all") return TridiagonalBenchmarkMode::all;
    if (value=="cpu") return TridiagonalBenchmarkMode::cpu;
    if (value=="serial_host_loop") return TridiagonalBenchmarkMode::serial_host_loop;
    if (value=="true_batched_gpu") return TridiagonalBenchmarkMode::true_batched_gpu;
    if (value=="true_batched_reuse") return TridiagonalBenchmarkMode::true_batched_reuse;
    if (value=="true_batched_device_resident")
        return TridiagonalBenchmarkMode::true_batched_device_resident;
    if (value=="true_batched_hybrid") return TridiagonalBenchmarkMode::true_batched_hybrid;
    if (value=="true_batched_fused") return TridiagonalBenchmarkMode::true_batched_fused;
    throw std::invalid_argument("Invalid benchmark mode: "+value);
}

}  // namespace

TridiagonalBenchmarkConfig parse_tridiagonal_benchmark_cli(
    const std::vector<std::string>& arguments) {
    TridiagonalBenchmarkConfig config;
    for (std::size_t i=0; i<arguments.size(); ++i) {
        const auto& option=arguments[i];
        if (option=="--help") {
            config.show_help=true;
            continue;
        }
        if (i+1==arguments.size())
            throw std::invalid_argument("Missing value for "+option);
        const auto& value=arguments[++i];
        if (option=="--sizes") config.system_sizes=size_list(value,"--sizes");
        else if (option=="--batches") config.batch_sizes=size_list(value,"--batches");
        else if (option=="--system-size")
            config.system_sizes={positive_size(value,"--system-size")};
        else if (option=="--batch-size")
            config.batch_sizes={positive_size(value,"--batch-size")};
        else if (option=="--mode") config.mode=mode(value);
        else if (option=="--warmups") config.warmups=count(value,"--warmups");
        else if (option=="--iterations") config.iterations=count(value,"--iterations");
        else if (option=="--output") config.output=value;
        else throw std::invalid_argument("Unknown option: "+option);
    }
    if (config.warmups<0 || config.iterations<=0)
        throw std::invalid_argument("Invalid benchmark counts");
    return config;
}

const char* tridiagonal_benchmark_mode_name(TridiagonalBenchmarkMode value) {
    switch (value) {
    case TridiagonalBenchmarkMode::all: return "all";
    case TridiagonalBenchmarkMode::cpu: return "cpu";
    case TridiagonalBenchmarkMode::serial_host_loop: return "serial_host_loop";
    case TridiagonalBenchmarkMode::true_batched_gpu: return "true_batched_gpu";
    case TridiagonalBenchmarkMode::true_batched_reuse: return "true_batched_reuse";
    case TridiagonalBenchmarkMode::true_batched_device_resident:
        return "true_batched_device_resident";
    case TridiagonalBenchmarkMode::true_batched_hybrid: return "true_batched_hybrid";
    case TridiagonalBenchmarkMode::true_batched_fused: return "true_batched_fused";
    }
    throw std::invalid_argument("Unknown tridiagonal benchmark mode");
}

bool tridiagonal_benchmark_mode_includes(
    TridiagonalBenchmarkMode selected,TridiagonalBenchmarkMode path) {
    return selected==TridiagonalBenchmarkMode::all || selected==path;
}

std::string tridiagonal_benchmark_help() {
    return
        "Usage: tridiagonal-benchmark [OPTIONS]\n"
        "  --system-size N   Run one system size (N > 0)\n"
        "  --batch-size B    Run one system count (B > 0)\n"
        "  --sizes LIST      Run comma-separated system sizes\n"
        "  --batches LIST    Run comma-separated system counts\n"
        "  --mode MODE       all, cpu, serial_host_loop, true_batched_gpu,\n"
        "                    true_batched_reuse, true_batched_device_resident,\n"
        "                    true_batched_hybrid, true_batched_fused\n"
        "  --warmups N       Warmup iterations\n"
        "  --iterations N    Measured iterations\n"
        "  --output DIR      Result directory\n"
        "\nExample:\n"
        "  ./build/tridiagonal-benchmark --system-size 4096 --batch-size 512 \\\n"
        "    --mode true_batched_device_resident --warmups 1 --iterations 1 \\\n"
        "    --output results/ncu-4096x512\n";
}

}  // namespace matmul_inspector
