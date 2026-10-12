#include "cuda_tridiagonal_comparison.hpp"

#include <charconv>
#include <iostream>
#include <stdexcept>

namespace matmul_inspector {
namespace {
template<class T> T number(const std::string& text,const char* option) {
    T value{}; const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::invalid_argument(std::string("Invalid value for ")+option);
    return value;
}
std::vector<std::size_t> sizes(const std::string& text,const char* option) {
    std::vector<std::size_t> result; std::size_t begin=0;
    while (begin<text.size()) {
        const auto comma=text.find(',',begin);
        const auto value=number<std::size_t>(text.substr(begin,comma==std::string::npos ? text.size()-begin : comma-begin),option);
        if (!value) throw std::invalid_argument(std::string(option)+" values must be positive");
        result.push_back(value); if (comma==std::string::npos) break; begin=comma+1;
    }
    if (result.empty()) throw std::invalid_argument(std::string(option)+" must not be empty");
    return result;
}
}  // namespace

TridiagonalComparisonConfig parse_tridiagonal_comparison_cli(const std::vector<std::string>& arguments) {
    TridiagonalComparisonConfig config;
    for (std::size_t i=0;i<arguments.size();++i) {
        const auto& option=arguments[i];
        if (option=="--help") { config.show_help=true; continue; }
        if (++i==arguments.size()) throw std::invalid_argument("Missing value for "+option);
        const auto& value=arguments[i];
        if (option=="--sizes") config.system_sizes=sizes(value,"--sizes");
        else if (option=="--batches") config.batch_sizes=sizes(value,"--batches");
        else if (option=="--warmups") config.warmups=number<int>(value,"--warmups");
        else if (option=="--rounds") config.rounds=number<int>(value,"--rounds");
        else if (option=="--iterations") config.iterations=number<int>(value,"--iterations");
        else if (option=="--seed") config.seed=number<std::uint32_t>(value,"--seed");
        else if (option=="--output") config.output=value;
        else throw std::invalid_argument("Unknown option: "+option);
    }
    if (config.warmups<0 || config.rounds<=0 || config.iterations<=0 || config.output.empty())
        throw std::invalid_argument("Invalid comparison configuration");
    return config;
}

std::string tridiagonal_comparison_help() {
    return "Usage: tridiagonal-comparison [OPTIONS]\n"
        "  --sizes LIST       System sizes (default: 128,256,512,1024)\n"
        "  --batches LIST     Batch counts (default: 1,32,1024)\n"
        "  --warmups N        Warmups per candidate (default: 5)\n"
        "  --rounds N         Interleaved rounds (default: 5)\n"
        "  --iterations N     Samples per candidate per round (default: 20)\n"
        "  --seed N           Deterministic base seed\n"
        "  --output DIR       New result directory\n";
}
}  // namespace matmul_inspector

int main(int argc,char** argv) {
    try {
        std::vector<std::string> arguments;
        for (int i=1;i<argc;++i) arguments.emplace_back(argv[i]);
        const auto config=matmul_inspector::parse_tridiagonal_comparison_cli(arguments);
        if (config.show_help) { std::cout << matmul_inspector::tridiagonal_comparison_help(); return 0; }
        return matmul_inspector::run_cuda_tridiagonal_comparison(config);
    } catch (const std::exception& error) {
        std::cerr << "tridiagonal-comparison: " << error.what() << '\n'; return 2;
    }
}
