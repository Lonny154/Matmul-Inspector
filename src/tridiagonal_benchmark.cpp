#include "tridiagonal_benchmark.hpp"
#include "tridiagonal.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace matmul_inspector {
namespace {

using Solver = std::vector<double> (*)(
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);

TridiagonalBenchmarkResult benchmark_cpu(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups,
    const char* algorithm, Solver solver) {
    if (systems.empty() || iterations <= 0 || warmups < 0)
        throw std::invalid_argument("Invalid tridiagonal benchmark configuration");
    // Correctness is a precondition for collecting timing data.
    for (const auto& system : systems)
        verify_tridiagonal_solution(
            solver(system.lower,system.diag,system.upper,system.rhs),system.expected);
    std::vector<benchmark::Sample> samples;
    std::vector<std::vector<double>> output;
    for (int phase = 0; phase < 2; ++phase) {
        const bool warmup = phase == 0;
        const int count = warmup ? warmups : iterations;
        for (int iteration = 0; iteration < count; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            output.clear();
            output.reserve(systems.size());
            for (const auto& system : systems)
                output.push_back(solver(system.lower, system.diag, system.upper, system.rhs));
            const auto stop = std::chrono::steady_clock::now();
            samples.push_back({0, iteration, warmup,
                std::chrono::duration<float, std::milli>(stop-start).count()});
        }
    }
    for (std::size_t i=0; i<systems.size(); ++i)
        verify_tridiagonal_solution(output[i], systems[i].expected);
    benchmark::AnalysisOptions options;
    options.bootstrap_samples = 0;
    return {algorithm, "cpu", "end_to_end", "serial_host_loop",
        systems.front().diag.size(), systems.size(), warmups, iterations,
        benchmark::analyze(samples,options)};
}

}  // namespace

std::vector<TridiagonalSystem> make_benchmark_systems(
    std::size_t system_size, std::size_t batch_size) {
    if (!system_size || !batch_size)
        throw std::invalid_argument("System and batch sizes must be positive");
    std::vector<TridiagonalSystem> systems;
    systems.reserve(batch_size);
    for (std::size_t batch=0; batch<batch_size; ++batch) {
        TridiagonalSystem system;
        system.lower.assign(system_size-1,-1.0);
        system.diag.assign(system_size,4.0);
        system.upper.assign(system_size-1,-1.0);
        system.expected.resize(system_size);
        system.rhs.resize(system_size);
        for (std::size_t i=0; i<system_size; ++i)
            system.expected[i] = 1.0 + double((i + 3*batch) % 17) / 8.0;
        for (std::size_t i=0; i<system_size; ++i) {
            system.rhs[i] = system.diag[i]*system.expected[i];
            if (i) system.rhs[i] += system.lower[i-1]*system.expected[i-1];
            if (i+1<system_size) system.rhs[i] += system.upper[i]*system.expected[i+1];
        }
        systems.push_back(std::move(system));
    }
    return systems;
}

void verify_tridiagonal_solution(const std::vector<double>& actual,
                                 const std::vector<double>& expected,
                                 double tolerance) {
    if (actual.size()!=expected.size() || !std::isfinite(tolerance) || tolerance<0)
        throw std::invalid_argument("Invalid tridiagonal verification input");
    for (std::size_t i=0; i<actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>tolerance)
            throw std::runtime_error("Tridiagonal solution verification failed at index "+std::to_string(i));
}

TridiagonalBenchmarkResult benchmark_thomas_batch(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    return benchmark_cpu(systems,iterations,warmups,"thomas",solve_thomas);
}

TridiagonalBenchmarkResult benchmark_cpu_pcr_batch(
    const std::vector<TridiagonalSystem>& systems, int iterations, int warmups) {
    return benchmark_cpu(systems,iterations,warmups,"pcr",solve_pcr);
}

std::string tridiagonal_benchmark_csv_header() {
    return "algorithm,backend,timing_scope,batch_execution,system_size,batch_size,warmups,iterations,median_ms,mean_ms,min_ms,max_ms,selected_path,dispatch_rule\n";
}

std::string tridiagonal_benchmark_csv_row(const TridiagonalBenchmarkResult& r) {
    std::ostringstream out;
    out << std::setprecision(17) << r.algorithm << ',' << r.backend << ',' << r.timing_scope
        << ',' << r.batch_execution << ',' << r.system_size << ',' << r.batch_size
        << ',' << r.warmups << ',' << r.iterations
        << ',' << r.timing.median_ms << ',' << r.timing.mean_ms << ',' << r.timing.min_ms
        << ',' << r.timing.max_ms << ',' << r.selected_path << ',' << r.dispatch_rule << '\n';
    return out.str();
}

}  // namespace matmul_inspector
