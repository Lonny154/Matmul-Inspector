#pragma once
#include "matrix.hpp"
#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace benchmark {
struct Sample {
    int trial = 0, iteration = 0;
    bool warmup = false;
    float latency_ms = 0;
};
struct AnalysisOptions {
    int bootstrap_samples = 1000;
    double confidence_level = 0.95;
    std::uint32_t seed = 42;
    std::vector<int> percentiles{5,25,75,95};
};
struct Statistics {
    // Keep the original aggregate initializer/API and mean-based artifact fields.
    double mean_ms = 0, median_ms = 0, min_ms = 0, stddev_ms = 0;
    double max_ms = 0, iqr_ms = 0, mad_ms = 0;
    std::optional<double> cv{}, median_ci_low_ms{}, median_ci_high_ms{};
    std::map<int,double> percentiles{};
    std::vector<std::string> warnings{};
    std::vector<Sample> samples{};
};
// Linear interpolation at (n-1)*p/100 on sorted finite nonnegative samples.
double percentile(std::vector<double> values, double p);
// Strict analysis rejects empty measurement sets and malformed sample identities.
Statistics analyze(const std::vector<Sample>& samples, const AnalysisOptions& options = {});
// Legacy empty-set behavior remains zero statistics; no bootstrap at capture time.
Statistics statistics(const std::vector<float>& samples, const std::vector<float>& warmups = {});
struct HostMeasurement { Matrix output; Statistics timing; };
// Times a synchronous host operation including its returned output allocation.
// Previous-output destruction, sample recording and all analysis are outside.
template<class Operation>
HostMeasurement measure_host(Operation operation, int repetitions, int warmups) {
    if (repetitions <= 0 || warmups < 0) throw std::invalid_argument("Invalid benchmark counts");
    std::vector<float> samples, warmup_samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    warmup_samples.reserve(static_cast<std::size_t>(warmups));
    Matrix output(0, 0);
    for (int phase = 0; phase < 2; ++phase) {
        const int count = phase == 0 ? warmups : repetitions;
        for (int i = 0; i < count; ++i) {
            const auto start = std::chrono::steady_clock::now();
            Matrix next = operation();
            const auto stop = std::chrono::steady_clock::now();
            (phase == 0 ? warmup_samples : samples).push_back(
                std::chrono::duration<float, std::milli>(stop - start).count());
            output = std::move(next);
        }
    }
    return {std::move(output), statistics(samples, warmup_samples)};
}
HostMeasurement cpu_matmul(const Matrix& a, const Matrix& b, int repetitions = 20, int warmups = 3);
}  // namespace benchmark
