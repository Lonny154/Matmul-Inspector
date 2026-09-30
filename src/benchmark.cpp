#include "benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

namespace benchmark {
namespace {
double quantile(const std::vector<double>& ordered, double p) {
    const double index = (ordered.size()-1) * p / 100;
    const auto lo = static_cast<std::size_t>(index);
    const auto hi = std::min(lo+1, ordered.size()-1);
    return ordered[lo] + (ordered[hi]-ordered[lo]) * (index-lo);
}
void validate(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("Empty timing samples");
    for (double value : values)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid timing sample");
}
// Standardized mt19937 words + rejection sampling avoid implementation-specific
// uniform_int_distribution mappings. Sample populations must fit uint32_t.
std::size_t draw(std::mt19937& rng, std::size_t count) {
    if (!count || count > UINT32_MAX) throw std::invalid_argument("Bootstrap population too large");
    const std::uint64_t range = std::uint64_t{1} << 32;
    const auto limit = range - range % count;
    std::uint32_t value;
    do { value = rng(); } while (value >= limit);
    return value % count;
}
} // namespace

double percentile(std::vector<double> values, double p) {
    validate(values);
    if (!std::isfinite(p) || p < 0 || p > 100) throw std::invalid_argument("Invalid percentile");
    std::sort(values.begin(), values.end());
    return quantile(values,p);
}

Statistics analyze(const std::vector<Sample>& samples, const AnalysisOptions& options) {
    if (options.bootstrap_samples < 0 || !std::isfinite(options.confidence_level)
        || options.confidence_level <= 0 || options.confidence_level >= 1)
        throw std::invalid_argument("Invalid bootstrap settings");
    for (int p : options.percentiles) if (p < 0 || p > 100) throw std::invalid_argument("Invalid percentile");
    std::map<int,std::vector<double>> trials;
    std::map<std::pair<int,bool>,int> next_iteration;
    std::set<int> started;
    std::vector<double> values;
    for (const auto& sample : samples) {
        if (sample.trial < 0 || sample.iteration < 0 || !std::isfinite(sample.latency_ms) || sample.latency_ms < 0
            || sample.iteration != next_iteration[{sample.trial,sample.warmup}]++)
            throw std::invalid_argument("Malformed timing sample or iteration sequence");
        if (sample.warmup && started.count(sample.trial)) throw std::invalid_argument("Warmup after measurement");
        trials[sample.trial];
        if (!sample.warmup) {
            started.insert(sample.trial);
            trials[sample.trial].push_back(sample.latency_ms);
            values.push_back(sample.latency_ms);
        }
    }
    validate(values);
    std::vector<std::vector<double>> blocks;
    for (const auto& entry : trials) {
        if (entry.second.empty()) throw std::invalid_argument("Trial has no measurements");
        blocks.push_back(entry.second);
    }
    Statistics result;
    result.samples = samples;
    result.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    auto ordered = values;
    std::sort(ordered.begin(),ordered.end());
    result.min_ms = ordered.front(); result.max_ms = ordered.back();
    result.median_ms = quantile(ordered,50);
    result.iqr_ms = quantile(ordered,75)-quantile(ordered,25);
    for (double v : values) result.stddev_ms += (v-result.mean_ms)*(v-result.mean_ms);
    result.stddev_ms = std::sqrt(result.stddev_ms/values.size());
    if (result.mean_ms > 0) result.cv = result.stddev_ms/result.mean_ms;
    std::vector<double> deviations;
    for (double v : values) deviations.push_back(std::fabs(v-result.median_ms));
    result.mad_ms = percentile(deviations,50);
    for (int p : options.percentiles) result.percentiles[p] = quantile(ordered,p);
    auto warn = [&](const std::string& message) { result.warnings.push_back(message); };
    if (values.size() < 10) warn("fewer_than_10_measurements");
    if (blocks.size() < 5) warn("fewer_than_5_trials");
    if (result.cv && *result.cv > 0.20) warn("cv_gt_20_percent");
    if (result.min_ms == 0) warn("zero_latency_timer_resolution");
    else if (result.max_ms/result.min_ms > 3) warn("max_min_ratio_gt_3");
    if (result.median_ms > 0 && result.iqr_ms/result.median_ms > 0.25) warn("iqr_median_gt_25_percent");
    for (const auto& entry : trials) {
        const auto& trial = entry.second;
        if (trial.size() >= 5) {
            const double rest = percentile(std::vector<double>(trial.begin()+1,trial.end()),50);
            if (rest > 0 && std::fabs(trial.front()/rest-1) > 0.5)
                warn("trial_"+std::to_string(entry.first)+"_first_iteration_deviation_gt_50_percent");
        }
        if (trial.size() >= 8) {
            const auto window = trial.size()/4;
            const double first = percentile(std::vector<double>(trial.begin(),trial.begin()+window),50);
            const double last = percentile(std::vector<double>(trial.end()-window,trial.end()),50);
            if (first > 0 && std::fabs(last/first-1) > 0.2)
                warn("trial_"+std::to_string(entry.first)+"_end_to_start_drift_gt_20_percent");
        }
    }
    if (options.bootstrap_samples && values.size() >= 2) {
        std::mt19937 rng(options.seed);
        std::vector<double> medians;
        medians.reserve(options.bootstrap_samples);
        std::vector<double> resampled;
        for (int b = 0; b < options.bootstrap_samples; ++b) {
            resampled.clear();
            if (blocks.size() > 1) {
                // Resample whole trials, preserving within-trial correlation.
                for (std::size_t t = 0; t < blocks.size(); ++t) {
                    const auto& block = blocks[draw(rng,blocks.size())];
                    resampled.insert(resampled.end(),block.begin(),block.end());
                }
            } else {
                for (std::size_t i = 0; i < values.size(); ++i) resampled.push_back(values[draw(rng,values.size())]);
            }
            medians.push_back(percentile(resampled,50));
        }
        const double tail = (1-options.confidence_level)*50;
        result.median_ci_low_ms = percentile(medians,tail);
        result.median_ci_high_ms = percentile(medians,100-tail);
    }
    return result;
}

Statistics statistics(const std::vector<float>& samples, const std::vector<float>& warmups) {
    if (samples.empty() && warmups.empty()) return {};
    std::vector<Sample> captured;
    for (std::size_t i=0; i<warmups.size(); ++i) captured.push_back({0,static_cast<int>(i),true,warmups[i]});
    for (std::size_t i=0; i<samples.size(); ++i) captured.push_back({0,static_cast<int>(i),false,samples[i]});
    AnalysisOptions options; options.bootstrap_samples = 0;
    return analyze(captured,options);
}
HostMeasurement cpu_matmul(const Matrix& a, const Matrix& b, int repetitions, int warmups) {
    return measure_host([&] { return matmul(a,b); },repetitions,warmups);
}
} // namespace benchmark
