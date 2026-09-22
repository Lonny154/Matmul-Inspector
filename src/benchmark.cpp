#include "benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace benchmark {
Statistics statistics(const std::vector<float>& samples) {
    if (samples.empty()) return {};
    for (float value : samples) {
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid timing sample");
    }
    std::vector<float> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    Statistics result;
    result.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    result.min_ms = ordered.front();
    const auto middle = ordered.size() / 2;
    result.median_ms = ordered.size() % 2 ? ordered[middle]
        : (static_cast<double>(ordered[middle - 1]) + ordered[middle]) / 2;
    for (float value : samples) result.stddev_ms += (value - result.mean_ms) * (value - result.mean_ms);
    result.stddev_ms = std::sqrt(result.stddev_ms / samples.size());
    return result;
}
}  // namespace benchmark
