#pragma once
#include <vector>

namespace benchmark {
struct Statistics {
    double mean_ms = 0, median_ms = 0, min_ms = 0, stddev_ms = 0;
};
// Population standard deviation; an empty sample set represents no launches.
Statistics statistics(const std::vector<float>& samples);
}  // namespace benchmark
