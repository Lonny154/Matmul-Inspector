#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace numeric {

static_assert(sizeof(float) == sizeof(std::uint32_t)
              && std::numeric_limits<float>::is_iec559
              && std::numeric_limits<float>::digits == 24,
              "Numeric helpers require IEEE-754 binary32 floats");

std::uint32_t float_to_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

std::bitset<32> float_bits(float value) {
    return std::bitset<32>(float_to_bits(value));
}

bool bitwise_equal(float actual, float expected) {
    return float_to_bits(actual) == float_to_bits(expected);
}

float absolute_error(float actual, float expected) {
    return std::fabs(actual - expected);
}

float relative_error(float actual, float expected) {
    return expected != 0.0f ? absolute_error(actual, expected) / std::fabs(expected) : 0.0f;
}

std::uint32_t float_to_ordered(float value) {
    std::uint32_t bits = float_to_bits(value);

    if (bits & 0x80000000u) {
        return ~bits;
    }

    return bits | 0x80000000u;
}

std::uint32_t ulp_distance(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<std::uint32_t>::max();
    }

    // Treat +0.0f and -0.0f as numerically identical.
    if (a == b) {
        return 0;
    }

    std::uint32_t a_ordered = float_to_ordered(a);
    std::uint32_t b_ordered = float_to_ordered(b);

    return a_ordered > b_ordered
        ? a_ordered - b_ordered
        : b_ordered - a_ordered;
}

float comparison_tolerance(
    float actual,
    float expected,
    float abs_tolerance,
    float rel_tolerance
) {
    float scale = std::max(
        std::fabs(actual),
        std::fabs(expected)
    );

    return std::max(abs_tolerance, rel_tolerance * scale);
}

bool nearly_equal(
    float actual,
    float expected,
    float abs_tolerance,
    float rel_tolerance
) {
    if (std::isnan(actual) || std::isnan(expected)) {
        return false;
    }

    if (actual == expected) {
        return true;
    }

    float difference = absolute_error(actual, expected);
    float tolerance = comparison_tolerance(
        actual, expected, abs_tolerance, rel_tolerance
    );

    return difference <= tolerance;
}

}  // namespace numeric
