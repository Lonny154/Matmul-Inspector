#pragma once

#include <bitset>
#include <cstdint>

namespace numeric {

// These helpers operate on IEEE-754 binary32 floats.
std::uint32_t float_to_bits(float value);
std::bitset<32> float_bits(float value);
bool bitwise_equal(float actual, float expected);
float absolute_error(float actual, float expected);
// Uses the expected value as the reference; returns zero when expected is zero
// to preserve Inspector's original reporting convention.
float relative_error(float actual, float expected);

// Monotonic bit ordering for non-NaN values; signed zeros have adjacent keys.
std::uint32_t float_to_ordered(float value);

// Equal values (including signed zeros) have distance zero. If either operand
// is NaN, returns std::numeric_limits<std::uint32_t>::max().
std::uint32_t ulp_distance(float a, float b);

// max(abs_tolerance, rel_tolerance * max(abs(actual), abs(expected))).
float comparison_tolerance(
    float actual,
    float expected,
    float abs_tolerance,
    float rel_tolerance
);

// NaNs never compare equal. Exact equality is accepted first; otherwise the
// absolute difference must be <= comparison_tolerance().
bool nearly_equal(
    float actual,
    float expected,
    float abs_tolerance,
    float rel_tolerance
);

}  // namespace numeric
