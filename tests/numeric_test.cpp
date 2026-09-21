#include "numeric.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

// Keep checks active in Release builds, where assert() would be disabled.
void check(bool condition, const char* description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

void test_bits_and_ordering() {
    check(numeric::float_to_bits(0.0f) == 0x00000000u, "positive zero bits");
    check(numeric::float_to_bits(-0.0f) == 0x80000000u, "negative zero bits");
    check(numeric::float_to_bits(1.0f) == 0x3f800000u, "positive float bits");
    check(numeric::float_to_bits(-1.0f) == 0xbf800000u, "negative float bits");
    check(numeric::float_bits(1.0f).to_string()
              == "00111111100000000000000000000000", "IEEE-754 bit display");
    check(numeric::float_bits(-0.0f).to_string()
              == "10000000000000000000000000000000", "signed zero bit display");

    const float values[] = {-2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 1; i < 6; ++i) {
        check(numeric::float_to_ordered(values[i - 1])
                  < numeric::float_to_ordered(values[i]), "sign-aware ordering");
    }
}

void test_ulp_distance() {
    check(numeric::ulp_distance(0.0f, -0.0f) == 0, "signed zero ULP distance");
    check(numeric::ulp_distance(-0.0f, 0.0f) == 0, "reversed signed zero distance");

    for (float value : {1.0f, -1.0f}) {
        for (float direction : {-2.0f, 2.0f}) {
            float adjacent = std::nextafter(value, direction);
            check(numeric::ulp_distance(value, adjacent) == 1,
                  "adjacent floats have ULP distance one");
            check(numeric::ulp_distance(adjacent, value) == 1,
                  "ULP distance is symmetric");
        }
    }

    const float infinity = std::numeric_limits<float>::infinity();
    for (float value : {0.0f, -0.0f, 1.0f, -1.0f, 123.5f, infinity, -infinity}) {
        check(numeric::ulp_distance(value, value) == 0, "identical float distance");
    }

    const float tiny = std::numeric_limits<float>::denorm_min();
    check(numeric::ulp_distance(0.0f, tiny) == 1, "positive subnormal distance");
    check(numeric::ulp_distance(-0.0f, -tiny) == 1, "negative subnormal distance");
    // Preserve the original ordering's separate keys for the two signed zeros.
    check(numeric::ulp_distance(-tiny, tiny) == 3, "cross-zero ordering distance");
}

void test_nan() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto sentinel = std::numeric_limits<std::uint32_t>::max();
    for (float value : {0.0f, 1.0f, -1.0f, nan}) {
        check(numeric::ulp_distance(nan, value) == sentinel, "NaN ULP sentinel");
        check(numeric::ulp_distance(value, nan) == sentinel, "second operand NaN ULP sentinel");
        check(!numeric::nearly_equal(nan, value, 1.0f, 1.0f), "NaN never nearly equal");
        check(!numeric::nearly_equal(value, nan, 1.0f, 1.0f), "second operand NaN never nearly equal");
    }
}

void test_tolerances() {
    check(numeric::nearly_equal(1.0f, 1.0f, 0.0f, 0.0f), "exact equality");
    check(numeric::nearly_equal(0.0f, -0.0f, 0.0f, 0.0f), "signed zero equality");
    check(!numeric::nearly_equal(1.0f, std::nextafter(1.0f, 2.0f), 0.0f, 0.0f),
          "zero tolerances reject distinct values");

    check(numeric::nearly_equal(0.0625f, 0.0f, 0.125f, 0.0f), "within absolute tolerance");
    check(numeric::nearly_equal(-0.125f, 0.0f, 0.125f, 0.0f), "absolute boundary inclusive");
    check(!numeric::nearly_equal(0.25f, 0.0f, 0.125f, 0.0f), "outside absolute tolerance");

    check(numeric::nearly_equal(8.0f, 7.5f, 0.0f, 0.125f), "within relative tolerance");
    check(numeric::nearly_equal(8.0f, 7.0f, 0.0f, 0.125f), "relative boundary inclusive");
    check(numeric::nearly_equal(7.0f, 8.0f, 0.0f, 0.125f), "relative scale uses larger operand");
    check(numeric::nearly_equal(-8.0f, -7.0f, 0.0f, 0.125f), "relative scale uses magnitudes");
    check(!numeric::nearly_equal(8.0f, 6.0f, 0.0f, 0.125f), "outside relative tolerance");

    check(numeric::comparison_tolerance(0.125f, 0.0f, 0.125f, 0.125f) == 0.125f,
          "absolute tolerance dominates near zero");
    check(numeric::nearly_equal(0.125f, 0.0f, 0.125f, 0.125f), "combined absolute tolerance");
    check(numeric::comparison_tolerance(8.0f, 7.0f, 0.125f, 0.125f) == 1.0f,
          "relative tolerance dominates at larger scale");
    check(numeric::nearly_equal(8.0f, 7.0f, 0.125f, 0.125f), "combined relative tolerance");
    check(!numeric::nearly_equal(8.0f, 6.5f, 1.0f, 0.125f), "tolerances use maximum, not sum");

    const float infinity = std::numeric_limits<float>::infinity();
    check(numeric::nearly_equal(infinity, infinity, 0.0f, 0.0f), "equal infinities match");
    // Retain the existing arithmetic for unequal infinities as well.
    check(numeric::nearly_equal(infinity, 1.0f, 0.0f, 0.125f), "infinite relative tolerance");
    check(!numeric::nearly_equal(infinity, 1.0f, 0.0f, 0.0f), "infinite absolute difference");
}

}  // namespace

int main() {
    test_bits_and_ordering();
    test_ulp_distance();
    test_nan();
    test_tolerances();
    return failures == 0 ? 0 : 1;
}
