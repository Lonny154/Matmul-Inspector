#include "inspector.hpp"
#include "numeric.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

std::string report(const Matrix& expected, const Matrix& actual, bool errors = false) {
    std::ostringstream output;
    auto& stream = errors ? std::cerr : std::cout;
    const auto flags = stream.flags();
    const auto precision = stream.precision();
    auto* original = stream.rdbuf(output.rdbuf());
    Inspector::compare_results(expected, actual);
    stream.rdbuf(original);
    stream.flags(flags);
    stream.precision(precision);
    return output.str();
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void test_diagnostics() {
    Matrix cpu(2, 2, 3), gpu(2, 2, 5);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 2; ++col) {
            cpu(row, col) = gpu(row, col) = 1.0f;
        }
    }
    cpu.data()[2] = 123.0f;
    gpu.data()[2] = -456.0f;
    check(report(cpu, gpu) == "numeric: MATCH\nbitwise: MATCH\nNo numeric mismatches found\nNo bitwise mismatches found\n",
          "comparison ignores different strides and padding");

    gpu(0, 1) = std::nextafter(1.0f, 2.0f);
    auto text = report(cpu, gpu);
    check(contains(text, "numeric: MATCH\nbitwise: MISMATCH\n"), "one ULP need not be a numerical mismatch");
    check(contains(text, "FIRST BITWISE MISMATCH\nC[0,1]\n"), "first bitwise coordinate");
    check(contains(text, "ULP distance:   1\n"), "ULP diagnostic");
    check(contains(text, "Bitwise differences are within numerical tolerance."), "rounding distinction explained");
    check(contains(text, "absolute error:") && contains(text, "relative error:")
          && contains(text, "tolerance:") && contains(text, "actual bits:"), "numeric diagnostics present");

    gpu(1, 0) = 1.25f;
    gpu(1, 1) = 3.0f;
    text = report(cpu, gpu);
    check(contains(text, "numeric: MISMATCH\nbitwise: MISMATCH\n"), "numerical mismatch summary");
    check(contains(text, "FIRST NUMERIC MISMATCH\nC[1,0]\n"), "first numerical coordinate independent of bitwise");
    check(contains(text, "FIRST BITWISE MISMATCH\nC[0,1]\n"), "first bitwise coordinate retained");
    check(contains(text, "absolute error: 0.25\n") && contains(text, "relative error: 0.25\n"),
          "error diagnostics use CPU reference");

    Matrix positive_zero(1, 1), negative_zero(1, 1);
    negative_zero(0, 0) = -0.0f;
    text = report(positive_zero, negative_zero);
    check(contains(text, "numeric: MATCH\nbitwise: MISMATCH\n")
          && contains(text, "ULP distance:   0\n"), "signed zero reporting");

    negative_zero(0, 0) = std::numeric_limits<float>::quiet_NaN();
    text = report(positive_zero, negative_zero);
    check(contains(text, "numeric: MISMATCH\n"), "NaN is numerical mismatch");
    check(contains(text, "ULP distance:   4294967295\n"), "NaN ULP sentinel reported");

    check(contains(report(cpu, Matrix(1, 2), true), "different dimensions"), "reject different dimensions");
    check(contains(report(Matrix(0, 2), Matrix(0, 2, 4)), "numeric: MATCH\nbitwise: MATCH\n"), "empty results match");
}

void test_numeric_helpers() {
    check(numeric::bitwise_equal(1.0f, 1.0f), "identical bits");
    check(!numeric::bitwise_equal(0.0f, -0.0f), "signed zero bits differ");
    check(numeric::absolute_error(-2.0f, -4.0f) == 2.0f, "absolute error helper");
    check(numeric::relative_error(-2.0f, -4.0f) == 0.5f, "relative error uses reference magnitude");
    check(numeric::relative_error(2.0f, 0.0f) == 0.0f, "preserve zero-reference reporting convention");
}

}  // namespace

int main() {
    test_diagnostics();
    test_numeric_helpers();
    return failures == 0 ? 0 : 1;
}
