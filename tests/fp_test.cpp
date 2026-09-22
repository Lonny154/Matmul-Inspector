#include "experiment.hpp"
#include "numeric.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
void aggregates() {
    Matrix a(1,8,10), b(1,8,11);
    const unsigned distances[] = {0,1,2,3,4,5,8,9};
    for (unsigned col = 0; col < 8; ++col) {
        a(0,col) = b(0,col) = -1;
        for (unsigned step = 0; step < distances[col]; ++step)
            b(0,col) = std::nextafter(b(0,col), -2.0f);
    }
    auto r = comparison::compare(a,b,0,0,2);
    check(r.divergent_count == 7 && r.tolerance_failures == 7, "divergences and tolerance count");
    check(r.divergent_percent == 87.5 && r.tolerance_failure_percent == 87.5, "percent denominators");
    check(r.max_ulp == 9 && r.mean_divergent_ulp == 32.0/7, "max and mean negative ULP");
    check(r.ulp_bins == std::array<std::size_t,6>{0,1,1,2,2,1}, "ULP bins");
    check(r.max_absolute_error == 9 * 0x1p-23 && r.max_relative_error == 9 * 0x1p-23, "maximum errors");
    check(r.mismatches.size() == 2 && r.mismatches_truncated && r.mismatch_count == 7, "bounded sample");
    check(r.first_bitwise->col == 1 && r.first_numeric->col == 1, "first mismatch retained");
    r = comparison::compare(a,b,1e-5f,0,0);
    check(r.tolerance_failures == 0 && r.mismatches.empty() && r.mismatches_truncated, "absolute tolerance and zero cap");
    r = comparison::compare(a,b,0,1e-5f);
    check(r.tolerance_pass, "relative tolerance");
}
void special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Matrix a(1,6), b(1,6);
    b(0,0) = -0.0f;
    a(0,1) = b(0,1) = nan;
    a(0,2) = b(0,2) = inf;
    a(0,3) = inf; b(0,3) = -inf;
    a(0,4) = inf; b(0,4) = 1;
    b(0,5) = 1;
    auto r = comparison::compare(a,b);
    check(r.nan_pairs == 1 && r.infinity_pairs == 3 && r.finite_pairs == 2, "nonfinite classification");
    check(r.divergent_count == 4 && r.tolerance_failures == 4, "NaN fails even with identical bits; unequal infinity fails");
    check(r.ulp_bins[0] == 1 && r.finite_divergent_count == 2, "signed zero divergent but zero ULP");
    check(r.zero_reference_nonzero == 1 && r.max_relative_error == 0, "explicit legacy zero denominator convention");
    check(r.first_bitwise->col == 0 && r.first_numeric->col == 1, "independent firsts for signed zero and NaN");
    check(numeric::ulp_distance(-std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min()) == 3,
          "existing ordered mapping includes signed zero gap across signs");
    auto empty = comparison::compare(Matrix(0,0), Matrix(0,0));
    check(empty.mean_divergent_ulp == 0 && empty.divergent_percent == 0, "empty aggregates");
}
void configuration() {
    for (const auto* kernel : {"cuda-naive-fma", "cuda-naive-no-fma", "cuda-naive-reordered"}) {
        auto c = experiment::parse({"benchmark", "--candidate", kernel, "--input", "cancellation", "--max-mismatches", "5"});
        check(c.candidate == kernel && c.max_mismatches == 5, "variant CLI selection");
        check(experiment::metadata_json(c, {{"fp_verified", "false"}}).find("\"fp_verified\": false") != std::string::npos, "verification serialized boolean");
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
        {"compare","--input","bad"}, {"compare","--max-mismatches","1001"},
        {"compare","--input","cancellation","--sizes","2"}}) {
        bool rejected = false;
        try { experiment::parse(args); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid experiment options rejected");
    }
    Matrix a(2,4,6), b(4,3,5), a2(2,4), b2(4,3);
    experiment::Config c; c.input = "cancellation";
    experiment::fill_inputs(a,b,c); c.seed = 999; experiment::fill_inputs(a2,b2,c);
    check(comparison::compare(a,a2).divergent_count == 0 && comparison::compare(b,b2).divergent_count == 0, "fixture deterministic independent of seeds/layout");
    check(a(0,0) == 16777216 && a(0,2) == -16777216 && a.data()[4] == 0, "cancellation fixture and padding");
    c.input = "fma-sensitive"; experiment::fill_inputs(a,b,c);
    check(std::fma(a(0,1), b(1,0), -1.0f) == -0x1p-46f, "FMA sensitive fixture golden residual");
}
}
int main() { aggregates(); special_values(); configuration(); return failures ? 1 : 0; }
