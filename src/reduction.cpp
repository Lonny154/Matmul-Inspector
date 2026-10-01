#include "reduction.hpp"
#include "numeric.hpp"
#include <cmath>
#include <stdexcept>

namespace reduction {
bool fixture_supported(const std::string& name) {
    return name == "random" || name == "random_uniform" || name == "ascending_magnitude"
        || name == "descending_magnitude" || name == "alternating_sign" || name == "cancellation"
        || name == "large_dynamic_range" || name == "repeated_small_plus_large";
}
void fill_structured(Matrix& a, Matrix& b, operation::Kind kind, const std::string& fixture, std::uint32_t seed) {
    operation::validate_vectors(kind,a,b);
    if (!fixture_supported(fixture) || fixture == "random" || fixture == "random_uniform")
        throw std::invalid_argument("Expected a structured vector fixture");
    const auto length=a.cols();
    for (std::size_t i=0;i<length;++i) {
        float value=0;
        if (fixture == "ascending_magnitude" || fixture == "descending_magnitude") {
            // Monotone powers 2^-20 ... 2^20, same multiset in opposite order.
            const auto index=fixture == "ascending_magnitude" ? i : length-1-i;
            const int exponent=length == 1 ? 0 : -20+static_cast<int>(40.0*double(index)/double(length-1));
            value=std::ldexp(1.0f,exponent);
        } else if (fixture == "alternating_sign") {
            value=(i%2 ? -1.0f : 1.0f)*(1.0f+float(i%7)/8.0f);
        } else if (fixture == "cancellation") {
            const auto index=i%257;
            value=index == 0 ? 16777216.0f : index == 256 ? -16777216.0f : 1.0f;
        } else if (fixture == "repeated_small_plus_large") {
            value=i == 0 ? 16777216.0f : 1.0f;
        } else {
            seed=1664525u*seed+1013904223u;
            value=std::ldexp((seed & 1u) ? -1.0f : 1.0f, int((seed>>8)%41)-20);
        }
        a(0,i)=value;
        if (kind == operation::Kind::dot) b(i,0)=1.0f;
    }
}
double reference_fp64(operation::Kind kind, const Matrix& a, const Matrix& b) {
    operation::validate_vectors(kind,a,b);
    double sum=0;
    for (std::size_t i=0;i<a.cols();++i)
        sum += kind == operation::Kind::dot ? double(a(0,i))*double(b(i,0)) : double(a(0,i));
    return sum;
}
Observation observe(float value, float baseline, double fp64, float atol, float rtol, std::optional<float> forward) {
    Observation result;
    result.value=value; result.baseline=baseline; result.fp32_forward=forward.value_or(baseline); result.fp64_reference=fp64;
    result.tolerance_pass=numeric::nearly_equal(value,baseline,atol,rtol);
    if (std::isfinite(value) && std::isfinite(baseline)) result.ulp=numeric::ulp_distance(value,baseline);
    if (std::isfinite(value) && std::isfinite(fp64)) {
        result.absolute_error=std::fabs(double(value)-fp64);
        if (std::fabs(fp64)>relative_floor) result.relative_error=*result.absolute_error/std::fabs(fp64);
    }
    return result;
}
std::string classification(float value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return "infinity";
    return "finite";
}
std::string determinism(const std::vector<Observation>& values, float atol, float rtol) {
    if (values.size()<2) return "insufficient_repeats";
    bool same=true, near=true;
    for (const auto& v : values) {
        same=same && numeric::bitwise_equal(v.value,values.front().value);
        near=near && numeric::nearly_equal(v.value,values.front().value,atol,rtol);
    }
    return same ? "bitwise_deterministic" : near ? "numerically_equal_bitwise_varying" : "numerically_varying";
}
}
