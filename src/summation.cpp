#include "summation.hpp"
#include "reduction.hpp"
#include "numeric.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace summation {
bool supported(const std::string& method) {
    return method == "fp32_forward" || method == "fp32_reverse" || method == "fp32_pairwise"
        || method == "kahan_fp32" || method == "neumaier_fp32" || method == "fp64_accumulation";
}
const char* fp_policy() { return MI_SUMMATION_FP_POLICY; }
Matrix execute(const std::string& method, operation::Kind kind, const Matrix& a, const Matrix& b) {
    if (method == "cpu" || method == "fp32_forward") return operation::cpu(kind,a,b);
    if (method == "cpu-reverse" || method == "fp32_reverse") return operation::cpu(kind,a,b,true);
    if (!supported(method)) throw std::invalid_argument("Unknown summation method: " + method);
#if !MI_SUMMATION_STRICT
    if (method != "fp64_accumulation") throw std::runtime_error("Stable summation requires GCC, Clang or MSVC strict-FP support");
#endif
    operation::validate_vectors(kind,a,b);
    Matrix result(1,1);
    if (method == "fp64_accumulation") {
        result(0,0)=static_cast<float>(reduction::reference_fp64(kind,a,b));
        return result;
    }
    // This translation unit disables contraction/reassociation. Dot terms are
    // rounded FP32 products; compensation corrects summation, not multiplication.
    auto term=[&](std::size_t i) { return kind == operation::Kind::dot ? a(0,i)*b(i,0) : a(0,i); };
    if (method == "fp32_pairwise") {
        std::vector<float> values(a.cols());
        for (std::size_t i=0;i<a.cols();++i) values[i]=term(i);
        // Adjacent pairs, with an unpaired tail carried unchanged at each level.
        // In-place writes never overwrite a term still needed by this level.
        for (std::size_t count=values.size();count>1;) {
            std::size_t next=0;
            for (std::size_t i=0;i<count;i+=2)
                values[next++]=i+1<count ? values[i]+values[i+1] : values[i];
            count=next;
        }
        result(0,0)=values[0];
        return result;
    }
    float sum=0, compensation=0;
    for (std::size_t i=0;i<a.cols();++i) {
        const float x=term(i);
        if (method == "kahan_fp32") {
            const float y=x-compensation;
            const float t=sum+y;
            // Preserve ordinary infinity/NaN propagation instead of creating
            // a spurious inf-inf compensation for a single infinite input.
            compensation=std::isfinite(t) ? (t-sum)-y : 0.0f;
            sum=t;
        } else {
            const float t=sum+x;
            if (std::isfinite(t)) {
                compensation += std::fabs(sum)>=std::fabs(x) ? (sum-t)+x : (x-t)+sum;
            } else compensation=0;
            sum=t;
        }
    }
    result(0,0)=method == "neumaier_fp32" ? sum+compensation : sum;
    return result;
}
Analysis analyze(float value, float forward, double fp64) {
    Analysis result;
    result.rounded_target=static_cast<float>(fp64);
    result.target_bitwise_match=numeric::bitwise_equal(value,result.rounded_target);
    if (std::isfinite(value) && std::isfinite(result.rounded_target))
        result.ulp_to_target=numeric::ulp_distance(value,result.rounded_target);
    if (!std::isfinite(value) || !std::isfinite(fp64)) return result;
    result.signed_error=double(value)-fp64;
    result.absolute_error=std::fabs(*result.signed_error);
    if (std::fabs(fp64)>reduction::relative_floor) result.relative_error=*result.absolute_error/std::fabs(fp64);
    if (!std::isfinite(forward)) return result;
    result.baseline_error=std::fabs(double(forward)-fp64);
    result.error_reduction=*result.baseline_error-*result.absolute_error;
    if (*result.baseline_error == 0 && *result.absolute_error == 0) {
        result.reduction_factor=1;
        result.factor_status="both_zero";
    } else if (*result.baseline_error == 0) {
        result.reduction_factor=0;
        result.factor_status="zero_baseline";
    } else if (*result.absolute_error == 0) {
        result.reduction_factor=std::numeric_limits<double>::infinity();
        result.factor_status="eliminated_error";
    } else {
        result.reduction_factor=*result.baseline_error / *result.absolute_error;
        result.factor_status="finite";
    }
    result.classification=result.target_bitwise_match ? "exact_match"
        : *result.absolute_error<*result.baseline_error ? "improved"
        : *result.absolute_error>*result.baseline_error ? "worsened" : "no_improvement";
    return result;
}
}
