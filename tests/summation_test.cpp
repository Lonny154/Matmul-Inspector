#include "summation.hpp"
#include "reduction.hpp"
#include "experiment.hpp"
#include "numeric.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, int line) { if (!condition) throw std::runtime_error("Summation assertion failed at line " + std::to_string(line)); }
#define require(condition) check((condition), __LINE__)
int main() {
    using operation::Kind;
    for (auto kind : {Kind::reduction_sum,Kind::dot}) {
        for (std::size_t n : {1,3,5,17,257}) {
            Matrix a(1,n,n+2),b(n,1,3);
            for (std::size_t i=0;i<n;++i) { a(0,i)=float(i%5); b(i,0)=1; }
            auto fp64=reduction::reference_fp64(kind,a,b);
            for (const std::string method : {"fp32_forward","fp32_reverse","fp32_pairwise","kahan_fp32","neumaier_fp32","fp64_accumulation"}) {
                require(summation::execute(method,kind,a,b)(0,0)==fp64);
                auto config=experiment::parse({"compare","--operation",operation::name(kind),"--candidate",method});
                require(config.candidate==method);
            }
            for (std::size_t i=0;i<n;++i) a(0,i)=0;
            for (const auto method : {"fp32_pairwise","kahan_fp32","neumaier_fp32"})
                require(summation::execute(method,kind,a,b)(0,0)==0);
        }
    }
    Matrix a(1,257),b(257,1);
    reduction::fill_structured(a,b,Kind::dot,"cancellation",42);
    require(reduction::reference_fp64(Kind::dot,a,b)==255);
    require(summation::execute("neumaier_fp32",Kind::dot,a,b)(0,0)==255);
    // The final cancellation can still lose the low-order compensation.
    require(summation::execute("kahan_fp32",Kind::dot,a,b)(0,0)==256);
    reduction::fill_structured(a,b,Kind::dot,"repeated_small_plus_large",42);
    require(summation::execute("fp32_forward",Kind::dot,a,b)(0,0)==16777216);
    require(summation::execute("kahan_fp32",Kind::dot,a,b)(0,0)==16777472);
    require(summation::execute("neumaier_fp32",Kind::dot,a,b)(0,0)==16777472);
    // FP64 can represent this cancellation exactly, but Kahan can still lose 1.
    Matrix dynamic(1,3),ones(3,1);
    dynamic(0,0)=134217728; dynamic(0,1)=1; dynamic(0,2)=-134217728;
    for (unsigned i=0;i<3;++i) ones(i,0)=1;
    require(reduction::reference_fp64(Kind::dot,dynamic,ones)==1);
    require(summation::execute("kahan_fp32",Kind::dot,dynamic,ones)(0,0)==0);
    require(summation::execute("neumaier_fp32",Kind::dot,dynamic,ones)(0,0)==1);
    // Dot products are individually rounded in compensated/pairwise methods.
    Matrix product(1,1),multiplier(1,1);
    product(0,0)=0x1.000002p0f; multiplier(0,0)=0x1.fffffcp-1f;
    const auto ref=reduction::reference_fp64(Kind::dot,product,multiplier);
    require(ref!=1 && summation::execute("neumaier_fp32",Kind::dot,product,multiplier)(0,0)==1);
    auto metrics=summation::analyze(1,std::nextafter(1.0f,2.0f),1.0+std::ldexp(1.0,-25));
    require(metrics.rounded_target==1 && metrics.target_bitwise_match && metrics.ulp_to_target==0);
    require(metrics.classification=="exact_match" && *metrics.absolute_error>0);
    require(*metrics.signed_error<0 && *metrics.reduction_factor==3);
    require(summation::analyze(1,1,0).relative_error==std::nullopt);
    require(!summation::analyze(1,1,1e-13).relative_error);
    metrics=summation::analyze(1,2,1.5);
    require(metrics.signed_error==-0.5 && metrics.relative_error==1.0/3 && metrics.classification=="no_improvement");
    require(summation::analyze(1,1,1).factor_status=="both_zero");
    require(summation::analyze(1,1,1).reduction_factor==1);
    metrics=summation::analyze(2,1,1);
    require(metrics.factor_status=="zero_baseline" && metrics.reduction_factor==0 && metrics.classification=="worsened");
    metrics=summation::analyze(1,2,1);
    require(metrics.factor_status=="eliminated_error" && std::isinf(*metrics.reduction_factor));
    metrics=summation::analyze(2,4,1);
    require(metrics.classification=="improved" && metrics.error_reduction==2 && metrics.reduction_factor==3);
    metrics=summation::analyze(-0.0f,0,0);
    require(!metrics.target_bitwise_match && metrics.ulp_to_target==0 && metrics.classification=="no_improvement");
    metrics=summation::analyze(std::numeric_limits<float>::quiet_NaN(),1,1);
    require(metrics.classification=="unavailable" && !metrics.reduction_factor && !metrics.ulp_to_target);
    product(0,0)=std::numeric_limits<float>::infinity();
    for (const auto method : {"fp32_pairwise","kahan_fp32","neumaier_fp32"})
        require(std::isinf(summation::execute(method,Kind::reduction_sum,product,multiplier)(0,0)));
    bool rejected=false;
    try { experiment::parse({"compare","--candidate","kahan_fp32"}); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected);
    std::cout << "Stable methods, FP64 targets and error-attribution edge cases passed\n";
}
