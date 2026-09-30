#include "experiment.hpp"
#include "numeric.hpp"
#include "output.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

void require(bool value) { if (!value) throw std::runtime_error("Reduction assertion failed"); }
int main() {
    using operation::Kind;
    for (const auto& fixture : {"random_uniform","ascending_magnitude","descending_magnitude","alternating_sign",
                                "cancellation","large_dynamic_range","repeated_small_plus_large"}) {
        for (auto kind : {Kind::dot,Kind::reduction_sum}) {
            experiment::Config c; c.operation=kind; c.input=fixture;
            Matrix a(1,1003),b(1003,1),again(1,1003),b2(1003,1);
            experiment::fill_inputs(a,b,c); experiment::fill_inputs(again,b2,c);
            require(output::fingerprint(a)==output::fingerprint(again));
            if (kind == Kind::dot) require(output::fingerprint(b)==output::fingerprint(b2));
            for (std::size_t i=0;i<a.cols();++i) require(std::isfinite(a(0,i)));
        }
    }
    Matrix a(1,257),b(257,1),ascending(1,257),descending(1,257);
    reduction::fill_structured(a,b,Kind::dot,"cancellation",42);
    require(reduction::reference_fp64(Kind::dot,a,b)==255);
    require(operation::cpu(Kind::dot,a,b)(0,0)==0);
    reduction::fill_structured(a,b,Kind::dot,"repeated_small_plus_large",42);
    require(operation::cpu(Kind::dot,a,b)(0,0)==16777216);
    require(operation::cpu(Kind::dot,a,b,true)(0,0)==16777472);
    require(reduction::reference_fp64(Kind::dot,a,b)==16777472);
    reduction::fill_structured(ascending,b,Kind::dot,"ascending_magnitude",42);
    reduction::fill_structured(descending,b,Kind::dot,"descending_magnitude",42);
    for (std::size_t i=0;i<257;++i) require(ascending(0,i)==descending(0,256-i));
    auto point=reduction::observe(1,1,0,1e-6f,1e-5f);
    require(point.absolute_error==1 && !point.relative_error);
    require(!reduction::observe(1,1,1e-13,0,0).relative_error);
    require(reduction::observe(1,1,2,0,0).relative_error==0.5);
    auto zero=reduction::observe(-0.0f,0,0,0,0);
    require(zero.ulp==0 && zero.tolerance_pass && numeric::float_to_bits(zero.value)==0x80000000u);
    const float inf=std::numeric_limits<float>::infinity(), nan=std::numeric_limits<float>::quiet_NaN();
    for (float v : {inf,-inf,nan}) {
        auto special=reduction::observe(v,v,v,0,0);
        require(!special.absolute_error && !special.relative_error && !special.ulp);
        require(special.tolerance_pass==!std::isnan(v));
    }
    Matrix special_input(1,2),unused(0,0);
    special_input(0,0)=-0.0f; special_input(0,1)=0.0f;
    require(!std::signbit(operation::cpu(Kind::reduction_sum,special_input,unused)(0,0)));
    special_input(0,0)=inf;
    require(reduction::reference_fp64(Kind::reduction_sum,special_input,unused)==double(inf));
    special_input(0,1)=-inf;
    require(std::isnan(operation::cpu(Kind::reduction_sum,special_input,unused)(0,0)));
    special_input(0,0)=nan;
    require(std::isnan(reduction::reference_fp64(Kind::reduction_sum,special_input,unused)));
    require(reduction::classification(inf)=="infinity" && reduction::classification(nan)=="nan");
    auto one=reduction::observe(1,1,1,0,0);
    auto next=reduction::observe(std::nextafter(1.0f,2.0f),1,1,0,0);
    require(next.ulp==1);
    require(reduction::determinism({one},0,0)=="insufficient_repeats");
    require(reduction::determinism({one,one},0,0)=="bitwise_deterministic");
    require(reduction::determinism({one,next},1e-6f,0)=="numerically_equal_bitwise_varying");
    require(reduction::determinism({one,next},0,0)=="numerically_varying");
    for (unsigned block : {64,128,256,512}) {
        operation::validate_block_size(block);
        require(operation::stages(block+1,block)==2);
    }
    bool rejected=false;
    try { operation::validate_block_size(96); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected);
    auto c=experiment::parse({"compare","--operation","dot","--input","cancellation","--size","3",
                              "--candidate","cpu-reverse","--block-size","64","--reference-block-size","128","--repeats","3"});
    require(c.repeats==3 && c.reduction_block_size==64 && c.reference_block_size==128);
    require(experiment::generator(c)=="vector-cancellation-v1");
    std::cout << "Reduction fixtures, references, diagnostics and classification passed\n";
}
