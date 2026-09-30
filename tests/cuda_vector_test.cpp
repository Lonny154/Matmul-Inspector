#include "cuda_matmul.hpp"
#include "experiment.hpp"
#include "output.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>

int main() {
    std::string reason;
    if (!cuda_available(&reason)) { std::cout << reason << '\n'; return 77; }
    for (auto kind : {operation::Kind::dot,operation::Kind::reduction_sum}) {
        for (unsigned block : {64,128,256,512}) {
        for (std::size_t length : {1,3,17,255,256,257,1003,65537}) {
            Matrix a(1,length,length+3),b(length,1,2);
            // Exactly representable terms isolate indexing/stage correctness.
            for (std::size_t i=0;i<length;++i) { a(0,i)=float(int(i%7)-3); b(i,0)=float(int(i%3)-1); }
            auto cpu=operation::cpu(kind,a,b);
            auto gpu=cuda_vector(kind,a,b,2,1,block);
            if (!comparison::compare(cpu,gpu.output,0,0).tolerance_pass)
                throw std::runtime_error("Tree reduction incorrect for exact input");
            if (gpu.timing.samples.size()!=3) throw std::runtime_error("Missing timing samples");
            experiment::fill(a,42); experiment::fill(b,123);
            cpu=operation::cpu(kind,a,b);
            gpu=cuda_vector(kind,a,b,1,0,block);
            auto again=cuda_vector(kind,a,b,1,0,block);
            if (output::fingerprint(gpu.output)!=output::fingerprint(again.output))
                throw std::runtime_error("Nondeterministic tree");
            // Explicit exploratory tolerance for serial vs tree FP32 association.
            if (!comparison::compare(cpu,gpu.output,1e-2f,1e-3f).tolerance_pass)
                throw std::runtime_error("Random serial/tree disagreement exceeds test tolerance");
        }
    }
    }
    for (unsigned block : {64,128,256,512}) {
        Matrix a(1,1003),b(1003,1);
        reduction::fill_structured(a,b,operation::Kind::dot,"alternating_sign",42);
        const auto fp64=reduction::reference_fp64(operation::Kind::dot,a,b);
        const auto gpu=cuda_vector(operation::Kind::dot,a,b,1,0,block);
        if (std::fabs(double(gpu.output(0,0))-fp64)>1e-6) throw std::runtime_error("FP64 analysis discrepancy");
    }
    for (unsigned block : {64,128,256,512}) {
        Matrix a(1,64),unused(0,0);
        for (std::size_t i=0;i<64;++i) a(0,i)=-0.0f;
        auto result=cuda_vector(operation::Kind::reduction_sum,a,unused,1,0,block).output;
        if (result(0,0)!=0 || std::signbit(result(0,0))!=(block==64))
            throw std::runtime_error("Unexpected signed-zero tree/padding behavior");
        a(0,0)=std::numeric_limits<float>::infinity();
        result=cuda_vector(operation::Kind::reduction_sum,a,unused,1,0,block).output;
        if (!std::isinf(result(0,0)) || std::signbit(result(0,0))) throw std::runtime_error("Positive infinity lost");
        a(0,0)=-std::numeric_limits<float>::infinity();
        result=cuda_vector(operation::Kind::reduction_sum,a,unused,1,0,block).output;
        if (!std::isinf(result(0,0)) || !std::signbit(result(0,0))) throw std::runtime_error("Negative infinity lost");
        a(0,1)=std::numeric_limits<float>::infinity();
        result=cuda_vector(operation::Kind::reduction_sum,a,unused,1,0,block).output;
        if (!std::isnan(result(0,0))) throw std::runtime_error("Opposite infinities must yield NaN");
        a(0,1)=std::numeric_limits<float>::quiet_NaN();
        result=cuda_vector(operation::Kind::reduction_sum,a,unused,1,0,block).output;
        if (!std::isnan(result(0,0))) throw std::runtime_error("NaN must propagate");
    }
    bool rejected=false;
    try { cuda_vector(operation::Kind::dot,Matrix(1,3),Matrix(2,1)); }
    catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("Invalid vector lengths accepted");
    std::cout << "CUDA vector indexing, stages, timing and determinism passed\n";
}
