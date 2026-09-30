#include "cuda_matmul.hpp"
#include "experiment.hpp"
#include "output.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    std::string reason;
    if (!cuda_available(&reason)) { std::cout << reason << '\n'; return 77; }
    for (auto kind : {operation::Kind::dot,operation::Kind::reduction_sum}) {
        for (std::size_t length : {1,3,17,255,256,257,1003,65537}) {
            Matrix a(1,length,length+3),b(length,1,2);
            // Exactly representable terms isolate indexing/stage correctness.
            for (std::size_t i=0;i<length;++i) { a(0,i)=float(int(i%7)-3); b(i,0)=float(int(i%3)-1); }
            auto cpu=operation::cpu(kind,a,b);
            auto gpu=cuda_vector(kind,a,b,2,1);
            if (!comparison::compare(cpu,gpu.output,0,0).tolerance_pass)
                throw std::runtime_error("Tree reduction incorrect for exact input");
            if (gpu.timing.samples.size()!=3) throw std::runtime_error("Missing timing samples");
            experiment::fill(a,42); experiment::fill(b,123);
            cpu=operation::cpu(kind,a,b);
            gpu=cuda_vector(kind,a,b);
            auto again=cuda_vector(kind,a,b);
            if (output::fingerprint(gpu.output)!=output::fingerprint(again.output))
                throw std::runtime_error("Nondeterministic tree");
            // Explicit exploratory tolerance for serial vs tree FP32 association.
            if (!comparison::compare(cpu,gpu.output,1e-2f,1e-3f).tolerance_pass)
                throw std::runtime_error("Random serial/tree disagreement exceeds test tolerance");
        }
    }
    bool rejected=false;
    try { cuda_vector(operation::Kind::dot,Matrix(1,3),Matrix(2,1)); }
    catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("Invalid vector lengths accepted");
    std::cout << "CUDA vector indexing, stages, timing and determinism passed\n";
}
