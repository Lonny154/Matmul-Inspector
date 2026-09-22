#include "output.hpp"
#include "experiment.hpp"
#include "numeric.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
float tolerance(const char* text) {
    std::string value(text); std::size_t end;
    float result=std::stof(value,&end);
    if (end!=value.size() || !std::isfinite(result) || result<0) throw std::invalid_argument("Invalid tolerance");
    return result;
}
void number(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << experiment::json_string(std::isnan(value)?"nan":value<0?"-inf":"inf");
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=5) throw std::invalid_argument("Usage: matmul-compare-outputs REFERENCE.bin CANDIDATE.bin ATOL RTOL");
        auto reference=output::read(argv[1]), candidate=output::read(argv[2]);
        auto result=comparison::compare(reference,candidate,tolerance(argv[3]),tolerance(argv[4]),0);
        std::cout << std::setprecision(17) << "{\"diagnostics_version\":1,\"divergent_count\":" << result.divergent_count
            << ",\"divergent_percent\":" << result.divergent_percent << ",\"max_ulp\":" << result.max_ulp
            << ",\"mean_divergent_ulp\":" << result.mean_divergent_ulp << ",\"max_absolute_error\":";
        number(result.max_absolute_error); std::cout << ",\"max_relative_error\":"; number(result.max_relative_error);
        std::cout << ",\"tolerance_failures\":" << result.tolerance_failures
            << ",\"tolerance_failure_percent\":" << result.tolerance_failure_percent
            << ",\"nan_pairs\":" << result.nan_pairs << ",\"infinity_pairs\":" << result.infinity_pairs
            << ",\"zero_reference_nonzero\":" << result.zero_reference_nonzero << ",\"ulp_bins\":[";
        for (unsigned i=0;i<6;++i) { if (i) std::cout << ','; std::cout << result.ulp_bins[i]; }
        auto point=[](const std::optional<comparison::Divergence>& p) {
            if (!p) { std::cout << "null"; return; }
            std::cout << "{\"row\":" << p->row << ",\"col\":" << p->col << ",\"reference_value\":";
            number(p->reference); std::cout << ",\"candidate_value\":"; number(p->candidate);
            std::cout << ",\"reference_bits\":\"" << numeric::float_bits(p->reference) << "\",\"candidate_bits\":\""
                << numeric::float_bits(p->candidate) << "\",\"ulp_distance\":" << numeric::ulp_distance(p->reference,p->candidate)
                << ",\"absolute_error\":"; number(numeric::absolute_error(p->candidate,p->reference));
            std::cout << ",\"relative_error\":"; number(numeric::relative_error(p->candidate,p->reference));
            std::cout << '}';
        };
        std::cout << "],\"first_divergence\":"; point(result.first_bitwise);
        std::cout << ",\"first_tolerance_failure\":"; point(result.first_numeric);
        std::cout << "}\n";
        return 0; // A measured divergence is data, not a helper execution error.
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
