#include "experiment.hpp"
#include <algorithm>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>

namespace experiment {
namespace {
void optional(std::ostream& out, const std::optional<double>& value) { if (value) out << *value; }
std::string warnings(const benchmark::Statistics& stats) {
    std::string text;
    for (const auto& w : stats.warnings) { if (!text.empty()) text += ';'; text += w; }
    return text;
}
}
std::map<std::string,std::string> timing_artifacts(const Config& config, const std::vector<Row>& rows) {
    std::ostringstream samples, stats, pairs;
    for (auto* out : {&samples,&stats,&pairs}) { out->imbue(std::locale::classic()); *out << std::setprecision(17); }
    samples << "row_id,M,N,K,backend,kernel,reference_kernel,timing_mode,timer,trial,iteration,phase,latency_ms,seed\n";
    stats << "row_id,M,N,K,kernel,timing_mode,scope,trial,measurement_count,warmup_count,trial_count,min_ms,max_ms,mean_ms,median_ms,stddev_ms,iqr_ms,mad_ms,cv,median_ci_low_ms,median_ci_high_ms,bootstrap_method,stability_warnings";
    for (int p : config.analysis.percentiles) stats << ",p" << p << "_ms";
    stats << '\n';
    pairs << "reference_row_id,candidate_row_id,M,N,K,reference_kernel,candidate_kernel,reference_timing_mode,candidate_timing_mode,reference_median_ms,candidate_median_ms,latency_ratio,percent_difference,speedup,median_ci_overlap,tolerance_pass\n";
    bool any = false;
    for (std::size_t id=0; id<rows.size(); ++id) {
        const auto& row = rows[id];
        if (!row.timed || row.timing.samples.empty()) continue;
        any = true;
        std::map<int,std::vector<benchmark::Sample>> trials;
        for (const auto& sample : row.timing.samples) {
            samples << id << ',' << row.shape.m << ',' << row.shape.n << ',' << row.shape.k
                << ',' << (row.kernel == "cpu" ? "cpu" : "cuda") << ',' << csv_field(row.kernel) << ',' << csv_field(row.reference)
                << ',' << csv_field(row.timing_mode) << ',' << (row.timing_mode == "kernel_only" ? "cuda_event" : "steady_clock")
                << ',' << sample.trial << ',' << sample.iteration << ',' << (sample.warmup ? "warmup" : "measurement")
                << ',' << sample.latency_ms << ',' << config.seed << '\n';
            trials[sample.trial].push_back(sample);
        }
        auto record = [&](const benchmark::Statistics& s, const char* scope, int trial) {
            const auto warmups = std::count_if(s.samples.begin(),s.samples.end(),[](const auto& v) { return v.warmup; });
            std::set<int> ids;
            for (const auto& v : s.samples) ids.insert(v.trial);
            stats << id << ',' << row.shape.m << ',' << row.shape.n << ',' << row.shape.k << ',' << csv_field(row.kernel)
                << ',' << csv_field(row.timing_mode) << ',' << scope << ',';
            if (trial >= 0) stats << trial;
            stats << ',' << s.samples.size()-warmups << ',' << warmups << ',' << ids.size()
                << ',' << s.min_ms << ',' << s.max_ms << ',' << s.mean_ms << ',' << s.median_ms << ',' << s.stddev_ms
                << ',' << s.iqr_ms << ',' << s.mad_ms << ',';
            optional(stats,s.cv); stats << ','; optional(stats,s.median_ci_low_ms); stats << ','; optional(stats,s.median_ci_high_ms);
            stats << ',' << (s.median_ci_low_ms ? (ids.size()>1 ? "whole_trial" : "iid_measurement") : "unavailable") << ',' << csv_field(warnings(s));
            for (int p : config.analysis.percentiles) {
                stats << ',';
                auto found=s.percentiles.find(p); if (found!=s.percentiles.end()) stats << found->second;
            }
            stats << '\n';
        };
        record(row.timing,"overall",-1);
        for (const auto& entry : trials) record(benchmark::analyze(entry.second,config.analysis),"trial",entry.first);
        if (row.kernel == row.reference) continue;
        for (std::size_t ref_id=0; ref_id<rows.size(); ++ref_id) {
            const auto& ref = rows[ref_id];
            if (!ref.timed || ref.kernel != row.reference || ref.shape.m != row.shape.m
                || ref.shape.n != row.shape.n || ref.shape.k != row.shape.k
                || ref.timing_mode != (row.reference == "cpu" ? "host_matmul" : row.timing_mode)) continue;
            const auto& a=ref.timing; const auto& b=row.timing;
            pairs << ref_id << ',' << id << ',' << row.shape.m << ',' << row.shape.n << ',' << row.shape.k
                << ',' << csv_field(ref.kernel) << ',' << csv_field(row.kernel)
                << ',' << csv_field(ref.timing_mode) << ',' << csv_field(row.timing_mode)
                << ',' << a.median_ms << ',' << b.median_ms << ',';
            if (a.median_ms>0 && b.median_ms>0)
                pairs << b.median_ms/a.median_ms << ',' << (b.median_ms/a.median_ms-1)*100 << ',' << a.median_ms/b.median_ms;
            else pairs << ",,";
            pairs << ',';
            if (a.median_ci_low_ms && a.median_ci_high_ms && b.median_ci_low_ms && b.median_ci_high_ms)
                pairs << (std::max(*a.median_ci_low_ms,*b.median_ci_low_ms) <= std::min(*a.median_ci_high_ms,*b.median_ci_high_ms) ? "true" : "false");
            pairs << ',' << (row.comparison.tolerance_pass ? "true" : "false") << '\n';
            break;
        }
    }
    if (!any) return {};
    return {{"timing_samples.csv",samples.str()}, {"timing_statistics.csv",stats.str()}, {"pairwise.csv",pairs.str()}};
}
} // namespace experiment
