#include "experiment.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void close(double a,double b) { check(std::fabs(a-b)<1e-10,"statistic differs from expected value"); }
template<class F> void invalid(F f) {
    bool caught=false; try { f(); } catch (const std::invalid_argument&) { caught=true; }
    check(caught,"invalid input accepted");
}
bool warning(const benchmark::Statistics& s,const std::string& flag) {
    return std::find(s.warnings.begin(),s.warnings.end(),flag)!=s.warnings.end();
}
std::vector<benchmark::Sample> samples(const std::vector<float>& values,int trial=0) {
    std::vector<benchmark::Sample> out;
    for (std::size_t i=0;i<values.size();++i) out.push_back({trial,static_cast<int>(i),false,values[i]});
    return out;
}
int main(int argc,char** argv) {
    try {
        auto s=benchmark::statistics({1,2,3,4});
        close(s.mean_ms,2.5); close(s.median_ms,2.5); close(s.min_ms,1); close(s.max_ms,4);
        close(s.stddev_ms,std::sqrt(1.25)); close(s.iqr_ms,1.5); close(s.mad_ms,1);
        close(*s.cv,std::sqrt(1.25)/2.5);
        close(benchmark::percentile({1,2,3,4},5),1.15);
        close(benchmark::percentile({4,1,3,2},25),1.75);
        close(benchmark::percentile({1,2,3,4},95),3.85);
        close(benchmark::percentile({4,1,3,2},0),1);
        close(benchmark::percentile({4,1,3,2},100),4);
        auto zero=benchmark::statistics({0,0});
        check(!zero.cv && warning(zero,"zero_latency_timer_resolution"),"zero CV undefined");
        benchmark::AnalysisOptions options;
        options.bootstrap_samples=5; options.confidence_level=.8; options.seed=42;
        auto bootstrap=benchmark::analyze(samples({1,2,3,4}),options);
        // Golden MT19937/rejection draws: replicate medians 3,2,3,3,4.
        close(*bootstrap.median_ci_low_ms,2.4); close(*bootstrap.median_ci_high_ms,3.6);
        auto repeat=benchmark::analyze(samples({1,2,3,4}),options);
        check(bootstrap.median_ci_low_ms==repeat.median_ci_low_ms && bootstrap.median_ci_high_ms==repeat.median_ci_high_ms,"deterministic bootstrap");
        bool changes=false;
        for (unsigned seed=43;seed<50;++seed) {
            options.seed=seed;
            auto other=benchmark::analyze(samples({1,2,3,4}),options);
            changes=changes || other.median_ci_low_ms!=bootstrap.median_ci_low_ms || other.median_ci_high_ms!=bootstrap.median_ci_high_ms;
        }
        check(changes,"bootstrap seed affects resamples");
        options.bootstrap_samples=1000; options.confidence_level=.95;
        auto constant=benchmark::analyze(samples({7,7,7}),options);
        close(*constant.median_ci_low_ms,7); close(*constant.median_ci_high_ms,7);
        std::vector<benchmark::Sample> blocks{{0,0,true,999},{0,0,false,1},{0,1,false,1},{1,0,true,999},{1,0,false,9},{1,1,false,9}};
        auto pooled=benchmark::analyze(blocks,options);
        close(pooled.median_ms,5); close(pooled.mean_ms,5);
        close(*pooled.median_ci_low_ms,1); close(*pooled.median_ci_high_ms,9);
        check(pooled.samples.size()==6,"trial/warmup boundaries retained");
        check(!benchmark::analyze(samples({1}),options).median_ci_low_ms,"singleton CI unavailable");
        options.bootstrap_samples=0;
        check(!benchmark::analyze(samples({1,2}),options).median_ci_low_ms,"disabled CI unavailable");
        auto noisy=benchmark::statistics({100,1,1,1,1,1,1,1,1,1});
        check(warning(noisy,"cv_gt_20_percent") && warning(noisy,"max_min_ratio_gt_3"),"noise warnings");
        check(warning(noisy,"trial_0_first_iteration_deviation_gt_50_percent"),"first sample warning");
        check(warning(benchmark::statistics({1,1,1,1,2,2,2,2}),"trial_0_end_to_start_drift_gt_20_percent"),"drift warning");
        auto stable=benchmark::statistics(std::vector<float>(20,1));
        check(!warning(stable,"cv_gt_20_percent") && !warning(stable,"max_min_ratio_gt_3"),"stable series falsely flagged");
        invalid([] { benchmark::analyze({}); });
        invalid([] { benchmark::analyze({{0,0,true,1}}); });
        invalid([] { benchmark::analyze({{0,0,false,1},{0,0,false,2}}); });
        invalid([] { benchmark::analyze({{0,1,false,1}}); });
        invalid([] { benchmark::analyze({{0,0,false,1},{0,0,true,1}}); });
        invalid([] { benchmark::statistics({-1}); });
        invalid([] { benchmark::statistics({std::numeric_limits<float>::quiet_NaN()}); });
        invalid([] { benchmark::percentile({},50); });
        invalid([] { benchmark::percentile({1},101); });
        auto config=experiment::parse({"benchmark","--trials","3","--seed","9","--bootstrap-samples","50","--confidence-level","0.9","--percentiles","99,25,50"});
        check(config.trials==3 && config.analysis.seed==9 && config.analysis.percentiles==std::vector<int>({25,50,99}),"analysis config");
        for (const auto& args: std::vector<std::vector<std::string>>{
            {"compare","--trials","2"},{"benchmark","--trials","0"},{"benchmark","--confidence-level","1"},
            {"benchmark","--confidence-level","nan"},{"benchmark","--percentiles","25,25"},{"benchmark","--percentiles","101"},
            {"benchmark","--bootstrap-samples","-1"}}) invalid([&] { experiment::parse(args); });
        if (argc==2) {
            config=experiment::parse({"benchmark","--sizes","4","--trials","2","--bootstrap-samples","100","--iterations","4","--warmups","2"});
            std::vector<experiment::Row> rows;
            for (int kernel=0;kernel<2;++kernel) {
                experiment::Row row;
                row.shape={4,4,4}; row.kernel=kernel ? "tiled" : "naive"; row.reference="naive";
                row.timed=true; row.timing_mode="kernel_only";
                std::vector<benchmark::Sample> raw;
                for (int trial=0;trial<2;++trial) {
                    raw.push_back({trial,0,true,100}); raw.push_back({trial,1,true,50});
                    for (int i=0;i<4;++i) raw.push_back({trial,i,false,static_cast<float>((1+trial+i)*(kernel ? 1 : 2))});
                }
                row.timing=benchmark::analyze(raw,config.analysis);
                row.speedup=row.median_speedup=kernel ? 2 : 1;
                rows.push_back(row);
            }
            experiment::create_run_directory(argv[1]);
            experiment::write_artifacts(argv[1],config,{{"status","complete"},{"timing_schema_version","1"},{"fixture","synthetic statistical test data"}},rows,"test fixture\n");
        }
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
    return 0;
}
