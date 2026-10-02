#include "experiment.hpp"
#include "inspector.hpp"
#include "output.hpp"
#include "summation.hpp"
#include <fstream>
#ifdef MATMUL_INSPECTOR_HAS_CUDA
#include "cuda_matmul.hpp"
#endif

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>

namespace experiment {
namespace {
class Tee : public std::streambuf {
public:
    Tee(std::ostream& stream, std::ostringstream& capture)
        : stream_(stream), capture_(capture), original_(stream.rdbuf(this)),
          flags_(stream.flags()), precision_(stream.precision()) {}
    ~Tee() {
        stream_.rdbuf(original_);
        stream_.flags(flags_);
        stream_.precision(precision_);
    }
private:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
        capture_.put(traits_type::to_char_type(ch));
        return original_->sputc(traits_type::to_char_type(ch));
    }
    std::streamsize xsputn(const char* data, std::streamsize length) override {
        capture_.write(data, length);
        return original_->sputn(data, length);
    }
    int sync() override { return original_->pubsync(); }
    std::ostream& stream_;
    std::ostringstream& capture_;
    std::streambuf* original_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
};

#ifdef MATMUL_INSPECTOR_HAS_CUDA
CudaMatmulKernel cuda_kernel(const std::string& name) {
    if (name == "naive") return CudaMatmulKernel::naive;
    if (name == "tiled") return CudaMatmulKernel::tiled;
    if (name == "register-blocked-2x2")
    return CudaMatmulKernel::register_blocked_2x2;
    if (name == "cuda-naive-fma") return CudaMatmulKernel::naive_fma;
    if (name == "cuda-naive-no-fma") return CudaMatmulKernel::naive_no_fma;
    if (name == "cuda-naive-reordered") return CudaMatmulKernel::naive_reordered;
    throw std::invalid_argument("Unknown CUDA kernel: " + name);
}
#endif

std::string capture_output(const Matrix& matrix, const Config& config, const Shape& shape,
                           const std::string& kernel, const std::string& hash, const std::string& id, unsigned threads = 256) {
    if (!config.save_output) return "";
    const std::string filename = "output-" + id + ".bin";
    output::save(matrix, config.output / filename);
    std::ofstream sidecar(config.output / (filename + ".json"));
    sidecar.exceptions(std::ios::failbit | std::ios::badbit);
    sidecar << "{\"format\":\"mifp32le-v1\",\"dtype\":\"float32\",\"byte_order\":\"little\",\"rows\":"
        << matrix.rows() << ",\"cols\":" << matrix.cols() << ",\"M\":" << shape.m << ",\"N\":" << shape.n
        << ",\"K\":" << shape.k << ",\"operation\":" << json_string(operation::name(config.operation)) << ",\"kernel\":" << json_string(kernel)
        << ",\"seed\":" << config.seed << ",\"seed_b\":" << config.seed_b
        << ",\"generator\":" << json_string(generator(config))
        << ",\"contraction_mode\":" << json_string(contraction_mode(kernel))
        << ",\"accumulation_mode\":" << json_string(accumulation_mode(kernel,config.operation,threads))
        << ",\"reduction_block_size\":" << (kernel == "cuda-tree" ? threads : 0)
        << ",\"output_sha256\":" << json_string(hash) << "}\n";
    sidecar.close();
    return filename;
}
void capture_pair(Row& row, const Matrix& ref, const Matrix& actual, const Config& config, std::size_t id) {
    row.reference_sha256 = output::fingerprint(ref);
    row.output_sha256 = output::fingerprint(actual);
    row.reference_output_file = capture_output(ref, config, row.shape, row.reference, row.reference_sha256, std::to_string(id)+"-reference", config.reference_block_size);
    row.output_file = capture_output(actual, config, row.shape, row.kernel, row.output_sha256, std::to_string(id)+"-candidate", config.reduction_block_size);
    std::cout << "reference SHA-256: " << row.reference_sha256 << "\ncandidate SHA-256: " << row.output_sha256 << '\n';
}

Matrix execute(const Matrix& a, const Matrix& b, const std::string& kernel, operation::Kind kind, unsigned threads) {
    if (operation::is_cpu(kernel)) return summation::execute(kernel,kind,a,b);
#ifdef MATMUL_INSPECTOR_HAS_CUDA
    if (kind != operation::Kind::matmul) return cuda_vector(kind,a,b,1,0,threads).output;
    return cuda_matmul(a, b, cuda_kernel(kernel));
#else
    (void)threads;
    throw std::runtime_error("CUDA support was not built");
#endif
}

template<class Operation>
auto repeated(Operation operation, const Config& config) {
    auto result = operation();
    const std::string first_hash = output::fingerprint(result.output);
    auto samples = result.timing.samples;
    for (int trial = 1; trial < config.trials; ++trial) {
        auto next = operation();
        // Outside every timer: ensure later-trial outputs preserve the checked result.
        if (output::fingerprint(next.output) != first_hash)
            throw std::runtime_error("Output changed bitwise across benchmark trials");
        for (auto sample : next.timing.samples) { sample.trial = trial; samples.push_back(sample); }
        result = std::move(next);
    }
    result.timing = benchmark::analyze(samples,config.analysis);
    return result;
}

#ifdef MATMUL_INSPECTOR_HAS_CUDA
void timing_line(const Row& row, operation::Kind kind = operation::Kind::matmul) {
    const double gflops = row.timing.mean_ms > 0
        ? operation::flops(kind,row.shape.m,row.shape.n,row.shape.k) / (row.timing.mean_ms * 1e6) : 0;
    std::cout << std::setprecision(9) << row.kernel << " " << row.timing_mode << ": median=" << row.timing.median_ms
        << " ms mean=" << row.timing.mean_ms << " ms min=" << row.timing.min_ms
        << " ms stddev=" << row.timing.stddev_ms << " ms GFLOP/s=" << gflops
        << " mean_speedup=" << row.speedup << "x median_speedup=" << row.median_speedup << "x";
    if (row.timing.median_ci_low_ms && row.timing.median_ci_high_ms)
        std::cout << " median_CI=[" << *row.timing.median_ci_low_ms << ',' << *row.timing.median_ci_high_ms << "] ms";
    std::cout << '\n';
}
#endif
}  // namespace

int run_cli(const std::vector<std::string>& arguments) {
    if (arguments == std::vector<std::string>{"--help"}
        || (arguments.size() == 2 && (arguments[0] == "compare" || arguments[0] == "benchmark" || arguments[0] == "crossover") && arguments[1] == "--help")) {
        std::cout << usage();
        return 0;
    }
    Config config;
    try { config = parse(arguments); }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n' << usage();
        return 2;
    }
    Metadata values;
    std::vector<Row> rows;
    std::ostringstream console;
    bool owns_directory = false;
    int exit_code = 0;
    {
        Tee output(std::cout, console), errors(std::cerr, console);
        try {
            values = metadata();
            std::string command = "matmul-inspector";
            for (const auto& arg : arguments) command += " " + json_string(arg);
            values["command"] = command;
            values["timing_methodology"] = config.mode == "compare" ? "untimed correctness comparison" : values["timing_methodology"];
            if (config.mode == "crossover") {
                values["cpu_implementation"] = "single-threaded FP32 row/column/K matmul; compiler-default contraction; no BLAS";
                values["cpu_timing_methodology"] = "steady_clock; existing matmul call including host output allocation/zero initialization; input generation, prior output destruction and analysis excluded";
                values["kernel_only_timing_methodology"] = "CUDA events; allocations, transfers, warmups and host analysis excluded";
                values["end_to_end_timing_methodology"] = "steady_clock around synchronous cuda_matmul: validation/device queries, host output allocation/initialization, cudaMalloc, pageable H2D A/B, cudaMemset C, launch, synchronization, D2H C, cudaFree; context/module priming and analysis excluded";
                values["crossover_statistic"] = "median_ms; speedup = CPU median / GPU median; strict GPU < CPU; first sampled win, no interpolation or sustained-win assumption";
                values["crossover_execution_order"] = "ascending sizes; CPU then kernel_only then end_to_end; one untimed GPU priming call per timing mode per trial in addition to configured warmups";
                values["timing_methodology"] = "CPU host_matmul and GPU end_to_end use steady_clock; GPU kernel_only uses CUDA events; see per-mode timing methodology fields";
            }
            if (config.mode != "compare") {
                values["timing_methodology"] += "; timing extension v1: individually timed/synchronized warmups; one GPU priming call per trial; raw trial boundaries retained";
                values["gpu_priming"] = "one untimed synchronous call per GPU kernel/timing mode/trial before configured warmups; not a measured warmup sample";
                values["timing_schema_version"] = "1";
                values["primary_latency_statistic"] = "median_ms";
                values["bootstrap_method"] = "percentile CI of pooled measurement median; whole-trial resampling with replacement when trials>1; IID measurement resampling within a single trial otherwise; mt19937 rejection mapping; linear quantiles (n-1)*p";
                values["statistics_method"] = "pooled measured samples only; population stddev; IQR=p75-p25; MAD=median absolute deviation (unscaled); CV=stddev/mean (undefined at zero mean)";
                values["warmup_method"] = "each trial repeats configured warmups; warmups timed separately by the same timer with per-launch synchronization on CUDA; excluded from summaries and bootstrap; no adaptive warmup";
                values["trial_method"] = "sequential blocks in one process, not independent process restarts; all reference trials precede candidate trials; crossover CPU then kernel_only then end_to_end; warmups repeated for every trial; inter-trial analysis/fingerprinting untimed";
                values["stability_thresholds"] = "heuristics only: CV>0.20; max/min>3; IQR/median>0.25; first vs remaining median deviation>50% (n>=5); last vs first quarter median deviation>20% (n>=8); n<10 or trials<5 limited evidence; zero latency timer resolution";
            }
            if (config.operation != operation::Kind::matmul) {
                values["operation_extension"] = "3";
                values["summation_fp_policy"] = summation::fp_policy();
                values["summation_fp_policy_scope"] = "src/summation.cpp only: pairwise, Kahan, Neumaier and error analysis; existing forward/reverse and FP64 reference keep their original compilation flags";
                values["summation_methods"] = "fp32_forward/reverse reuse existing serial kernels; pairwise combines adjacent pairs carrying odd tails; Kahan/Neumaier use FP32 sum and compensation; fp64_accumulation casts FP64 reference to FP32";
                values["summation_dot_products"] = "pairwise/Kahan/Neumaier use separately rounded FP32 products; forward/reverse preserve compiler-default contraction; FP64 reference multiplies promoted FP32 operands";
                values["summation_analysis"] = "improvement baseline=fp32_forward; target=FP64 reference cast to FP32; exact_match means target bits, not exact mathematical sum";
                values["fp64_analysis"] = "serial increasing-index FP64 accumulation; dot multiplies promoted FP32 operands in FP64; not mathematically exact";
                values["relative_error_fp64_floor"] = "1e-12; omit relative error when abs(reference)<=floor or nonfinite";
                values["repeat_method"] = "compare-mode candidate executions in one process; fresh allocations per CUDA execution; observations against fixed FP32 baseline; not independent process restarts";
                values["cpu_implementation"] = "cpu: serial FP32 increasing index; cpu-reverse: decreasing index; compiler-default contraction";
                values["cuda_implementation"] = std::to_string(config.reduction_block_size) + "-thread shared-memory binary tree; fixed multi-stage reduction; explicit RN multiply and add; no atomics";
                values["reduction_block_size"] = std::to_string(config.reduction_block_size);
                values["reduction_stages"] = "ceil-divide length by selected block size repeatedly until one partial remains; at least one stage; see summary rows";
                if (config.mode == "benchmark") values["timing_methodology"] =
                    "CPU: steady_clock including scalar output allocation; CUDA: event pair around all reduction stages, synchronized each sample; inputs, allocation, transfers, priming, analysis excluded from CUDA timing; configured warmups excluded from statistics; sequential trials";
            }
            create_run_directory(config.output);
            owns_directory = !config.output.empty();
            std::cout << config.mode << " operation=" << operation::name(config.operation) << ": reference=" << config.reference << " candidate=" << config.candidate
                << " input=" << config.input << " seed=" << config.seed << " seed_b=" << config.seed_b << '\n'
                << std::setprecision(9) << "atol=" << config.atol << " rtol=" << config.rtol << '\n';
            if (values["git_dirty"] == "true") {
                std::cerr << "WARNING: DIRTY SOURCE TREE OR BUILD. git_dirty=true; the commit alone cannot reproduce this run.\n";
            } else if (values["git_dirty"] == "unknown") {
                std::cerr << "WARNING: Git provenance is unknown; source cleanliness cannot be verified.\n";
            }
            if (values["git_commit"] != values["runtime_git_commit"]) {
                std::cerr << "WARNING: Build and runtime source commits differ (or are unavailable).\n";
            }
            const bool controlled = config.reference.find("cuda-naive-") == 0 || config.candidate.find("cuda-naive-") == 0;
            if (controlled) {
                std::cout << "FP instruction verification: " << values["fp_verified"] << " (" << values["fp_verification_method"] << ")\n";
                std::cout << "reference: " << contraction_mode(config.reference) << ", " << accumulation_mode(config.reference)
                    << "; candidate: " << contraction_mode(config.candidate) << ", " << accumulation_mode(config.candidate) << '\n';
                if (contraction_mode(config.reference) != contraction_mode(config.candidate)
                    && accumulation_mode(config.reference) != accumulation_mode(config.candidate))
                    std::cerr << "WARNING: Both contraction setting and accumulation strategy differ; use cuda-naive-fma as the controlled reference.\n";
            }
            const bool needs_gpu = !operation::is_cpu(config.reference) || !operation::is_cpu(config.candidate);
            bool available = !needs_gpu;
            std::string reason = "CUDA support was not built";
#ifdef MATMUL_INSPECTOR_HAS_CUDA
            if (needs_gpu) {
                available = cuda_available(&reason);
                // Optional probes are outside measured execution and do not make a run fail.
                if (available) {
                    for (const auto& entry : cuda_metadata()) values[entry.first] = entry.second;
                    auto driver = command_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader");
                    if (!driver.empty()) values["nvidia_driver_version"] = driver.substr(0, driver.find('\n'));
                }
            }
#endif
            if (!available) {
                values["status"] = "skipped";
                values["skip_reason"] = reason;
                std::cout << "GPU " << config.mode << " skipped: " << reason << '\n';
                exit_code = config.legacy ? 0 : 77;
            } else {
                if (controlled && values["fp_verified"] != "true")
                    throw std::runtime_error("Controlled kernels are unverified. Build with Python3, cuobjdump and embedded PTX; inspect fp_verification.json.");
                if (config.mode == "benchmark" && config.operation != operation::Kind::matmul)
                    std::cout << "CPU host_serial: steady_clock including scalar allocation; GPU kernel_only: CUDA events around all stages, allocation/transfers excluded\n";
                if (config.mode == "benchmark" && config.operation == operation::Kind::matmul) std::cout << "CUDA events; warmups=" << config.warmups << " iterations=" << config.iterations
                    << "; allocation/transfers/comparison excluded\n";
                if (config.mode == "crossover") std::cout << "CPU host_matmul: steady_clock; GPU kernel_only: CUDA events; "
                    << "GPU end_to_end: steady_clock including allocation/transfers; warmups=" << config.warmups
                    << " iterations=" << config.iterations << "; plus one untimed GPU priming call per mode/trial\n";
                if (config.mode != "compare") std::cout << "trials=" << config.trials << " bootstrap_samples="
                    << config.analysis.bootstrap_samples << " confidence_level=" << config.analysis.confidence_level
                    << "; primary statistic=median; reliability flags are heuristics\n";
                std::map<std::string, std::size_t> first_crossover;
                for (const auto shape : config.shapes) {
                    Matrix a(shape.m, shape.k), b(config.operation == operation::Kind::reduction_sum ? 0 : shape.k, shape.n);
                    fill_inputs(a, b, config);
                    if (config.operation == operation::Kind::matmul)
                        std::cout << "M=" << shape.m << " N=" << shape.n << " K=" << shape.k << '\n';
                    else std::cout << "length=" << shape.k << '\n';
                    Row candidate;
                    candidate.shape = shape;
                    candidate.kernel = config.candidate;
                    candidate.reference = config.reference;
                    candidate.reduction_block_size=config.reduction_block_size;
                    candidate.reference_block_size=config.reference_block_size;
                    candidate.tile_size =
                        (config.candidate == "tiled" ||
                        config.candidate == "register-blocked-2x2")
                            ? 16
                            : 0;
                    if (config.mode == "crossover") {
#ifdef MATMUL_INSPECTOR_HAS_CUDA
                        auto cpu = repeated([&] { return benchmark::cpu_matmul(a, b, config.iterations, config.warmups); }, config);
                        Row reference;
                        reference.shape = shape;
                        reference.kernel = reference.reference = "cpu";
                        reference.timed = true;
                        reference.timing_mode = "host_matmul";
                        reference.timing = cpu.timing;
                        reference.speedup = cpu.timing.mean_ms > 0 ? 1 : 0;
                        reference.median_speedup = cpu.timing.median_ms > 0 ? 1 : 0;
                        reference.comparison = comparison::compare(cpu.output, cpu.output, config.atol, config.rtol, config.max_mismatches);
                        reference.output_sha256 = reference.reference_sha256 = output::fingerprint(cpu.output);
                        reference.output_file = reference.reference_output_file = capture_output(cpu.output, config, shape, "cpu",
                            reference.output_sha256, std::to_string(rows.size()) + "-cpu");
                        rows.push_back(reference);
                        const auto kernel = cuda_kernel(config.candidate);
                        for (const std::string mode : {"kernel_only", "end_to_end"}) {
                            auto actual = repeated([&] {
                                if (mode == "kernel_only") cuda_matmul(a,b,kernel);
                                return mode == "kernel_only"
                                ? benchmark_cuda_kernel(a, b, kernel, config.iterations, config.warmups)
                                : benchmark_cuda_end_to_end(a, b, kernel, config.iterations, config.warmups); }, config);
                            Row measured = candidate;
                            measured.timed = true;
                            measured.timing_mode = mode;
                            measured.timing = actual.timing;
                            measured.speedup = actual.timing.mean_ms > 0 ? cpu.timing.mean_ms / actual.timing.mean_ms : 0;
                            measured.median_speedup = actual.timing.median_ms > 0 ? cpu.timing.median_ms / actual.timing.median_ms : 0;
                            measured.reference_sha256 = reference.output_sha256;
                            measured.reference_output_file = reference.output_file;
                            measured.output_sha256 = output::fingerprint(actual.output);
                            measured.output_file = capture_output(actual.output, config, shape, measured.kernel,
                                measured.output_sha256, std::to_string(rows.size()) + "-" + mode);
                            measured.comparison = comparison::compare(cpu.output, actual.output, config.atol, config.rtol, config.max_mismatches);
                            if (!measured.comparison.tolerance_pass) exit_code = 1;
                            const double cpu_ms = cpu.timing.median_ms, gpu_ms = actual.timing.median_ms;
                            std::cout << measured.kernel << ' ' << mode << " CPU median=" << cpu_ms << " ms GPU median=" << gpu_ms << " ms";
                            if (gpu_ms > 0 && cpu_ms > 0) {
                                std::cout << " speedup=" << cpu_ms / gpu_ms << "x";
                                if (gpu_ms < cpu_ms && !first_crossover.count(mode)) first_crossover[mode] = shape.m;
                            } else std::cout << " speedup=unavailable";
                            std::cout << " tolerance=" << (measured.comparison.tolerance_pass ? "PASS" : "FAIL") << '\n';
                            rows.push_back(std::move(measured));
                        }
#endif
                        continue;
                    }
                    if (config.operation != operation::Kind::matmul && config.mode == "benchmark") {
                        auto measure = [&](const std::string& kernel, unsigned threads) -> benchmark::HostMeasurement {
                            if (operation::is_cpu(kernel)) return benchmark::measure_host(
                                [&] { return summation::execute(kernel,config.operation,a,b); }, config.iterations,config.warmups);
#ifdef MATMUL_INSPECTOR_HAS_CUDA
                            auto result = cuda_vector(config.operation,a,b,config.iterations,config.warmups,threads);
                            return {std::move(result.output),std::move(result.timing)};
#else
                            (void)threads;
                            throw std::runtime_error("CUDA support was not built");
#endif
                        };
                        auto ref = repeated([&] { return measure(config.reference,config.reference_block_size); },config);
                        auto actual = repeated([&] { return measure(config.candidate,config.reduction_block_size); },config);
                        capture_pair(candidate,ref.output,actual.output,config,rows.size());
                        candidate.comparison = comparison::compare(ref.output,actual.output,config.atol,config.rtol,config.max_mismatches);
                        candidate.timed = true;
                        candidate.timing_mode = operation::is_cpu(config.candidate) ? "host_serial" : "kernel_only";
                        candidate.timing = actual.timing;
                        candidate.speedup = actual.timing.mean_ms > 0 ? ref.timing.mean_ms/actual.timing.mean_ms : 0;
                        candidate.median_speedup = actual.timing.median_ms > 0 ? ref.timing.median_ms/actual.timing.median_ms : 0;
                        Row reference = candidate;
                        reference.reduction_block_size=config.reference_block_size;
                        reference.kernel = config.reference;
                        reference.timing_mode = operation::is_cpu(config.reference) ? "host_serial" : "kernel_only";
                        reference.timing = ref.timing;
                        reference.speedup = reference.median_speedup = 1;
                        reference.output_sha256 = candidate.reference_sha256;
                        reference.output_file = candidate.reference_output_file;
                        reference.comparison = comparison::compare(ref.output,ref.output,config.atol,config.rtol,config.max_mismatches);
                        const auto fp64=reduction::reference_fp64(config.operation,a,b);
                        const auto forward=operation::cpu(config.operation,a,b)(0,0);
                        candidate.observations.push_back(reduction::observe(actual.output(0,0),ref.output(0,0),fp64,config.atol,config.rtol,forward));
                        reference.observations.push_back(reduction::observe(ref.output(0,0),ref.output(0,0),fp64,config.atol,config.rtol,forward));
                        if (config.reference != config.candidate) rows.push_back(reference);
                        std::cout << "reference scalar=" << ref.output(0,0) << " candidate scalar=" << actual.output(0,0)
                            << " reference median=" << ref.timing.median_ms << " ms candidate median=" << actual.timing.median_ms << " ms\n";
                        Inspector::report_comparison(candidate.comparison);
                        if (!candidate.comparison.tolerance_pass) exit_code = 1;
                        rows.push_back(candidate);
                        continue;
                    }
                    if (config.mode == "compare") {
                        Matrix reference = execute(a, b, config.reference, config.operation,config.reference_block_size);
                        Matrix actual = execute(a, b, config.candidate, config.operation,config.reduction_block_size);
                        if (config.operation != operation::Kind::matmul)
                            std::cout << "reference scalar=" << reference(0,0) << " candidate scalar=" << actual(0,0) << '\n';
                        capture_pair(candidate, reference, actual, config, rows.size());
                        candidate.comparison = comparison::compare(reference, actual, config.atol, config.rtol, config.max_mismatches);
                        Inspector::report_comparison(candidate.comparison);
                        if (config.operation != operation::Kind::matmul) {
                            const auto fp64=reduction::reference_fp64(config.operation,a,b);
                            const auto forward=operation::cpu(config.operation,a,b)(0,0);
                            for (int repeat=0;repeat<config.repeats;++repeat) {
                                if (repeat) actual=execute(a,b,config.candidate,config.operation,config.reduction_block_size);
                                auto point=reduction::observe(actual(0,0),reference(0,0),fp64,config.atol,config.rtol,forward);
                                if (!point.tolerance_pass) exit_code=1;
                                candidate.observations.push_back(point);
                            }
                            std::cout << "FP64 accumulation analysis (not exact): " << std::setprecision(17) << fp64
                                << "; repeats=" << config.repeats << " "
                                << reduction::determinism(candidate.observations,config.atol,config.rtol) << '\n';
                        }
                    } else {
#ifdef MATMUL_INSPECTOR_HAS_CUDA
                        auto ref = repeated([&] { cuda_matmul(a,b,cuda_kernel(config.reference)); return benchmark_cuda_kernel(a, b, cuda_kernel(config.reference), config.iterations, config.warmups); }, config);
                        auto actual = repeated([&] { cuda_matmul(a,b,cuda_kernel(config.candidate)); return benchmark_cuda_kernel(a, b, cuda_kernel(config.candidate), config.iterations, config.warmups); }, config);
                        // Hashing, capture and all analysis occur after both timed executions.
                        capture_pair(candidate, ref.output, actual.output, config, rows.size());
                        candidate.comparison = comparison::compare(ref.output, actual.output, config.atol, config.rtol, config.max_mismatches);
                        candidate.timed = true;
                        candidate.timing_mode = "kernel_only";
                        candidate.timing = actual.timing;
                        candidate.speedup = actual.timing.mean_ms > 0 ? ref.timing.mean_ms / actual.timing.mean_ms : 0;
                        candidate.median_speedup = actual.timing.median_ms > 0 ? ref.timing.median_ms / actual.timing.median_ms : 0;
                        Row reference = candidate;
                        reference.reduction_block_size=config.reference_block_size;
                        reference.kernel = config.reference;
                        reference.output_sha256 = candidate.reference_sha256;
                        reference.output_file = candidate.reference_output_file;
                        reference.tile_size = (config.reference == "tiled" || config.candidate == "register-blocked-2x2") ? 16 : 0;
                        reference.timing = ref.timing;
                        reference.speedup = reference.timing.mean_ms > 0 ? 1 : 0;
                        reference.median_speedup = reference.timing.median_ms > 0 ? 1 : 0;
                        reference.comparison = comparison::compare(ref.output, ref.output, config.atol, config.rtol, config.max_mismatches);
                        timing_line(reference); timing_line(candidate);
                        rows.push_back(reference);
                        std::cout << "Post-timing comparison: bitwise=" << (candidate.comparison.divergent_count == 0 ? "yes" : "no")
                            << " divergent=" << candidate.comparison.divergent_count
                            << " tolerance=" << (candidate.comparison.tolerance_pass ? "PASS" : "FAIL") << '\n';
                        if (config.legacy || controlled) Inspector::report_comparison(candidate.comparison);
#endif
                    }
                    if (!candidate.comparison.tolerance_pass) exit_code = 1;
                    rows.push_back(candidate);
                }
                if (config.mode == "crossover") {
                    for (const std::string mode : {"kernel_only", "end_to_end"}) {
                        std::cout << mode << " first sampled GPU win: ";
                        if (first_crossover.count(mode)) std::cout << "N=" << first_crossover[mode];
                        else std::cout << "none observed";
                        std::cout << " (median; no interpolation)\n";
                    }
                }
                for (const auto& row : rows) if (row.timed && !row.timing.warnings.empty())
                    std::cout << "Reliability " << row.kernel << " " << row.timing_mode << " M=" << row.shape.m
                        << (config.operation == operation::Kind::matmul ? "" : " length=" + std::to_string(row.shape.k))
                        << ": " << row.timing.warnings.size() << " heuristic flag(s); see timing_statistics.csv\n";
                values["status"] = exit_code == 0 ? "complete" : "tolerance_failed";
            }
        } catch (const std::exception& error) {
            values["status"] = "failed";
            values["error"] = error.what();
            std::cerr << "Experiment failed: " << error.what() << '\n';
            exit_code = 1;
        }
    }
    if (owns_directory) {
        try { write_artifacts(config.output, config, values, rows, console.str()); }
        catch (const std::exception& error) {
            std::cerr << "Artifact write failed: " << error.what() << '\n';
            return 1;
        }
    }
    return exit_code;
}
}  // namespace experiment
