#include "experiment.hpp"
#include "inspector.hpp"
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
    if (name == "cuda-naive-fma") return CudaMatmulKernel::naive_fma;
    if (name == "cuda-naive-no-fma") return CudaMatmulKernel::naive_no_fma;
    if (name == "cuda-naive-reordered") return CudaMatmulKernel::naive_reordered;
    throw std::invalid_argument("Unknown CUDA kernel: " + name);
}
#endif

Matrix execute(const Matrix& a, const Matrix& b, const std::string& kernel) {
    if (kernel == "cpu") return matmul(a, b);
#ifdef MATMUL_INSPECTOR_HAS_CUDA
    return cuda_matmul(a, b, cuda_kernel(kernel));
#else
    throw std::runtime_error("CUDA support was not built");
#endif
}

#ifdef MATMUL_INSPECTOR_HAS_CUDA
void timing_line(const Row& row) {
    const double gflops = row.timing.mean_ms > 0
        ? 2.0 * row.shape.m * row.shape.n * row.shape.k / (row.timing.mean_ms * 1e6) : 0;
    std::cout << std::setprecision(9) << row.kernel << " CUDA kernel: mean=" << row.timing.mean_ms
        << " ms median=" << row.timing.median_ms << " ms min=" << row.timing.min_ms
        << " ms stddev=" << row.timing.stddev_ms << " ms GFLOP/s=" << gflops << " speedup=" << row.speedup << "x\n";
}
#endif
}  // namespace

int run_cli(const std::vector<std::string>& arguments) {
    if (arguments == std::vector<std::string>{"--help"}
        || (arguments.size() == 2 && (arguments[0] == "compare" || arguments[0] == "benchmark") && arguments[1] == "--help")) {
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
            create_run_directory(config.output);
            owns_directory = !config.output.empty();
            std::cout << config.mode << ": reference=" << config.reference << " candidate=" << config.candidate
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
            const bool needs_gpu = config.reference != "cpu" || config.candidate != "cpu";
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
                if (config.mode == "benchmark") std::cout << "CUDA events; warmups=" << config.warmups << " iterations=" << config.iterations
                    << "; allocation/transfers/comparison excluded\n";
                for (const auto shape : config.shapes) {
                    Matrix a(shape.m, shape.k), b(shape.k, shape.n);
                    fill_inputs(a, b, config);
                    std::cout << "M=" << shape.m << " N=" << shape.n << " K=" << shape.k << '\n';
                    Row candidate;
                    candidate.shape = shape;
                    candidate.kernel = config.candidate;
                    candidate.reference = config.reference;
                    candidate.tile_size = config.candidate == "tiled" ? 16 : 0;
                    if (config.mode == "compare") {
                        Matrix reference = execute(a, b, config.reference);
                        Matrix actual = execute(a, b, config.candidate);
                        candidate.comparison = comparison::compare(reference, actual, config.atol, config.rtol, config.max_mismatches);
                        Inspector::report_comparison(candidate.comparison);
                    } else {
#ifdef MATMUL_INSPECTOR_HAS_CUDA
                        auto ref = benchmark_cuda_kernel(a, b, cuda_kernel(config.reference), config.iterations, config.warmups);
                        auto actual = benchmark_cuda_kernel(a, b, cuda_kernel(config.candidate), config.iterations, config.warmups);
                        // No analysis or file/console output occurs until all timing is done.
                        candidate.comparison = comparison::compare(ref.output, actual.output, config.atol, config.rtol, config.max_mismatches);
                        candidate.timed = true;
                        candidate.timing = actual.timing;
                        candidate.speedup = actual.timing.mean_ms > 0 ? ref.timing.mean_ms / actual.timing.mean_ms : 0;
                        Row reference = candidate;
                        reference.kernel = config.reference;
                        reference.tile_size = config.reference == "tiled" ? 16 : 0;
                        reference.timing = ref.timing;
                        reference.speedup = reference.timing.mean_ms > 0 ? 1 : 0;
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
