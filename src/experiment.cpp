#include "experiment.hpp"
#include "numeric.hpp"
#include "cuda_matmul.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace experiment {
namespace {
std::uint64_t integer(const std::string& value, std::uint64_t maximum, bool zero = false) {
    std::uint64_t result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
        || result > maximum || (!zero && result == 0)) {
        throw std::invalid_argument("Invalid integer: " + value);
    }
    return result;
}
float tolerance(const std::string& text) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    float result;
    if (!(stream >> result) || !stream.eof() || !std::isfinite(result) || result < 0) {
        throw std::invalid_argument("Tolerance must be finite and nonnegative: " + text);
    }
    return result;
}
void write_file(const std::filesystem::path& path, const std::string& text) {
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << text;
    out.close();
    std::filesystem::rename(temporary, path);
}
}  // namespace

std::string usage() {
    return "Usage: matmul-inspector [compare|benchmark] [options]\n"
           "  --sizes 4,256,257,1024   Square sizes (compare default: 4)\n"
           "  --m M --n N --k K      One rectangular configuration; exclusive with --sizes\n"
           "  --reference KERNEL --candidate KERNEL (defaults naive, tiled)\n"
           "    Kernels: cpu (compare only), naive, tiled, cuda-naive-fma,\n"
           "             cuda-naive-no-fma, cuda-naive-reordered\n"
           "  --input random|cancellation|fma-sensitive (default random)\n"
           "  --max-mismatches 0..1000  Saved samples per configuration (default 100)\n"
           "  --seed UINT32          A seed (default 42); B defaults to seed + 81 modulo 2^32\n"
           "  --seed-b UINT32        Explicit B seed (default 123 with seed 42)\n"
           "  --atol FLOAT --rtol FLOAT  Nonnegative tolerances (defaults 1e-6, 1e-5)\n"
           "  --warmups INT --iterations INT  Benchmark only (defaults 3, 20)\n"
           "  --output DIRECTORY     Save artifacts in a new directory\n"
           "  --help                 Show this help\n"
           "No arguments: original CPU/GPU demo. Legacy: --benchmark-cuda [iterations].\n";
}

Config parse(const std::vector<std::string>& args) {
    if (args.empty()) throw std::invalid_argument("Missing command");
    Config config;
    if (args[0] == "--benchmark-cuda") {
        if (args.size() > 2) throw std::invalid_argument("Too many legacy arguments");
        config.mode = "benchmark";
        config.legacy = true;
        config.shapes = {{4,4,4}, {256,256,256}, {257,257,257}, {1024,1024,1024}};
        if (args.size() == 2) config.iterations = static_cast<int>(integer(args[1], std::numeric_limits<int>::max()));
        return config;
    }
    config.mode = args[0];
    if (config.mode != "compare" && config.mode != "benchmark") throw std::invalid_argument("Unknown command: " + args[0]);
    if (config.mode == "benchmark") config.shapes = {{4,4,4}, {256,256,256}, {257,257,257}, {1024,1024,1024}};
    std::set<std::string> seen;
    Shape shape{0,0,0};
    for (std::size_t i = 1; i < args.size(); i += 2) {
        const auto& option = args[i];
        if (i + 1 == args.size()) throw std::invalid_argument("Missing value for " + option);
        if (!seen.insert(option).second) throw std::invalid_argument("Duplicate option: " + option);
        const auto& value = args[i + 1];
        if (option == "--sizes") {
            config.shapes.clear();
            std::size_t begin = 0;
            do {
                auto end = value.find(',', begin);
                auto size = static_cast<std::size_t>(integer(value.substr(begin, end - begin), std::numeric_limits<std::size_t>::max()));
                config.shapes.push_back({size, size, size});
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
        } else if (option == "--m" || option == "--n" || option == "--k") {
            auto size = static_cast<std::size_t>(integer(value, std::numeric_limits<std::size_t>::max()));
            if (option == "--m") shape.m = size;
            if (option == "--n") shape.n = size;
            if (option == "--k") shape.k = size;
        } else if (option == "--seed" || option == "--seed-b") {
            auto seed = static_cast<std::uint32_t>(integer(value, UINT32_MAX, true));
            if (option == "--seed") config.seed = seed;
            else config.seed_b = seed;
        } else if (option == "--iterations" || option == "--warmups") {
            if (config.mode != "benchmark") throw std::invalid_argument(option + " requires benchmark mode");
            auto count = static_cast<int>(integer(value, std::numeric_limits<int>::max(), option == "--warmups"));
            if (option == "--iterations") config.iterations = count;
            else config.warmups = count;
        } else if (option == "--atol") config.atol = tolerance(value);
        else if (option == "--rtol") config.rtol = tolerance(value);
        else if (option == "--output") {
            if (value.empty()) throw std::invalid_argument("Empty output path");
            config.output = value;
        } else if (option == "--reference" || option == "--candidate") {
            if (value != "cpu" && value != "naive" && value != "tiled"
                && value != "cuda-naive-fma" && value != "cuda-naive-no-fma" && value != "cuda-naive-reordered") throw std::invalid_argument("Unknown kernel: " + value);
            if (option == "--reference") config.reference = value;
            else config.candidate = value;
        } else if (option == "--input") {
            if (value != "random" && value != "cancellation" && value != "fma-sensitive")
                throw std::invalid_argument("Unknown input: " + value);
            config.input = value;
        } else if (option == "--max-mismatches") config.max_mismatches = integer(value, 1000, true);
        else throw std::invalid_argument("Unknown option: " + option);
    }
    if (shape.m || shape.n || shape.k) {
        if (seen.count("--sizes") || !shape.m || !shape.n || !shape.k) throw std::invalid_argument("Specify either --sizes or all of --m, --n, --k");
        config.shapes = {shape};
    }
    if (config.mode == "benchmark" && (config.reference == "cpu" || config.candidate == "cpu"))
        throw std::invalid_argument("Benchmark requires CUDA kernels");
    for (const auto& dimensions : config.shapes) {
        if ((config.input == "cancellation" && dimensions.k < 4) || (config.input == "fma-sensitive" && dimensions.k < 2))
            throw std::invalid_argument("Sensitive input requires K >= 4 (cancellation) or K >= 2 (fma-sensitive)");
    }
    if (!seen.count("--seed-b")) config.seed_b = config.seed + std::uint32_t{81};
    return config;
}

void fill(Matrix& matrix, std::uint32_t seed) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            seed = 1664525u * seed + 1013904223u;
            matrix(row, col) = static_cast<float>(static_cast<int>((seed >> 16) % 101) - 50) / 100.0f;
        }
    }
}

std::string contraction_mode(const std::string& kernel) {
    if (kernel == "cuda-naive-no-fma") return "separate_rn_mul_add";
    if (kernel == "cuda-naive-fma" || kernel == "cuda-naive-reordered") return "explicit_fma_rn";
    return "compiler_default";
}

std::string accumulation_mode(const std::string& kernel) {
    if (kernel == "cuda-naive-reordered") return "even_odd_partials";
    if (kernel == "tiled") return "increasing_k_zero_padded_tiles";
    return "increasing_k";
}

void fill_inputs(Matrix& a, Matrix& b, const Config& config) {
    if (a.cols() != b.rows()) throw std::invalid_argument("Input dimensions do not agree");
    if (config.input == "random") { fill(a, config.seed); fill(b, config.seed_b); return; }
    if (config.input != "cancellation" && config.input != "fma-sensitive")
        throw std::invalid_argument("Unknown input generator");
    const float cancellation[] = {16777216.0f, 1.0f, -16777216.0f, 1.0f};
    for (std::size_t row = 0; row < a.rows(); ++row)
        for (std::size_t k = 0; k < a.cols(); ++k)
            a(row,k) = config.input == "cancellation" ? cancellation[k % 4]
                : (k == 0 ? -1.0f : (k == 1 ? 0x1.000002p0f : 0.0f));
    for (std::size_t k = 0; k < b.rows(); ++k)
        for (std::size_t col = 0; col < b.cols(); ++col)
            b(k,col) = config.input == "cancellation" ? 1.0f : (k == 1 ? 0x1.fffffcp-1f : 1.0f);
}

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch);
        else out << ch;
    }
    out << '"';
    return out.str();
}

std::string csv_field(const std::string& value) {
    std::string result = "\"";
    for (char ch : value) { if (ch == '"') result += '"'; result += ch; }
    return result + '"';
}

std::string metadata_json(const Config& config, const Metadata& values) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n  \"schema_version\": 2";
    for (const auto& [key, value] : values) {
        out << ",\n  " << json_string(key) << ": ";
        if ((key.find("dirty") != std::string::npos || key.find("fast_math") != std::string::npos || key == "cuda_support" || key == "fp_verified" || key == "mismatches_truncated")
            && (value == "true" || value == "false")) out << value;
        else out << json_string(value);
    }
    out << ",\n  \"config\": {\n    \"mode\": " << json_string(config.mode)
        << ", \"dtype\": \"float32\", \"layout\": \"row_major_packed\""
        << ",\n    \"reference\": " << json_string(config.reference) << ", \"candidate\": " << json_string(config.candidate)
        << ", \"tile_size\": " << cuda_matmul_tile_size
        << ",\n    \"seed\": " << config.seed << ", \"seed_b\": " << config.seed_b
        << ", \"generator\": " << json_string(config.input == "random" ? "lcg32-v1" : config.input + "-v1")
        << ", \"input\": " << json_string(config.input)
        << ", \"max_mismatches\": " << config.max_mismatches
        << ",\n    \"reference_contraction\": " << json_string(contraction_mode(config.reference))
        << ", \"candidate_contraction\": " << json_string(contraction_mode(config.candidate))
        << ",\n    \"reference_accumulation\": " << json_string(accumulation_mode(config.reference))
        << ", \"candidate_accumulation\": " << json_string(accumulation_mode(config.candidate))
        << ",\n    \"warmups\": " << (config.mode == "benchmark" ? config.warmups : 0)
        << ", \"iterations\": " << (config.mode == "benchmark" ? config.iterations : 0)
        << ", \"atol\": " << config.atol << ", \"rtol\": " << config.rtol
        << ",\n    \"shapes\": [";
    for (std::size_t i = 0; i < config.shapes.size(); ++i) {
        const auto shape = config.shapes[i];
        if (i) out << ',';
        out << "{\"M\":" << shape.m << ",\"N\":" << shape.n << ",\"K\":" << shape.k << '}';
    }
    out << "]\n  }\n}\n";
    return out.str();
}

void create_run_directory(const std::filesystem::path& path) {
    if (path.empty()) return;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    if (!std::filesystem::create_directory(path)) throw std::runtime_error("Output directory already exists: " + path.string());
}

void write_artifacts(const std::filesystem::path& path, const Config& config,
                     const Metadata& values, const std::vector<Row>& rows, const std::string& console) {
    if (path.empty()) return;
    std::ostringstream summary, mismatches;
    summary.imbue(std::locale::classic()); mismatches.imbue(std::locale::classic());
    summary << std::setprecision(17) << std::boolalpha;
    mismatches << std::setprecision(17) << std::boolalpha;
    summary << "M,N,K,kernel,tile_size,mean_ms,median_ms,min_ms,stddev_ms,gflops,reference_kernel,speedup,bitwise_equal,divergent_count,tolerance_pass,contraction_mode,accumulation_mode,reference_contraction,reference_accumulation,divergent_percent,max_ulp,mean_divergent_ulp,max_absolute_error,max_relative_error,tolerance_failures,tolerance_failure_percent,ulp_0,ulp_1,ulp_2,ulp_3_4,ulp_5_8,ulp_gt_8,finite_pairs,finite_divergent_count,nan_pairs,infinity_pairs,zero_reference_nonzero,mismatch_count,mismatches_saved,mismatches_truncated\n";
    mismatches << "M,N,K,kernel,reference_kernel,kind,row,col,reference_value,candidate_value,reference_bits,candidate_bits,ulp_distance,absolute_error,relative_error,tolerance,tolerance_pass\n";
    bool any_mismatch = false;
    for (const auto& row : rows) {
        summary << row.shape.m << ',' << row.shape.n << ',' << row.shape.k << ',' << csv_field(row.kernel) << ',' << row.tile_size << ',';
        if (row.timed) {
            summary << row.timing.mean_ms << ',' << row.timing.median_ms << ',' << row.timing.min_ms << ',' << row.timing.stddev_ms << ',';
            if (row.timing.mean_ms > 0) summary << 2.0 * row.shape.m * row.shape.n * row.shape.k / (row.timing.mean_ms * 1e6);
        } else summary << ",,,,";
        summary << ',' << csv_field(row.reference) << ',';
        if (row.timed && row.speedup > 0) summary << row.speedup;
        summary << ',' << (row.comparison.divergent_count == 0) << ',' << row.comparison.divergent_count << ',' << row.comparison.tolerance_pass;
        const auto& c = row.comparison;
        summary << ',' << csv_field(contraction_mode(row.kernel)) << ',' << csv_field(accumulation_mode(row.kernel))
            << ',' << csv_field(contraction_mode(row.reference)) << ',' << csv_field(accumulation_mode(row.reference))
            << ',' << c.divergent_percent << ',' << c.max_ulp << ',' << c.mean_divergent_ulp
            << ',' << c.max_absolute_error << ',' << c.max_relative_error << ',' << c.tolerance_failures
            << ',' << c.tolerance_failure_percent;
        for (auto bin : c.ulp_bins) summary << ',' << bin;
        summary << ',' << c.finite_pairs << ',' << c.finite_divergent_count << ',' << c.nan_pairs
            << ',' << c.infinity_pairs << ',' << c.zero_reference_nonzero << ',' << c.mismatch_count
            << ',' << c.mismatches.size() << ',' << c.mismatches_truncated << '\n';
        auto mismatch = [&](const std::optional<comparison::Divergence>& point, const char* kind) {
            if (!point) return;
            any_mismatch = true;
            float a = point->candidate, e = point->reference;
            mismatches << row.shape.m << ',' << row.shape.n << ',' << row.shape.k << ','
                << csv_field(row.kernel) << ',' << csv_field(row.reference) << ',' << kind << ',' << point->row << ',' << point->col << ','
                << e << ',' << a << ',' << csv_field(numeric::float_bits(e).to_string()) << ',' << csv_field(numeric::float_bits(a).to_string()) << ','
                << numeric::ulp_distance(a, e) << ',' << numeric::absolute_error(a, e) << ',' << numeric::relative_error(a, e) << ','
                << numeric::comparison_tolerance(a, e, row.comparison.atol, row.comparison.rtol) << ','
                << numeric::nearly_equal(a, e, row.comparison.atol, row.comparison.rtol) << '\n';
        };
        for (const auto& point : c.mismatches) {
            const char* kind = "sample";
            if (c.first_numeric && point.row == c.first_numeric->row && point.col == c.first_numeric->col) kind = "first_numeric";
            if (c.first_bitwise && point.row == c.first_bitwise->row && point.col == c.first_bitwise->col) kind = "first_bitwise";
            mismatch(point, kind);
        }
    }
    write_file(path / "summary.csv", summary.str());
    if (any_mismatch) write_file(path / "mismatches.csv", mismatches.str());
    write_file(path / "console.txt", console);
    // Metadata is the completion marker and is written last.
    auto recorded = values;
    bool truncated = false;
    for (const auto& row : rows) truncated = truncated || row.comparison.mismatches_truncated;
    recorded["mismatches_truncated"] = truncated ? "true" : "false";
    write_file(path / "metadata.json", metadata_json(config, recorded));
}
}  // namespace experiment
