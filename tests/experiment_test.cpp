#include "experiment.hpp"
#include "numeric.hpp"
#include "cuda_matmul.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures = 0;
void check(bool value, const char* message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class Function> void invalid(Function function) {
    try { function(); check(false, "invalid input accepted"); }
    catch (const std::invalid_argument&) {}
}

void test_config() {
    auto blas = experiment::parse({"compare", "--reference", "cublas", "--candidate", "cublas"});
    check(blas.reference == "cublas" && blas.candidate == "cublas", "cuBLAS CLI roles");
    check(experiment::contraction_mode("cublas") == "cublas_pedantic_fp32", "cuBLAS math metadata");
    check(experiment::accumulation_mode("cublas") == "cublas_unspecified_order", "cuBLAS order unspecified");
    check(experiment::metadata_json(blas,{}).find("cublas_leading_dimensions") != std::string::npos,
          "cuBLAS layout metadata");
    for (std::size_t size : {15u,16u,17u,1024u,3136u,3199u,3200u,3201u,3328u}) {
        const auto expected = size < register_blocked_auto_square_threshold
            ? CudaMatmulKernel::register_blocked_2x2 : CudaMatmulKernel::register_blocked_4x2;
        check(resolve_cuda_matmul_kernel(CudaMatmulKernel::register_blocked_auto,size,size,size) == expected,
              "auto square dispatch boundary");
        check(resolve_cuda_matmul_kernel(CudaMatmulKernel::tiled,size,size,size) == CudaMatmulKernel::tiled,
              "explicit kernel unchanged");
    }
    check(resolve_cuda_matmul_kernel(CudaMatmulKernel::register_blocked_auto,31,47,19)
          == CudaMatmulKernel::register_blocked_2x2, "auto rectangular fallback");
    check(resolve_cuda_matmul_kernel(CudaMatmulKernel::register_blocked_auto,4096,4096,17)
          == CudaMatmulKernel::register_blocked_2x2, "square output with nonsquare inputs uses fallback");
    auto automatic = experiment::parse({"compare", "--reference", "register-blocked-auto",
                                       "--candidate", "register-blocked-auto", "--sizes", "3199,3200,3201"});
    const auto metadata = experiment::metadata_json(automatic,{});
    check(metadata.find("resolved_candidate") != std::string::npos
          && metadata.find("register-blocked-2x2") != std::string::npos
          && metadata.find("register-blocked-4x2") != std::string::npos, "auto per-shape metadata");
    auto blocked = experiment::parse({"compare", "--reference", "register-blocked-4x4",
                                     "--candidate", "register-blocked-4x4"});
    check(blocked.reference == "register-blocked-4x4" && blocked.candidate == blocked.reference,
          "4x4 accepted in both CLI roles");
    check(experiment::accumulation_mode(blocked.candidate) == "increasing_k_zero_padded_tiles",
          "4x4 tiled accumulation metadata");
    blocked = experiment::parse({"compare", "--reference", "register-blocked-4x2",
                                     "--candidate", "register-blocked-4x2"});
    check(blocked.reference == "register-blocked-4x2" && blocked.candidate == blocked.reference,
          "4x2 accepted in both CLI roles");
    check(experiment::accumulation_mode(blocked.candidate) == "increasing_k_zero_padded_tiles",
          "4x2 tiled accumulation metadata");
    auto config = experiment::parse({"benchmark", "--sizes", "4,257", "--seed", "0", "--warmups", "0", "--iterations", "2"});
    check(config.shapes.size() == 2 && config.shapes[1].k == 257, "square shapes");
    check(config.seed == 0 && config.seed_b == 81 && config.warmups == 0 && config.iterations == 2, "explicit counts and seeds");
    config = experiment::parse({"compare", "--m", "2", "--n", "3", "--k", "5", "--reference", "cpu", "--candidate", "cpu", "--seed", "4294967295", "--atol", "0"});
    check(config.shapes[0].m == 2 && config.shapes[0].n == 3 && config.shapes[0].k == 5, "rectangular shape");
    check(config.seed_b == 80 && config.atol == 0, "seed wrap and zero tolerance");
    check(experiment::parse({"compare", "--seed-b", "7", "--seed", "8"}).seed_b == 7, "explicit B seed independent of option order");
    check(experiment::parse({"--benchmark-cuda", "50"}).iterations == 50, "legacy alias");
    for (const auto& args : std::vector<std::vector<std::string>>{
        {}, {"bad"}, {"compare", "--sizes", ""}, {"compare", "--sizes", "4,"},
        {"compare", "--m", "2"}, {"compare", "--m", "2", "--n", "2", "--k", "2", "--sizes", "4"},
        {"compare", "--seed", "-1"}, {"compare", "--seed", "4294967296"}, {"compare", "--seed", "2x"},
        {"compare", "--seed", "1", "--seed", "2"}, {"compare", "--atol", "nan"}, {"compare", "--rtol", "inf"},
        {"compare", "--atol", "-1"}, {"compare", "--iterations", "1"}, {"benchmark", "--iterations", "0"},
        {"benchmark", "--warmups", "-1"}, {"benchmark", "--reference", "cpu"}, {"compare", "--unknown", "1"},
        {"compare", "--seed"}, {"compare", "--candidate", "bad"}}) {
        invalid([&] { experiment::parse(args); });
    }
}

void test_generation_and_comparison() {
    Matrix a(2, 3), b(2, 3, 5), other(2, 3);
    experiment::fill(a, 42); experiment::fill(b, 42); experiment::fill(other, 43);
    auto same = comparison::compare(a, b);
    check(same.divergent_count == 0 && same.tolerance_pass, "seeded generation independent of stride");
    check(comparison::compare(a, other).divergent_count > 0, "different seed changes input");
    check(b.data()[3] == 0 && b.data()[4] == 0, "padding unchanged");
    // Fixed generator contract, not std::uniform_real_distribution's implementation.
    std::uint32_t state = 42;
    state = 1664525u * state + 1013904223u;
    check(state == 1083814273u, "LCG32 first state");
    check(numeric::bitwise_equal(a(0,0), 0.24f), "generator golden first value");
    Matrix reference(1, 3), candidate(1, 3);
    reference(0,0) = candidate(0,0) = -1;
    candidate(0,0) = std::nextafter(-1.0f, -2.0f);
    candidate(0,1) = 1;
    candidate(0,2) = -0.0f;
    auto result = comparison::compare(reference, candidate);
    check(result.divergent_count == 3 && result.first_bitwise->col == 0 && result.first_numeric->col == 1,
          "counts and independent first divergences");
    check(numeric::ulp_distance(reference(0,0), candidate(0,0)) == 1, "negative float ULP");
    invalid([&] { comparison::compare(a, reference); });
    invalid([&] { comparison::compare(a, b, -1); });
}

void test_statistics() {
    const auto stats = benchmark::statistics({1, 2, 3, 4});
    check(stats.mean_ms == 2.5 && stats.median_ms == 2.5 && stats.min_ms == 1, "timing statistics");
    check(std::fabs(stats.stddev_ms - std::sqrt(1.25)) < 1e-12, "population stddev");
    check(benchmark::statistics({3,1,2}).median_ms == 2, "odd median");
    check(benchmark::statistics({}).mean_ms == 0 && benchmark::statistics({2}).stddev_ms == 0, "empty and singleton statistics");
    invalid([] { benchmark::statistics({-1}); });
    invalid([] { benchmark::statistics({std::numeric_limits<float>::infinity()}); });
}

void fixture(const std::filesystem::path& path) {
    experiment::Config config;
    config.reference = "cpu"; config.candidate = "cpu";
    config.shapes = {{1,2,1}};
    Matrix reference(1,2), candidate(1,2);
    reference(0,0) = 1; candidate(0,0) = std::nextafter(1.0f, 2.0f); candidate(0,1) = 1;
    experiment::Row row;
    row.shape = config.shapes[0]; row.kernel = "quoted,\"kernel"; row.reference = "cpu";
    row.comparison = comparison::compare(reference, candidate);
    auto metadata = experiment::metadata();
    metadata["escape_test"] = "quote\" slash\\ newline\n tab\t control\x01";
    metadata["status"] = "complete";
    experiment::create_run_directory(path);
    experiment::write_artifacts(path, config, metadata, {row}, "fixture console\n");
}
}

int main(int argc, char** argv) {
    try {
        test_config(); test_generation_and_comparison(); test_statistics();
        auto metadata = experiment::metadata();
        check(metadata.count("git_commit") && metadata.count("git_dirty") && metadata.count("timestamp_utc"), "metadata keys");
        check(!metadata["timestamp_utc"].empty(), "timestamp");
        if (argc == 2) fixture(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
    return failures == 0 ? 0 : 1;
}
