#include "tridiagonal.hpp"
#include "tridiagonal_benchmark.hpp"
#include "tridiagonal_benchmark_cli.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

    std::vector<double> apply_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& x) {

    const std::size_t n = diag.size();
    std::vector<double> rhs(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] += diag[i] * x[i];

        if (i > 0) {
            rhs[i] += lower[i - 1] * x[i - 1];
        }

        if (i + 1 < n) {
            rhs[i] += upper[i] * x[i + 1];
        }
    }

    return rhs;
}

bool approximately_equal(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

bool test_size(std::size_t n) {
    std::vector<double> lower(n > 0 ? n - 1 : 0, -1.0);
    std::vector<double> diag(n, 4.0);
    std::vector<double> upper(n > 0 ? n - 1 : 0, -1.0);

    std::vector<double> expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = static_cast<double>(i + 1);
    }

    const auto rhs =
        apply_tridiagonal(lower, diag, upper, expected);

    const auto thomas =
        matmul_inspector::solve_thomas(
            lower, diag, upper, rhs);

    const auto pcr =
        matmul_inspector::solve_pcr(
            lower, diag, upper, rhs);

    constexpr double tolerance = 1e-10;

    for (std::size_t i = 0; i < n; ++i) {
        if (!approximately_equal(
                thomas[i], expected[i], tolerance)) {

            std::cerr
                << "Thomas failed for n=" << n
                << " at index " << i
                << ": expected " << expected[i]
                << ", got " << thomas[i] << '\n';

            return false;
        }

        if (!approximately_equal(
                pcr[i], expected[i], tolerance)) {

            std::cerr
                << "PCR failed for n=" << n
                << " at index " << i
                << ": expected " << expected[i]
                << ", got " << pcr[i] << '\n';

            return false;
        }
    }

    return true;
}

bool test_known_system() {
    const std::vector<double> lower{-1.0, -1.0, -1.0};
    const std::vector<double> diag{2.0, 2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0, -1.0};
    const std::vector<double> rhs{0.0, 0.0, 0.0, 5.0};

    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};

    const auto actual =
        matmul_inspector::solve_thomas(lower, diag, upper, rhs);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approximately_equal(actual[i], expected[i])) {
            std::cerr
                << "Known-system mismatch at index " << i
                << ": expected " << expected[i]
                << ", got " << actual[i] << '\n';
            return false;
        }
    }

    return true;
}

bool test_single_equation() {
    const std::vector<double> lower{};
    const std::vector<double> diag{4.0};
    const std::vector<double> upper{};
    const std::vector<double> rhs{12.0};

    const auto actual =
        matmul_inspector::solve_thomas(lower, diag, upper, rhs);

    if (actual.size() != 1 || !approximately_equal(actual[0], 3.0)) {
        std::cerr << "1x1 system failed\n";
        return false;
    }

    return true;
}

bool test_invalid_dimensions() {
    try {
        const std::vector<double> lower{-1.0};
        const std::vector<double> diag{2.0, 2.0, 2.0};
        const std::vector<double> upper{-1.0, -1.0};
        const std::vector<double> rhs{1.0, 2.0, 3.0};

        (void)matmul_inspector::solve_thomas(
            lower, diag, upper, rhs);
    } catch (const std::invalid_argument&) {
        return true;
    }

    std::cerr << "Invalid dimensions did not throw\n";
    return false;
}

bool test_zero_pivot() {
    try {
        const std::vector<double> lower{1.0};
        const std::vector<double> diag{0.0, 2.0};
        const std::vector<double> upper{1.0};
        const std::vector<double> rhs{1.0, 1.0};

        (void)matmul_inspector::solve_thomas(
            lower, diag, upper, rhs);
    } catch (const std::runtime_error&) {
        return true;
    }

    std::cerr << "Zero pivot did not throw\n";
    return false;
}

bool test_pcr_known_system() {
    const std::vector<double> lower{-1.0, -1.0, -1.0};
    const std::vector<double> diag{2.0, 2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0, -1.0};
    const std::vector<double> rhs{0.0, 0.0, 0.0, 5.0};

    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};

    const auto actual =
        matmul_inspector::solve_pcr(lower, diag, upper, rhs);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approximately_equal(actual[i], expected[i])) {
            std::cerr
                << "PCR known-system mismatch at index " << i
                << ": expected " << expected[i]
                << ", got " << actual[i] << '\n';
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    if (!test_known_system()) {
        return 1;
    }

    if (!test_single_equation()) {
        return 1;
    }

    if (!test_invalid_dimensions()) {
        return 1;
    }

    if (!test_zero_pivot()) {
        return 1;
    }

    if (!test_pcr_known_system()) {
        return 1;
    }

    const std::vector<std::size_t> sizes{
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 100
    };

    for (const auto n : sizes) {
        if (!test_size(n)) {
            return 1;
        }
    }

    const auto systems=matmul_inspector::make_benchmark_systems(33,3);
    const auto thomas=matmul_inspector::benchmark_thomas_batch(systems,3,1);
    const auto pcr=matmul_inspector::benchmark_cpu_pcr_batch(systems,3,1);
    if (thomas.algorithm!="thomas" || pcr.algorithm!="pcr" ||
        thomas.backend!="cpu" || thomas.timing_scope!="end_to_end" ||
        thomas.batch_execution!="serial_host_loop" ||
        thomas.system_size!=33 || thomas.batch_size!=3 ||
        thomas.timing.samples.size()!=4 || thomas.timing.max_ms<thomas.timing.min_ms ||
        pcr.timing.samples.size()!=4) {
        std::cerr << "CPU tridiagonal benchmark metadata/statistics failed\n";
        return 1;
    }
    const auto header=matmul_inspector::tridiagonal_benchmark_csv_header();
    const auto row=matmul_inspector::tridiagonal_benchmark_csv_row(thomas);
    if (header.find("timing_scope") == std::string::npos ||
        header.find("median_ms") == std::string::npos ||
        header.find("batch_execution") == std::string::npos ||
        row.find("thomas,cpu,end_to_end,serial_host_loop,33,3,1,3") != 0) {
        std::cerr << "Tridiagonal benchmark CSV failed\n";
        return 1;
    }
    try {
        matmul_inspector::make_benchmark_systems(0,1);
        std::cerr << "Invalid benchmark size accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    using Mode=matmul_inspector::TridiagonalBenchmarkMode;
    const auto defaults=matmul_inspector::parse_tridiagonal_benchmark_cli({});
    if (defaults.system_sizes!=std::vector<std::size_t>{32,64,128,256,512,1024,2048,4096} ||
        defaults.batch_sizes!=std::vector<std::size_t>{1,8,32,128,512} ||
        defaults.mode!=Mode::all || defaults.warmups!=3 || defaults.iterations!=20) {
        std::cerr << "Tridiagonal benchmark CLI defaults changed\n";
        return 1;
    }
    const auto single_size=matmul_inspector::parse_tridiagonal_benchmark_cli(
        {"--system-size","33"});
    const auto single_batch=matmul_inspector::parse_tridiagonal_benchmark_cli(
        {"--batch-size","3"});
    const auto combined=matmul_inspector::parse_tridiagonal_benchmark_cli(
        {"--system-size","33","--batch-size","3","--mode","cpu"});
    if (single_size.system_sizes!=std::vector<std::size_t>{33} ||
        single_size.batch_sizes!=defaults.batch_sizes ||
        single_batch.system_sizes!=defaults.system_sizes ||
        single_batch.batch_sizes!=std::vector<std::size_t>{3} ||
        combined.system_sizes!=std::vector<std::size_t>{33} ||
        combined.batch_sizes!=std::vector<std::size_t>{3} || combined.mode!=Mode::cpu) {
        std::cerr << "Tridiagonal benchmark CLI filters failed\n";
        return 1;
    }
    const std::vector<std::pair<std::string,Mode>> modes{
        {"all",Mode::all},
        {"cpu",Mode::cpu},
        {"serial_host_loop",Mode::serial_host_loop},
        {"true_batched_gpu",Mode::true_batched_gpu},
        {"true_batched_reuse",Mode::true_batched_reuse},
        {"true_batched_device_resident",Mode::true_batched_device_resident},
        {"true_batched_hybrid",Mode::true_batched_hybrid}
    };
    for (const auto& [name,expected] : modes) {
        const auto parsed=matmul_inspector::parse_tridiagonal_benchmark_cli(
            {"--mode",name});
        if (parsed.mode!=expected ||
            std::string(matmul_inspector::tridiagonal_benchmark_mode_name(expected))!=name ||
            !matmul_inspector::tridiagonal_benchmark_mode_includes(Mode::all,expected) ||
            !matmul_inspector::tridiagonal_benchmark_mode_includes(expected,expected)) {
            std::cerr << "Tridiagonal benchmark mode parsing failed for " << name << '\n';
            return 1;
        }
    }
    for (const auto& arguments : std::vector<std::vector<std::string>>{
             {"--system-size","0"},{"--system-size","-1"},
             {"--batch-size","0"},{"--batch-size","-1"},
             {"--mode","unknown"}}) {
        try {
            matmul_inspector::parse_tridiagonal_benchmark_cli(arguments);
            std::cerr << "Invalid tridiagonal benchmark CLI arguments accepted\n";
            return 1;
        } catch (const std::invalid_argument&) {}
    }
    if (matmul_inspector::tridiagonal_benchmark_help().find("--system-size") ==
        std::string::npos) {
        std::cerr << "Tridiagonal benchmark CLI help is incomplete\n";
        return 1;
    }

    std::cout << "All Thomas solver tests passed\n";
    return 0;
}
