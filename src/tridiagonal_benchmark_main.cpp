#include "tridiagonal_benchmark.hpp"
#include "tridiagonal_benchmark_cli.hpp"
#ifdef MATMUL_INSPECTOR_HAS_CUDA
#include "cuda_tridiagonal_benchmark.hpp"
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    try {
        std::vector<std::string> arguments;
        for (int i=1; i<argc; ++i) arguments.emplace_back(argv[i]);
        const auto config=matmul_inspector::parse_tridiagonal_benchmark_cli(arguments);
        if (config.show_help) {
            std::cout << matmul_inspector::tridiagonal_benchmark_help();
            return 0;
        }
#ifndef MATMUL_INSPECTOR_HAS_CUDA
        if (config.mode!=matmul_inspector::TridiagonalBenchmarkMode::all &&
            config.mode!=matmul_inspector::TridiagonalBenchmarkMode::cpu)
            throw std::runtime_error("Selected benchmark mode requires CUDA support");
#endif
        const std::filesystem::path output=config.output;
        if (!std::filesystem::create_directories(output) || !std::filesystem::is_empty(output))
            throw std::runtime_error("Output directory exists or cannot be created: "+output.string());
        std::ofstream csv(output/"summary.csv");
        if (!csv) throw std::runtime_error("Cannot create summary.csv");
        csv << matmul_inspector::tridiagonal_benchmark_csv_header();
        using Mode=matmul_inspector::TridiagonalBenchmarkMode;
        for (const auto n : config.system_sizes) for (const auto batch : config.batch_sizes) {
            const auto systems=matmul_inspector::make_benchmark_systems(n,batch);
            if (matmul_inspector::tridiagonal_benchmark_mode_includes(config.mode,Mode::cpu))
                csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_thomas_batch(
                               systems,config.iterations,config.warmups))
                    << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_cpu_pcr_batch(
                               systems,config.iterations,config.warmups));
#ifdef MATMUL_INSPECTOR_HAS_CUDA
            if (matmul_inspector::tridiagonal_benchmark_mode_includes(
                    config.mode,Mode::serial_host_loop))
                csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_cuda_pcr_kernel_only(
                               systems,config.iterations,config.warmups))
                    << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_cuda_pcr_end_to_end(
                               systems,config.iterations,config.warmups));
            if (matmul_inspector::tridiagonal_benchmark_mode_includes(
                    config.mode,Mode::true_batched_gpu))
                csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_cuda_pcr_true_batched_kernel_only(
                               systems,config.iterations,config.warmups))
                    << matmul_inspector::tridiagonal_benchmark_csv_row(
                           matmul_inspector::benchmark_cuda_pcr_true_batched_end_to_end(
                               systems,config.iterations,config.warmups));
            if (matmul_inspector::tridiagonal_benchmark_mode_includes(
                    config.mode,Mode::true_batched_reuse))
                for (const auto& result :
                     matmul_inspector::benchmark_cuda_pcr_true_batched_reuse(
                         systems,config.iterations,config.warmups))
                    csv << matmul_inspector::tridiagonal_benchmark_csv_row(result);
            const bool hybrid_selected=
                matmul_inspector::tridiagonal_benchmark_mode_includes(
                    config.mode,Mode::true_batched_hybrid);
            if (matmul_inspector::tridiagonal_benchmark_mode_includes(
                    config.mode,Mode::true_batched_device_resident) || hybrid_selected)
                for (const auto& result :
                     matmul_inspector::benchmark_cuda_pcr_device_resident(
                         systems,config.iterations,config.warmups))
                    csv << matmul_inspector::tridiagonal_benchmark_csv_row(result);
            if (hybrid_selected)
                csv << matmul_inspector::tridiagonal_benchmark_csv_row(
                    matmul_inspector::benchmark_cuda_pcr_hybrid_device_resident(
                        systems,config.iterations,config.warmups));
#endif
            std::cout << "N=" << n << " B=" << batch << " complete\n";
        }
        std::ofstream methodology(output/"methodology.txt");
        methodology << "selected_mode: "
                    << matmul_inspector::tridiagonal_benchmark_mode_name(config.mode) << '\n'
                    << "CPU: steady_clock around a full batch solve, including output/work allocations.\n"
                    << "batch_size: number of independent systems in one sample; batch_execution identifies serial_host_loop, true_batched_gpu, true_batched_reuse, true_batched_device_resident, or true_batched_hybrid.\n"
                    << "CUDA kernel_only: CUDA events around PCR kernels for the full system count; allocation, coefficient reset copies, and result copies excluded.\n"
                    << "CUDA end_to_end: steady_clock including allocation, H2D, kernels, synchronization, D2H, and cleanup.\n"
                    << "serial_host_loop launches each system separately; true_batched_gpu launches each PCR stage once over batch_size * system_size equations.\n"
                    << "true_batched_reuse owns persistent device buffers: allocation_setup uses steady_clock; h2d, kernel_only, and d2h use CUDA events; reusable_end_to_end uses steady_clock around reset, kernels, and download but excludes workspace allocation and cleanup.\n"
                    << "true_batched_device_resident uploads before sampling and retains output on device through the stop event. d2d_reset separately measures restoring mutable working coefficients from immutable device copies; device_resident measures only all PCR stages and the final solve kernel.\n"
                    << "true_batched_hybrid uses shared-memory stages through offset 256 and the established global-memory kernel at larger offsets; its device_resident row has the same timing boundary as the global baseline.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tridiagonal-benchmark: " << error.what() << '\n';
        return 2;
    }
}
