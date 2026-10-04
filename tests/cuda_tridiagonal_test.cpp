#include "cuda_tridiagonal.hpp"
#include "cuda_tridiagonal_benchmark.hpp"
#include "cuda_tridiagonal_workspace.hpp"
#include "tridiagonal.hpp"

#include <cmath>
#include <iostream>
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

    const auto cpu =
        matmul_inspector::solve_pcr(
            lower, diag, upper, rhs);

    const auto gpu =
        matmul_inspector::solve_pcr_cuda(
            lower, diag, upper, rhs);

    constexpr double tolerance = 1e-10;

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(gpu[i] - expected[i]) > tolerance) {
            std::cerr
                << "CUDA PCR failed for n=" << n
                << " at index " << i
                << ": expected " << expected[i]
                << ", got " << gpu[i] << '\n';
            return false;
        }

        if (std::abs(gpu[i] - cpu[i]) > tolerance) {
            std::cerr
                << "CPU/CUDA PCR mismatch for n=" << n
                << " at index " << i
                << ": CPU " << cpu[i]
                << ", CUDA " << gpu[i] << '\n';
            return false;
        }
    }

    return true;
}

bool test_true_batched(std::size_t n,std::size_t batch) {
    const auto systems=matmul_inspector::make_benchmark_systems(n,batch);
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }
    const auto gpu=matmul_inspector::solve_pcr_cuda_batched(lower,diag,upper,rhs);
    if (gpu.size()!=batch) return false;
    for (std::size_t system=0; system<batch; ++system) {
        const auto cpu=matmul_inspector::solve_pcr(
            systems[system].lower,systems[system].diag,
            systems[system].upper,systems[system].rhs);
        for (std::size_t equation=0; equation<n; ++equation) {
            if (std::abs(gpu[system][equation]-systems[system].expected[equation])>1e-9 ||
                std::abs(gpu[system][equation]-cpu[equation])>1e-9) {
                std::cerr << "True-batched PCR mismatch for N=" << n
                          << " B=" << batch << " system=" << system
                          << " equation=" << equation << '\n';
                return false;
            }
        }
    }
    return true;
}

bool test_workspace_reuse(std::size_t n,std::size_t batch) {
    auto systems=matmul_inspector::make_benchmark_systems(n,batch);
    matmul_inspector::CudaPcrBatchedWorkspace workspace(n,batch);
    auto run = [&](const std::vector<matmul_inspector::TridiagonalSystem>& input) {
        std::vector<std::vector<double>> lower,diag,upper,rhs;
        for (const auto& system : input) {
            lower.push_back(system.lower); diag.push_back(system.diag);
            upper.push_back(system.upper); rhs.push_back(system.rhs);
        }
        workspace.upload(lower,diag,upper,rhs);
        workspace.execute();
        std::vector<double> flat(n*batch);
        workspace.download(flat);
        const auto one_shot=matmul_inspector::solve_pcr_cuda_batched(lower,diag,upper,rhs);
        for (std::size_t system=0; system<batch; ++system) {
            const auto cpu=matmul_inspector::solve_pcr(
                input[system].lower,input[system].diag,
                input[system].upper,input[system].rhs);
            for (std::size_t equation=0; equation<n; ++equation) {
                const auto value=flat[system*n+equation];
                if (std::abs(value-input[system].expected[equation])>1e-9 ||
                    std::abs(value-cpu[equation])>1e-9 ||
                    std::abs(value-one_shot[system][equation])>1e-9)
                    return false;
            }
        }
        return true;
    };
    if (!run(systems)) return false;

    // Reuse the same allocation with a distinct known solution and RHS.
    for (std::size_t system=0; system<batch; ++system) {
        for (auto& value : systems[system].expected)
            value += 0.25*static_cast<double>(system+1);
        systems[system].rhs=apply_tridiagonal(
            systems[system].lower,systems[system].diag,
            systems[system].upper,systems[system].expected);
    }
    return run(systems);
}

bool test_device_resident(std::size_t n,std::size_t batch) {
    auto systems=matmul_inspector::make_benchmark_systems(n,batch);
    matmul_inspector::CudaPcrBatchedWorkspace workspace(n,batch);
    auto run = [&](const std::vector<matmul_inspector::TridiagonalSystem>& input) {
        std::vector<std::vector<double>> lower,diag,upper,rhs;
        for (const auto& system : input) {
            lower.push_back(system.lower); diag.push_back(system.diag);
            upper.push_back(system.upper); rhs.push_back(system.rhs);
        }
        workspace.upload(lower,diag,upper,rhs);
        workspace.make_device_resident();
        const auto one_shot=matmul_inspector::solve_pcr_cuda_batched(lower,diag,upper,rhs);
        for (int repeat=0; repeat<2; ++repeat) {
            if (repeat) workspace.reset_from_device();
            workspace.execute();
            std::vector<double> flat(n*batch);
            workspace.download(flat);
            for (std::size_t system=0; system<batch; ++system) {
                const auto cpu=matmul_inspector::solve_pcr(
                    input[system].lower,input[system].diag,
                    input[system].upper,input[system].rhs);
                for (std::size_t equation=0; equation<n; ++equation) {
                    const auto value=flat[system*n+equation];
                    if (std::abs(value-input[system].expected[equation])>1e-9 ||
                        std::abs(value-cpu[equation])>1e-9 ||
                        std::abs(value-one_shot[system][equation])>1e-9)
                        return false;
                }
            }
        }
        return true;
    };
    if (!run(systems)) return false;

    // Replace the immutable device snapshot without reallocating the workspace.
    for (std::size_t system=0; system<batch; ++system) {
        for (auto& value : systems[system].expected)
            value -= 0.125*static_cast<double>(system+1);
        systems[system].rhs=apply_tridiagonal(
            systems[system].lower,systems[system].diag,
            systems[system].upper,systems[system].expected);
    }
    return run(systems);
}

bool test_hybrid(std::size_t n,std::size_t batch) {
    const auto systems=matmul_inspector::make_benchmark_systems(n,batch);
    std::vector<std::vector<double>> lower,diag,upper,rhs;
    for (const auto& system : systems) {
        lower.push_back(system.lower); diag.push_back(system.diag);
        upper.push_back(system.upper); rhs.push_back(system.rhs);
    }
    const auto global=matmul_inspector::solve_pcr_cuda_batched(lower,diag,upper,rhs);
    matmul_inspector::CudaPcrBatchedWorkspace workspace(n,batch);
    workspace.upload(lower,diag,upper,rhs);
    workspace.make_device_resident();
    for (int repeat=0; repeat<2; ++repeat) {
        if (repeat) workspace.reset_from_device();
        workspace.execute_hybrid();
        std::vector<double> hybrid(n*batch);
        workspace.download(hybrid);
        for (std::size_t system=0; system<batch; ++system) {
            const auto cpu=matmul_inspector::solve_pcr(
                systems[system].lower,systems[system].diag,
                systems[system].upper,systems[system].rhs);
            for (std::size_t equation=0; equation<n; ++equation) {
                const auto value=hybrid[system*n+equation];
                if (std::abs(value-systems[system].expected[equation])>1e-9 ||
                    std::abs(value-cpu[equation])>1e-9 ||
                    std::abs(value-global[system][equation])>1e-9) {
                    std::cerr << "Hybrid PCR mismatch for N=" << n << " B=" << batch
                              << " system=" << system << " equation=" << equation << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<std::size_t> sizes{
        1, 2, 3, 4,
        7, 8,
        15, 16,
        31, 32,
        63, 64,
        100,
        127, 128,
        255, 256, 257,
        511, 512,
        1000
    };

    for (const auto n : sizes) {
        if (!test_size(n)) {
            return 1;
        }
    }

    // Non-power-of-two sizes, awkward system counts, and flattened work sizes
    // below, at, and above a 256-thread block boundary.
    for (const auto& [n,batch] : std::vector<std::pair<std::size_t,std::size_t>>{
             {1,3},{31,8},{32,8},{33,8},{33,5},{257,3}}) {
        if (!test_true_batched(n,batch)) return 1;
    }
    for (const auto& [n,batch] : std::vector<std::pair<std::size_t,std::size_t>>{
             {31,8},{32,8},{33,5}}) {
        if (!test_workspace_reuse(n,batch)) {
            std::cerr << "Reusable true-batched PCR failed for N=" << n
                      << " B=" << batch << '\n';
            return 1;
        }
        if (!test_device_resident(n,batch)) {
            std::cerr << "Device-resident true-batched PCR failed for N=" << n
                      << " B=" << batch << '\n';
            return 1;
        }
    }
    for (const auto& [n,batch] : std::vector<std::pair<std::size_t,std::size_t>>{
             {127,3},{256,5},{257,3},{512,3},{1024,3},{4096,3}}) {
        if (!test_hybrid(n,batch)) return 1;
    }

    const auto systems=matmul_inspector::make_benchmark_systems(33,3);
    const auto kernel=matmul_inspector::benchmark_cuda_pcr_kernel_only(systems,3,1);
    const auto end_to_end=matmul_inspector::benchmark_cuda_pcr_end_to_end(systems,2,1);
    const auto batched_kernel=
        matmul_inspector::benchmark_cuda_pcr_true_batched_kernel_only(systems,3,1);
    const auto batched_end_to_end=
        matmul_inspector::benchmark_cuda_pcr_true_batched_end_to_end(systems,2,1);
    const auto reuse=
        matmul_inspector::benchmark_cuda_pcr_true_batched_reuse(systems,2,1);
    const auto resident=
        matmul_inspector::benchmark_cuda_pcr_device_resident(systems,2,1);
    const auto hybrid=
        matmul_inspector::benchmark_cuda_pcr_hybrid_device_resident(systems,2,1);
    if (kernel.backend!="cuda" || kernel.algorithm!="pcr" ||
        kernel.timing_scope!="kernel_only" || kernel.system_size!=33 || kernel.batch_size!=3 ||
        kernel.batch_execution!="serial_host_loop" ||
        kernel.timing.samples.size()!=4 || kernel.timing.median_ms<=0 ||
        end_to_end.timing_scope!="end_to_end" || end_to_end.timing.samples.size()!=3 ||
        end_to_end.timing.median_ms<=0 ||
        batched_kernel.batch_execution!="true_batched_gpu" ||
        batched_kernel.timing_scope!="kernel_only" || batched_kernel.system_size!=33 ||
        batched_kernel.batch_size!=3 || batched_kernel.timing.samples.size()!=4 ||
        batched_kernel.timing.median_ms<=0 ||
        batched_end_to_end.batch_execution!="true_batched_gpu" ||
        batched_end_to_end.timing_scope!="end_to_end" ||
        batched_end_to_end.timing.samples.size()!=3 || batched_end_to_end.timing.median_ms<=0 ||
        reuse.size()!=5 || resident.size()!=2 ||
        hybrid.batch_execution!="true_batched_hybrid" ||
        hybrid.timing_scope!="device_resident" || hybrid.system_size!=33 ||
        hybrid.batch_size!=3 || hybrid.timing.samples.size()!=3 ||
        hybrid.timing.median_ms<=0) {
        std::cerr << "CUDA PCR benchmark timing/metadata failed\n";
        return 1;
    }
    const std::vector<std::string> resident_scopes{"d2d_reset","device_resident"};
    for (std::size_t i=0; i<resident.size(); ++i) {
        if (resident[i].batch_execution!="true_batched_device_resident" ||
            resident[i].timing_scope!=resident_scopes[i] || resident[i].system_size!=33 ||
            resident[i].batch_size!=3 || resident[i].timing.samples.size()!=3 ||
            resident[i].timing.median_ms<=0) {
            std::cerr << "Device-resident CUDA PCR timing metadata failed\n";
            return 1;
        }
    }
    const std::vector<std::string> reuse_scopes{
        "allocation_setup","h2d","kernel_only","d2h","reusable_end_to_end"};
    for (std::size_t i=0; i<reuse.size(); ++i) {
        if (reuse[i].batch_execution!="true_batched_reuse" ||
            reuse[i].timing_scope!=reuse_scopes[i] || reuse[i].system_size!=33 ||
            reuse[i].batch_size!=3 || reuse[i].timing.samples.size()!=3 ||
            reuse[i].timing.median_ms<=0) {
            std::cerr << "Reusable CUDA PCR timing decomposition failed\n";
            return 1;
        }
    }

    std::cout << "All CUDA PCR tests passed\n";
    return 0;
}
