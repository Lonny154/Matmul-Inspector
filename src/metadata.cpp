#include "experiment.hpp"
#include "build_options.hpp"
#include "git_provenance.hpp"
#include "fp_verification.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

namespace experiment {
namespace {
std::string shell_quote(const std::string& text) {
    std::string result = "'";
    for (char ch : text) result += ch == '\'' ? "'\\''" : std::string(1, ch);
    return result + "'";
}
std::string known(std::string value) {
    if (value.find_first_not_of(" \t\r\n") == std::string::npos || value.find("NOTFOUND") != std::string::npos) return "unknown";
    return value;
}
}

std::string command_output(const std::string& command) {
#if defined(__unix__) || defined(__APPLE__)
    FILE* pipe = popen((command + " 2>/dev/null").c_str(), "r");
    if (!pipe) return "unknown";
    std::array<char, 512> buffer{};
    std::string result;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) result += buffer.data();
    if (pclose(pipe) != 0) return "unknown";
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
#else
    (void)command;
    return "unknown";
#endif
}

Metadata metadata() {
    Metadata values{
        {"git_commit", build_info::git_commit}, {"build_git_dirty", build_info::git_dirty},
        {"build_type", known(build_info::build_type)},
        {"cxx_compiler", known(build_info::cxx_compiler)}, {"cxx_compiler_path", known(build_info::cxx_compiler_path)},
        {"cuda_compiler", known(build_info::cuda_compiler)}, {"cuda_compiler_path", known(build_info::cuda_compiler_path)},
        {"cxx_flags", build_info::cxx_flags}, {"cuda_flags", build_info::cuda_flags},
        {"cuda_architectures", known(build_info::cuda_architectures)},
        {"cmake_version", build_info::cmake_version}, {"warnings_enabled", build_info::warnings},
        {"cxx_standard", "17"}, {"os", build_info::system}, {"cpu_model", "unknown"},
        {"gpu_name", "unknown"}, {"gpu_compute_capability", "unknown"},
        {"cuda_runtime_version", "unknown"}, {"cuda_driver_api_version", "unknown"},
        {"nvidia_driver_version", "unknown"},
        {"timing_methodology", "CUDA events; default stream; individual launches; reference then candidate; allocations, transfers, initialization, warmups and host comparison excluded; population stddev"},
        {"fma_policy", "per-kernel config contraction modes; explicit RN intrinsics for controlled variants; recorded flags govern ordinary kernels"},
        {"status", "running"}};
    values["cuda_support"] = std::string(build_info::cuda_built) == "ON" ? "true" : "false";
    values["fp_verified"] = fp_verification::verified ? "true" : "false";
    values["fp_verification_method"] = fp_verification::method;
    values["fp_verification_library_sha256"] = fp_verification::binary_sha256;
    values["fp_verification_details"] = fp_verification::details;
    if (values["cuda_support"] == "false") {
        for (const auto* field : {"cuda_compiler", "cuda_compiler_path", "cuda_flags", "cuda_architectures"}) values[field] = "unknown";
    }
    const auto now = std::time(nullptr);
    if (const auto* utc = std::gmtime(&now)) {
        std::ostringstream timestamp;
        timestamp << std::put_time(utc, "%Y-%m-%dT%H:%M:%SZ");
        values["timestamp_utc"] = timestamp.str();
    } else values["timestamp_utc"] = "unknown";
    const auto git = "git -C " + shell_quote(build_info::source_dir);
    values["runtime_git_commit"] = known(command_output(git + " rev-parse HEAD"));
    const auto state = command_output(git + " status --porcelain --untracked-files=normal");
    values["runtime_git_dirty"] = state == "unknown" ? "unknown" : (state.empty() ? "false" : "true");
    values["git_dirty"] = values["build_git_dirty"] == "true" || values["runtime_git_dirty"] == "true" ? "true"
        : (values["build_git_dirty"] == "false" && values["runtime_git_dirty"] == "false" ? "false" : "unknown");
#ifdef __FAST_MATH__
    values["cxx_fast_math"] = "true";
#elif defined(__GNUC__) || defined(__clang__)
    values["cxx_fast_math"] = "false";
#else
    values["cxx_fast_math"] = "unknown";
#endif
    values["cuda_fast_math"] = values["cuda_compiler"].rfind("NVIDIA ", 0) != 0 ? "unknown"
        : (values["cuda_flags"].find("use_fast_math") != std::string::npos ? "true" : "false");
    values["fast_math"] = values["cxx_fast_math"] == "true" || values["cuda_fast_math"] == "true" ? "true"
        : (values["cxx_fast_math"] == "false" && (values["cuda_support"] == "false" || values["cuda_fast_math"] == "false") ? "false" : "unknown");
#if defined(__unix__) || defined(__APPLE__)
    struct utsname system{};
    if (uname(&system) == 0) values["os"] = std::string(system.sysname) + " " + system.release + " " + system.machine;
#endif
    std::ifstream cpu("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu, line)) {
        if (line.rfind("model name", 0) == 0 && line.find(':') != std::string::npos) {
            auto model = line.substr(line.find(':') + 1);
            const auto start = model.find_first_not_of(" \t");
            values["cpu_model"] = start == std::string::npos ? "unknown" : model.substr(start);
            break;
        }
    }
    return values;
}
}  // namespace experiment
