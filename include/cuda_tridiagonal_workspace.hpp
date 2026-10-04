#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace matmul_inspector {

// Owns reusable device storage for B tridiagonal systems of a fixed size N.
// Upload, execute, and download are separate so callers can measure or schedule
// those phases without reallocating device memory.
class CudaPcrBatchedWorkspace {
public:
    CudaPcrBatchedWorkspace(std::size_t system_size, std::size_t batch_size);
    ~CudaPcrBatchedWorkspace();

    CudaPcrBatchedWorkspace(const CudaPcrBatchedWorkspace&) = delete;
    CudaPcrBatchedWorkspace& operator=(const CudaPcrBatchedWorkspace&) = delete;
    CudaPcrBatchedWorkspace(CudaPcrBatchedWorkspace&&) noexcept;
    CudaPcrBatchedWorkspace& operator=(CudaPcrBatchedWorkspace&&) noexcept;

    std::size_t system_size() const noexcept;
    std::size_t batch_size() const noexcept;

    void upload(
        const std::vector<std::vector<double>>& lower,
        const std::vector<std::vector<double>>& diag,
        const std::vector<std::vector<double>>& upper,
        const std::vector<std::vector<double>>& rhs);

    // Re-upload the most recently supplied coefficients after PCR modified the
    // device working buffers.
    void reset();

    // Snapshot the current uploaded coefficients into immutable device buffers.
    // The source buffers are allocated lazily and then reused.
    void make_device_resident();

    // Restore mutable PCR working buffers from the resident device snapshot.
    void reset_from_device();

    void execute();

    // The caller supplies B*N storage so benchmarking can exclude host
    // allocation from the device-to-host transfer measurement.
    void download(std::vector<double>& flattened_result) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace matmul_inspector
