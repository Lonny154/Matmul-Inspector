#pragma once

#include "matrix.hpp"

#include <string>

// Probe the current/default CUDA device and runtime. On failure, optionally
// return a diagnostic for demos and test skips; CUDA headers are not required.
bool cuda_available(std::string* reason = nullptr);

// Naive FP32 matmul on the current/default device. Input padding is copied;
// output padding is zero. Empty products return zeros without using CUDA.
// Invalid dimensions/strides throw std::invalid_argument; CUDA errors throw
// std::runtime_error. The two-argument overload returns tightly packed rows.
Matrix cuda_matmul(const Matrix& a, const Matrix& b);
Matrix cuda_matmul(const Matrix& a, const Matrix& b, std::size_t row_stride);
