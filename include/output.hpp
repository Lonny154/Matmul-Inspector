#pragma once
#include "matrix.hpp"
#include <filesystem>
#include <string>

namespace output {
// SHA-256 of row-major logical FP32 bit patterns, each encoded little-endian.
// Shape is validated separately; padding and binary-file headers are not hashed.
std::string fingerprint(const Matrix& matrix);
// MIFP32LE + uint64_le rows + uint64_le cols + logical payload; exact bits.
void save(const Matrix& matrix, const std::filesystem::path& path);
Matrix read(const std::filesystem::path& path);
}
