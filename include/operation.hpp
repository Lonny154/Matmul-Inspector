#pragma once
#include "matrix.hpp"
#include <string>

// Small dispatch/model layer. Vector shapes are (M=1,N=1,K=length);
// the 1x1 Matrix is an output transport, not a matrix multiplication.
namespace operation {
enum class Kind { matmul, dot, reduction_sum };
Kind parse(const std::string& name);
std::string name(Kind kind);
double flops(Kind kind, std::size_t m, std::size_t n, std::size_t k);
double bytes(Kind kind, std::size_t m, std::size_t n, std::size_t k);
inline constexpr unsigned block_size = 256;
unsigned stages(std::size_t length);
void validate_vectors(Kind kind, const Matrix& a, const Matrix& b);
Matrix cpu(Kind kind, const Matrix& a, const Matrix& b);
}
