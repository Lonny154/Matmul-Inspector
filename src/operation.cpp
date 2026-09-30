#include "operation.hpp"
#include <stdexcept>
namespace operation {
Kind parse(const std::string& value) {
    if (value == "matmul") return Kind::matmul;
    if (value == "dot") return Kind::dot;
    if (value == "reduction_sum") return Kind::reduction_sum;
    throw std::invalid_argument("Unknown operation: " + value);
}
std::string name(Kind kind) {
    switch (kind) {
    case Kind::matmul: return "matmul";
    case Kind::dot: return "dot";
    case Kind::reduction_sum: return "reduction_sum";
    }
    throw std::invalid_argument("Unknown operation");
}
double flops(Kind kind, std::size_t m, std::size_t n, std::size_t k) {
    if (kind == Kind::matmul) return 2.0*m*n*k;
    if (kind == Kind::dot) return 2.0*k;
    return k ? static_cast<double>(k-1) : 0;
}
double bytes(Kind kind, std::size_t m, std::size_t n, std::size_t k) {
    if (kind == Kind::matmul) return 4.0*(double(m)*k+double(k)*n+double(m)*n);
    return 4.0*((kind == Kind::dot ? 2.0 : 1.0)*k+1);
}
unsigned stages(std::size_t length) {
    unsigned count = 0;
    do { length = length/block_size + (length%block_size != 0); ++count; } while (length > 1);
    return count;
}
void validate_vectors(Kind kind, const Matrix& a, const Matrix& b) {
    if (kind == Kind::matmul || a.rows() != 1 || a.cols() == 0
        || (kind == Kind::dot && (b.rows() != a.cols() || b.cols() != 1)))
        throw std::invalid_argument("Expected a nonempty row vector A and, for dot, matching column vector B");
}
Matrix cpu(Kind kind, const Matrix& a, const Matrix& b) {
    if (kind == Kind::matmul) return matmul(a,b);
    validate_vectors(kind,a,b);
    float sum = 0;
    for (std::size_t i=0; i<a.cols(); ++i) {
        if (kind == Kind::dot) sum += a(0,i)*b(i,0);
        else sum += a(0,i);
    }
    Matrix result(1,1);
    result(0,0)=sum;
    return result;
}
}
