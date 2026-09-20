#include <iostream>
#include <cmath>

#include "inspector.hpp"
#include "matrix.hpp"

int main() {
    Matrix a(2, 3);
    Matrix b(3, 2);

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;

    b(0, 0) = 7.0f;
    b(0, 1) = 8.0f;
    b(1, 0) = 9.0f;
    b(1, 1) = 10.0f;
    b(2, 0) = 11.0f;
    b(2, 1) = 12.0f;

    Matrix c = matmul(a, b);
    c(0, 1) = 64.00001f;

    std::cout << "Matrix C\n";

    for (std::size_t row = 0; row < c.rows(); ++row) {
        for (std::size_t col = 0; col < c.cols(); ++col) {
            std::cout << c(row, col) << ' ';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    Inspector::trace_matmul(a, b, c, 0, 1);
    Inspector::find_first_mismatch(a, b, c);
    std::cout << '\n';
    Inspector::find_first_bitwise_mismatch(a, b, c);

    return 0;
}