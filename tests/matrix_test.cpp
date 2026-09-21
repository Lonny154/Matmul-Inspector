#include "inspector.hpp"
#include "matrix.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << '\n';
        ++failures;
    }
}

template <typename Exception, typename Function>
void check_throws(Function function, const char* description) {
    try {
        function();
        check(false, description);
    } catch (const Exception&) {
    } catch (...) {
        check(false, description);
    }
}

// Inspector changes stream formatting; isolate each captured report.
class Capture {
public:
    Capture() : flags_(std::cout.flags()), precision_(std::cout.precision()),
                fill_(std::cout.fill()), original_(std::cout.rdbuf(output_.rdbuf())) {}
    ~Capture() {
        std::cout.rdbuf(original_);
        std::cout.flags(flags_);
        std::cout.precision(precision_);
        std::cout.fill(fill_);
    }
    std::string str() const { return output_.str(); }

private:
    std::ostringstream output_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
    std::streambuf* original_;
};

void check_contains(const std::string& text, const std::string& part, const char* description) {
    check(text.find(part) != std::string::npos, description);
}

void fill_padding(Matrix& matrix, float value) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = matrix.cols(); col < matrix.row_stride(); ++col) {
            matrix.data()[row * matrix.row_stride() + col] = value;
        }
    }
}

void check_padding(const Matrix& matrix, float value) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = matrix.cols(); col < matrix.row_stride(); ++col) {
            check(matrix.data()[row * matrix.row_stride() + col] == value,
                  "padding remains unchanged");
        }
    }
}

void test_storage() {
    Matrix packed(2, 3);
    check(packed.row_stride() == 3, "default stride equals column count");

    for (std::size_t stride : {3u, 5u}) {
        Matrix matrix(2, 3, stride);
        const Matrix& view = matrix;
        check(matrix.rows() == 2 && matrix.cols() == 3, "logical dimensions exclude padding");
        check(matrix.row_stride() == stride, "explicit stride is retained");
        for (std::size_t i = 0; i < 2 * stride; ++i) {
            check(view.data()[i] == 0.0f, "entire allocation is zero initialized");
        }
        fill_padding(matrix, -999.0f);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                matrix(row, col) = static_cast<float>(row * 3 + col + 1);
                check(&matrix(row, col) == matrix.data() + row * stride + col,
                      "mutable indexing has correct physical address");
                check(&view(row, col) == view.data() + row * stride + col,
                      "const indexing has correct physical address");
                check(view(row, col) == static_cast<float>(row * 3 + col + 1),
                      "logical values ignore padding");
            }
        }
        matrix.data()[stride + 2] = 42.0f;
        check(view(1, 2) == 42.0f, "raw storage writes are visible through logical indexing");
        check_padding(matrix, -999.0f);

        std::ostringstream expected;
        expected << "Memory dump\n-----------\n" << std::fixed << std::setprecision(3);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                expected << '[' << row << ',' << col << "]  "
                         << static_cast<const void*>(view.data() + row * stride + col)
                         << "  " << view(row, col) << '\n';
            }
        }
        Capture capture;
        Inspector::dump_memory(matrix);
        check(capture.str() == expected.str(), "dump contains logical values and physical addresses only");
    }
}

void test_matmul_and_inspector() {
    // Exercise each input and output independently, including mixed layouts.
    for (std::size_t a_stride : {3u, 5u}) {
        for (std::size_t b_stride : {2u, 4u}) {
            Matrix a(2, 3, a_stride);
            Matrix b(3, 2, b_stride);
            fill_padding(a, -999.0f);
            fill_padding(b, -999.0f);
            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 3; ++col) {
                    a(row, col) = static_cast<float>(row * 3 + col + 1);
                }
            }
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t col = 0; col < 2; ++col) {
                    b(row, col) = static_cast<float>(row * 2 + col + 7);
                }
            }

            for (std::size_t c_stride : {2u, 6u}) {
                Matrix c = c_stride == 2 ? matmul(a, b) : matmul(a, b, c_stride);
                check(c.rows() == 2 && c.cols() == 2 && c.row_stride() == c_stride,
                      "matmul result dimensions and stride");
                check(c(0, 0) == 58.0f && c(0, 1) == 64.0f
                      && c(1, 0) == 139.0f && c(1, 1) == 154.0f,
                      "matmul computes all logical entries");
                check_padding(c, 0.0f);
                fill_padding(c, -999.0f);
                {
                    Capture capture;
                    Inspector::find_first_mismatch(a, b, c);
                    Inspector::find_first_bitwise_mismatch(a, b, c);
                    check(capture.str() == "No numeric mismatches found\nNo bitwise mismatches found\n",
                          "mismatch detection ignores input and output padding");
                }
                {
                    Capture capture;
                    Inspector::trace_matmul(a, b, c, 1, 1);
                    const auto trace = capture.str();
                    for (std::size_t k = 0; k < 3; ++k) {
                        std::ostringstream a_address, b_address;
                        a_address << "A[1," << k << "] "
                                  << static_cast<const void*>(a.data() + a_stride + k) << " = ";
                        b_address << "B[" << k << ",1] "
                                  << static_cast<const void*>(b.data() + k * b_stride + 1) << " = ";
                        check_contains(trace, a_address.str(), "trace uses A's row stride");
                        check_contains(trace, b_address.str(), "trace uses B's row stride");
                    }
                    std::ostringstream c_address;
                    c_address << "C[1,1] " << static_cast<const void*>(c.data() + c_stride + 1) << " = 154\n";
                    check_contains(trace, c_address.str(), "trace uses C's row stride");
                    check_contains(trace, "recomputed result = 154\n", "trace recomputes correct value");
                    check_contains(trace, "numeric:         MATCH\n", "trace numerical match");
                    check_contains(trace, "bitwise:         MATCH\n", "trace bitwise match");
                }
                c(1, 0) += 1.0f;
                {
                    Capture capture;
                    Inspector::find_first_mismatch(a, b, c);
                    Inspector::find_first_bitwise_mismatch(a, b, c);
                    const auto report = capture.str();
                    check_contains(report, "FIRST NUMERIC MISMATCH\n----------------------\nC[1,0]\n",
                                   "numeric mismatch locates logical coordinate in second row");
                    check_contains(report, "FIRST BITWISE MISMATCH\n----------------------\nC[1,0]\n",
                                   "bitwise mismatch locates logical coordinate in second row");
                    check_contains(report, "actual:     140\nexpected:   139\n", "mismatch values respect stride");
                }
                check_padding(c, -999.0f);
            }
            check_padding(a, -999.0f);
            check_padding(b, -999.0f);
        }
    }
}

void test_validation_and_empty_matrices() {
    check_throws<std::invalid_argument>([] { Matrix matrix(2, 3, 2); }, "reject overlapping rows");
    check_throws<std::invalid_argument>([] { Matrix matrix(2, 3, 0); }, "reject explicit zero stride with columns");
    check_throws<std::length_error>([] {
        Matrix matrix(std::numeric_limits<std::size_t>::max() / 2 + 1, 1, 2);
    }, "reject storage size overflow");
    check_throws<std::invalid_argument>([] {
        matmul(Matrix(2, 3, 5), Matrix(2, 2, 4));
    }, "reject incompatible matmul dimensions");
    check_throws<std::invalid_argument>([] {
        matmul(Matrix(2, 3, 5), Matrix(3, 2, 4), 1);
    }, "reject insufficient output stride");

    Matrix empty_rows(0, 3, 5);
    Matrix empty_cols(2, 0);
    check(empty_rows.rows() == 0 && empty_rows.row_stride() == 5, "empty rows retain stride");
    check(empty_cols.cols() == 0 && empty_cols.row_stride() == 0, "empty columns default to zero stride");
    Matrix zero = matmul(Matrix(2, 0, 3), Matrix(0, 2, 4), 5);
    check(zero(0, 0) == 0.0f && zero(1, 1) == 0.0f, "empty inner dimension gives zero result");
    check_padding(zero, 0.0f);
}

}  // namespace

int main() {
    test_storage();
    test_matmul_and_inspector();
    test_validation_and_empty_matrices();
    return failures == 0 ? 0 : 1;
}
