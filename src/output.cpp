#include "output.hpp"
#include "numeric.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace output {
namespace {
// SHA-256 compression and padding, per FIPS 180-4. No native-endian loads.
class Sha256 {
    std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<unsigned char,64> block{};
    std::uint64_t bytes = 0;
    unsigned used = 0;
    static std::uint32_t rotate(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
    void compress() {
        constexpr std::uint32_t k[] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64];
        for (unsigned i=0;i<16;++i) {
            w[i]=0;
            for (unsigned j=0;j<4;++j) w[i]=(w[i]<<8)|block[4*i+j];
        }
        for (unsigned i=16;i<64;++i) {
            const auto x=w[i-15], y=w[i-2];
            w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for (unsigned i=0;i<64;++i) {
            const auto t1=v+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            v=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
public:
    void append(unsigned char byte) {
        block[used++]=byte; ++bytes;
        if (used==64) { compress(); used=0; }
    }
    std::string finish() {
        const std::uint64_t bits=bytes*8;
        append(0x80);
        while (used!=56) append(0);
        for (int i=7;i>=0;--i) append(static_cast<unsigned char>(bits>>(8*i)));
        std::ostringstream result;
        result.imbue(std::locale::classic());
        for (auto word:h) result << std::hex << std::setfill('0') << std::setw(8) << word;
        return result.str();
    }
};
std::uint32_t bits_at(const Matrix& matrix, std::size_t row, std::size_t col) {
    // Copy storage directly: do not evaluate a signaling NaN in FP arithmetic.
    std::uint32_t bits;
    std::memcpy(&bits, matrix.data()+row*matrix.row_stride()+col, 4);
    return bits;
}
void write_le(std::ostream& stream, std::uint64_t value, unsigned bytes) {
    for (unsigned i=0;i<bytes;++i) stream.put(static_cast<char>((value>>(8*i))&255));
}
std::uint64_t read_le(std::istream& stream, unsigned bytes) {
    std::uint64_t value=0;
    for (unsigned i=0;i<bytes;++i) value |= std::uint64_t(static_cast<unsigned char>(stream.get()))<<(8*i);
    return value;
}
}
std::string fingerprint(const Matrix& matrix) {
    Sha256 hash;
    for (std::size_t row=0;row<matrix.rows() && matrix.cols();++row)
        for (std::size_t col=0;col<matrix.cols();++col) {
            const auto bits=bits_at(matrix,row,col);
            for (unsigned i=0;i<4;++i) hash.append(static_cast<unsigned char>(bits>>(8*i)));
        }
    return hash.finish();
}
void save(const Matrix& matrix, const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    stream.write("MIFP32LE",8);
    write_le(stream,matrix.rows(),8); write_le(stream,matrix.cols(),8);
    for (std::size_t row=0;row<matrix.rows() && matrix.cols();++row)
        for (std::size_t col=0;col<matrix.cols();++col) write_le(stream,bits_at(matrix,row,col),4);
    stream.close();
}
Matrix read(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary);
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    char magic[8]; stream.read(magic,8);
    if (std::memcmp(magic,"MIFP32LE",8)) throw std::invalid_argument("Unsupported output format/dtype/byte order");
    const auto rows=read_le(stream,8), cols=read_le(stream,8);
    const auto max=std::numeric_limits<std::size_t>::max();
    if (rows>max || cols>max || (cols && rows>(max-24)/4/cols)) throw std::invalid_argument("Output dimensions overflow");
    if (std::filesystem::file_size(path)!=24+rows*cols*4) throw std::invalid_argument("Output size does not match dimensions");
    Matrix matrix(static_cast<std::size_t>(rows),static_cast<std::size_t>(cols));
    for (std::size_t row=0;row<matrix.rows() && matrix.cols();++row)
        for (std::size_t col=0;col<matrix.cols();++col) {
            const auto bits=static_cast<std::uint32_t>(read_le(stream,4));
            std::memcpy(matrix.data()+row*matrix.row_stride()+col,&bits,4);
        }
    return matrix;
}
}
