#include "output.hpp"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void put(Matrix& matrix, std::size_t col, std::uint32_t bits) { std::memcpy(matrix.data()+col,&bits,4); }
}
int main(int argc,char** argv) {
    try {
        Matrix a(2,3), b(2,3,5);
        for (std::size_t i=0;i<2;++i) for (std::size_t j=0;j<3;++j) a(i,j)=b(i,j)=static_cast<float>(i*3+j);
        b.data()[3]=b.data()[4]=b.data()[8]=b.data()[9]=99;
        check(output::fingerprint(a)==output::fingerprint(b),"padding must not affect hash");
        check(output::fingerprint(Matrix(0,0))=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA-256 empty vector");
        Matrix positive(1,1), negative(1,1);
        put(negative,0,0x80000000);
        check(output::fingerprint(positive)!=output::fingerprint(negative),"signed zeros hash differently");
        put(positive,0,0x7fc00001); put(negative,0,0x7fc00002);
        check(output::fingerprint(positive)!=output::fingerprint(negative),"NaN payload hashes differ");
        if (argc==2) {
            std::filesystem::path root(argv[1]); std::filesystem::create_directories(root);
            for (unsigned length : {0u,1u,2u,14u,15u,16u,17u,31u,32u,33u,70u}) {
                Matrix matrix(1,length,length+3);
                for (unsigned i=0;i<length;++i) put(matrix,i,0x9e3779b9u*i); // arbitrary exact bits, including NaNs
                const auto path=root/(std::to_string(length)+".bin");
                output::save(matrix,path);
                auto copy=output::read(path);
                check(copy.rows()==1 && copy.cols()==length,"binary shape roundtrip");
                check(output::fingerprint(copy)==output::fingerprint(matrix),"binary exact bit roundtrip");
                std::cout << path.filename().string() << ',' << output::fingerprint(matrix) << '\n';
            }
            Matrix special(1,5,8);
            const std::uint32_t bits[]={0,0x80000000,0x7fa12345,0x7fc12345,0xff800000};
            for (unsigned i=0;i<5;++i) put(special,i,bits[i]);
            output::save(special,root/"special.bin");
            auto copy=output::read(root/"special.bin");
            check(std::memcmp(copy.data(),special.data(),20)==0,"signaling/quiet NaN payload and signed zero roundtrip");
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
