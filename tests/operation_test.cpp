#include "experiment.hpp"
#include "operation.hpp"
#include "output.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool value) { if (!value) throw std::runtime_error("operation assertion failed"); }
int main() {
    using operation::Kind;
    auto config=experiment::parse({"benchmark","--operation","dot","--size","257"});
    require(config.operation == Kind::dot && config.shapes[0].m == 1 && config.shapes[0].n == 1 && config.shapes[0].k == 257);
    require(config.reference == "cpu" && config.candidate == "cuda-tree");
    for (auto args : std::vector<std::vector<std::string>>{
        {"compare","--operation","unknown"}, {"compare","--size","4"},
        {"compare","--operation","dot","--size","0"},
        {"compare","--operation","dot","--candidate","tiled"},
        {"compare","--operation","dot","--size","4","--sizes","8"},
        {"compare","--operation","dot","--m","1","--n","1","--k","4"},
        {"crossover","--operation","dot"}}) {
        bool rejected=false;
        try { experiment::parse(args); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected);
    }
    require(experiment::parse({"compare"}).operation == Kind::matmul);
    require(operation::flops(Kind::dot,1,1,3) == 6 && operation::bytes(Kind::dot,1,1,3) == 28);
    require(operation::flops(Kind::reduction_sum,1,1,3) == 2 && operation::bytes(Kind::reduction_sum,1,1,3) == 16);
    require(operation::flops(Kind::reduction_sum,1,1,1) == 0);
    require(operation::flops(Kind::matmul,2,3,4) == 48 && operation::bytes(Kind::matmul,2,3,4) == 104);
    require(operation::stages(1)==1 && operation::stages(256)==1 && operation::stages(257)==2 && operation::stages(65537)==3);
    Matrix a(1,3,5),b(3,1,2);
    a(0,0)=1; a(0,1)=-2; a(0,2)=3;
    b(0,0)=4; b(1,0)=5; b(2,0)=6;
    require(operation::cpu(Kind::dot,a,b)(0,0)==12);
    require(operation::cpu(Kind::reduction_sum,a,b)(0,0)==2);
    Matrix x(1,257),y(257,1),z(1,257);
    experiment::fill(x,42); experiment::fill(z,42); experiment::fill(y,123);
    require(output::fingerprint(x)==output::fingerprint(z));
    Matrix ref(1,1), actual(1,1);
    ref(0,0)=0; actual(0,0)=-0.0f;
    auto c=comparison::compare(ref,actual,0,0);
    require(c.divergent_count==1 && c.max_ulp==0 && c.tolerance_pass);
    actual(0,0)=std::numeric_limits<float>::quiet_NaN();
    require(!comparison::compare(ref,actual,1,1).tolerance_pass);
    ref(0,0)=actual(0,0)=std::numeric_limits<float>::infinity();
    require(comparison::compare(ref,actual,0,0).tolerance_pass);
    std::cout << "Operation reference/model/parser tests passed\n";
}
