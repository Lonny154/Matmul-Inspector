#include "experiment.hpp"
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        auto config = experiment::parse({"crossover", "--sizes", "8,2,4", "--output", "unused", "--iterations", "2", "--warmups", "0"});
        require(config.reference == "cpu" && config.shapes[0].m == 2 && config.shapes[2].m == 8, "sorted crossover sizes");
        for (const auto& args : std::vector<std::vector<std::string>>{
            {"crossover"}, {"crossover", "--output", "unused", "--reference", "naive"},
            {"crossover", "--output", "unused", "--candidate", "cpu"},
            {"crossover", "--output", "unused", "--sizes", "4,4"},
            {"crossover", "--output", "unused", "--m", "2", "--n", "2", "--k", "2"}}) {
            bool rejected = false;
            try { experiment::parse(args); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid crossover config accepted");
        }
        Matrix a(2,3,5), b(3,2,4);
        experiment::fill(a, 42); experiment::fill(b, 123);
        auto cpu = benchmark::cpu_matmul(a,b,3,1);
        require(comparison::compare(cpu.output, matmul(a,b)).divergent_count == 0, "CPU timing output");
        require(cpu.timing.median_ms > 0 && cpu.timing.min_ms > 0 && cpu.timing.stddev_ms >= 0, "CPU timing stats");
        int calls = 0;
        auto measured = benchmark::measure_host([&] { ++calls; return Matrix(1,1); }, 3, 2);
        require(calls == 5 && measured.output.rows() == 1, "host counts");
        bool rejected = false;
        try { benchmark::cpu_matmul(a,b,0,0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid host counts");
        if (argc == 2) {
            // Synthetic timings for serialization/analysis tests, not performance measurements.
            std::vector<experiment::Row> rows;
            for (const auto shape : config.shapes) {
                experiment::Row row;
                row.shape = shape; row.kernel = row.reference = "cpu";
                row.timing_mode = "host_matmul"; row.timed = true;
                row.timing = {3,1,0.5,0}; row.speedup = 1;
                rows.push_back(row);
                row.kernel = "tiled"; row.tile_size = 16;
                row.timing_mode = "kernel_only";
                const double gpu = shape.m == 4 ? 0.5 : 2;
                row.timing = {gpu*2,gpu,gpu/2,0}; row.speedup = 3/(gpu*2);
                rows.push_back(row);
                row.timing_mode = "end_to_end";
                const double end = shape.m == 8 ? 1 : 3;
                row.timing = {end*2,end,end/2,0}; row.speedup = 3/(end*2);
                rows.push_back(row);
            }
            config.output = argv[1];
            experiment::create_run_directory(config.output);
            experiment::write_artifacts(config.output, config, {{"status","complete"}, {"test_fixture","synthetic timings"}}, rows, "test fixture\n");
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
