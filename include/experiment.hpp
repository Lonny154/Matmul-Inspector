#pragma once

#include "benchmark.hpp"
#include "operation.hpp"
#include "reduction.hpp"
#include "comparison.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace experiment {

struct Shape { std::size_t m, n, k; };
struct Config {
    operation::Kind operation = operation::Kind::matmul;
    std::string mode = "compare";
    std::vector<Shape> shapes{{4, 4, 4}};
    std::string reference = "naive", candidate = "tiled";
    std::uint32_t seed = 42, seed_b = 123;
    unsigned reduction_block_size = 256, reference_block_size = 256;
    int repeats = 1;
    int warmups = 3, iterations = 20, trials = 1;
    benchmark::AnalysisOptions analysis;
    float atol = 1e-6f, rtol = 1e-5f;
    std::filesystem::path output;
    bool legacy = false;
    bool save_output = false;
    std::string input = "random";
    std::size_t max_mismatches = 100;
};

Config parse(const std::vector<std::string>& arguments);
std::string usage();
void fill(Matrix& matrix, std::uint32_t seed);
void fill_inputs(Matrix& a, Matrix& b, const Config& config);
std::string generator(const Config& config);
std::string contraction_mode(const std::string& kernel);
std::string accumulation_mode(const std::string& kernel, operation::Kind kind = operation::Kind::matmul, unsigned threads = 256);

using Metadata = std::map<std::string, std::string>;
// Optional metadata uses "unknown"; git/fast_math bool strings serialize as booleans.
Metadata metadata();
std::string command_output(const std::string& command);

struct Row {
    Shape shape;
    std::string kernel, reference;
    unsigned tile_size = 0;
    unsigned reduction_block_size = 256, reference_block_size = 256;
    std::vector<reduction::Observation> observations;
    bool timed = false;
    std::string timing_mode; // host_matmul, kernel_only, end_to_end; empty for untimed
    benchmark::Statistics timing;
    double speedup = 0, median_speedup = 0;
    comparison::Result comparison;
    std::string output_sha256, reference_sha256;
    std::string output_file, reference_output_file;
};

std::map<std::string,std::string> timing_artifacts(const Config& config, const std::vector<Row>& rows);

std::string json_string(const std::string& value);
std::string csv_field(const std::string& value);
std::string metadata_json(const Config& config, const Metadata& values);
// Refuse to overwrite an existing run directory, including an empty one.
void create_run_directory(const std::filesystem::path& path);
void write_artifacts(const std::filesystem::path& path, const Config& config,
                     const Metadata& values, const std::vector<Row>& rows, const std::string& console);
int run_cli(const std::vector<std::string>& arguments);

}  // namespace experiment
