#include "benchmark_shapes.h"
#include "benchmark_utils.h"
#include "cpu_ops.h"
#include "tensor_utils.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string current_timestamp_utc()
{
    auto now      = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &t);
#else
    utc = *std::gmtime(&t);
#endif

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &utc);
    return buf;
}

} // namespace

int main()
{
    constexpr int WARMUP_ITERS = 2;
    constexpr int TIMED_ITERS  = 10;

    // -------------------------------------------------------------------------
    // Benchmark session context
    // -------------------------------------------------------------------------

    benchmark_utils::BenchmarkContext ctx;
    ctx.timestamp_utc  = current_timestamp_utc();
    ctx.device_type    = "cpu";
    ctx.gpu_name       = "NA";
    ctx.driver_version = "NA";
    ctx.cuda_version   = "NA";
    ctx.seed           = 42;

    // -------------------------------------------------------------------------
    // Output file
    // -------------------------------------------------------------------------

    std::filesystem::create_directories("benchmarks/results");

    std::string path =
        "benchmarks/results/softmax_cpu_" + ctx.timestamp_utc + ".csv";

    std::ofstream csv(path);
    if (!csv.is_open()) {
        std::cerr << "Error: could not open output file: " << path << '\n';
        return 1;
    }

    benchmark_utils::write_csv_header(csv);

    // -------------------------------------------------------------------------
    // Benchmark loop
    // -------------------------------------------------------------------------

    for (const auto& shape : benchmark_shapes::SOFTMAX) {

        const std::size_t rows = shape.rows;
        const std::size_t cols = shape.cols;

        auto input = tensor_utils::generate_matrix(rows, cols);
        std::vector<float> output(rows * cols, 0.0f);

        auto stats = benchmark_utils::benchmark_cpu(
            [&]() { softmax_cpu(input.data(), output.data(), rows, cols); },
            WARMUP_ITERS,
            TIMED_ITERS);

        // Softmax has two correctness invariants GEMM does not:
        // 1. tensor_is_all_finite — no inf/NaN (catches overflow if max-sub fails)
        // 2. row_sums_near_one   — each row is a valid probability distribution
        bool correct =
            tensor_utils::tensor_is_all_finite(output) &&
            tensor_utils::row_sums_near_one(output, rows, cols);

        benchmark_utils::BenchmarkRecord rec;
        rec.op           = "softmax";
        rec.version      = "cpu_reference";
        rec.shape        = std::to_string(rows) + "x" + std::to_string(cols);
        rec.warmup_iters = WARMUP_ITERS;
        rec.timed_iters  = TIMED_ITERS;
        rec.atol         = 1e-5f;
        rec.rtol         = 1e-5f;
        rec.correct      = correct;
        rec.stats        = stats;

        benchmark_utils::write_csv_row(csv, ctx, rec);

        std::cout
            << "[SOFTMAX] " << rec.shape
            << " | median = " << std::fixed << std::setprecision(3)
            << stats.median_ms << " ms"
            << " | correct = " << std::boolalpha << correct
            << '\n';
    }

    std::cout << "\nResults written to: " << path << '\n';
    return 0;
}