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
    // Single source of truth for iteration counts.
    // Both the benchmark execution and the CSV record use these constants
    // so they can never silently drift apart.
    constexpr int WARMUP_ITERS = 2;
    constexpr int TIMED_ITERS  = 10;

    // -------------------------------------------------------------------------
    // Benchmark session context — set once, reused for every CSV row
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
    // Timestamped filename: each run produces a new file, no overwriting.
    // -------------------------------------------------------------------------

    std::filesystem::create_directories("benchmarks/results");

    std::string path =
        "benchmarks/results/gemm_cpu_" + ctx.timestamp_utc + ".csv";

    std::ofstream csv(path);
    if (!csv.is_open()) {
        std::cerr << "Error: could not open output file: " << path << '\n';
        return 1;
    }

    benchmark_utils::write_csv_header(csv);

    // -------------------------------------------------------------------------
    // Benchmark loop — one row per shape
    // -------------------------------------------------------------------------

    for (std::size_t size : benchmark_shapes::GEMM) {

        const std::size_t M = size;
        const std::size_t K = size;
        const std::size_t N = size;

        auto A = tensor_utils::generate_matrix(M, K);
        auto B = tensor_utils::generate_matrix(K, N);
        std::vector<float> C(M * N, 0.0f);

        // Lambda allocates nothing — only compute is timed
        auto stats = benchmark_utils::benchmark_cpu(
            [&]() { gemm_cpu(A.data(), B.data(), C.data(), M, K, N); },
            WARMUP_ITERS,
            TIMED_ITERS);

        // CPU is the reference — no independent ground truth yet.
        // Correctness = output is finite (no inf/NaN from overflow/underflow).
        // Phase 2 adds tensors_close(cpu_out, cuda_out) as the real check.
        bool correct = tensor_utils::tensor_is_all_finite(C);

        benchmark_utils::BenchmarkRecord rec;
        rec.op           = "gemm";
        rec.version      = "cpu_reference";
        rec.shape        = std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
        rec.warmup_iters = WARMUP_ITERS;
        rec.timed_iters  = TIMED_ITERS;
        rec.atol         = 1e-4f;
        rec.rtol         = 1e-4f;
        rec.correct      = correct;
        rec.stats        = stats;

        benchmark_utils::write_csv_row(csv, ctx, rec);

        std::cout
            << "[GEMM] " << rec.shape
            << " | median = " << std::fixed << std::setprecision(3)
            << stats.median_ms << " ms"
            << " | correct = " << std::boolalpha << correct
            << '\n';
    }

    std::cout << "\nResults written to: " << path << '\n';
    return 0;
}