#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace benchmark_utils {

// -----------------------------------------------------------------------------
// Benchmark statistics
// -----------------------------------------------------------------------------

struct BenchmarkStats {
    double median_ms{};
    double mean_ms{};
    double std_ms{};
    double min_ms{};
    double max_ms{};
};

// -----------------------------------------------------------------------------
// Benchmark environment — set once per session
// -----------------------------------------------------------------------------

struct BenchmarkContext {
    std::string timestamp_utc;
    std::string device_type;
    std::string gpu_name;
    std::string driver_version;
    std::string cuda_version;
    int seed{42};
};

// -----------------------------------------------------------------------------
// Per-run benchmark record
// -----------------------------------------------------------------------------

struct BenchmarkRecord {
    std::string op;
    std::string version;
    std::string shape;

    int warmup_iters{};
    int timed_iters{};

    float atol{};
    float rtol{};

    bool correct{};

    BenchmarkStats stats;
};

// -----------------------------------------------------------------------------
// Timing helper
// Uses steady_clock (monotonic) not high_resolution_clock (not guaranteed monotonic)
// Median over mean: robust to individual outlier runs from Colab shared-tenant noise
// Std dev: population (/ N), not sample (/ N-1) — acceptable for benchmark use
// -----------------------------------------------------------------------------

inline BenchmarkStats benchmark_cpu(
    const std::function<void()>& fn,
    int warmup = 2,
    int iters  = 10)
{
    using clock = std::chrono::steady_clock;

    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        auto start = clock::now();
        fn();
        auto end = clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());

    BenchmarkStats stats;

    stats.min_ms = samples.front();
    stats.max_ms = samples.back();

    stats.mean_ms =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

    // True median: average two middle values for even counts
    if (samples.size() % 2 == 0) {
        std::size_t mid = samples.size() / 2;
        stats.median_ms = (samples[mid - 1] + samples[mid]) / 2.0;
    } else {
        stats.median_ms = samples[samples.size() / 2];
    }

    double variance = 0.0;
    for (double x : samples) {
        variance += (x - stats.mean_ms) * (x - stats.mean_ms);
    }
    stats.std_ms = std::sqrt(variance / samples.size());

    return stats;
}

// -----------------------------------------------------------------------------
// CSV output
// Schema is fixed here — one place to change if columns are ever added
// -----------------------------------------------------------------------------

inline void write_csv_header(std::ofstream& out)
{
    out << "timestamp_utc,operator,version,shape,device_type,"
           "gpu_name,driver_version,cuda_version,seed,"
           "warmup_iters,timed_iters,median_ms,mean_ms,std_ms,"
           "min_ms,max_ms,atol,rtol,correct\n";
}

inline void write_csv_row(
    std::ofstream& out,
    const BenchmarkContext& ctx,
    const BenchmarkRecord& record)
{
    out
        << ctx.timestamp_utc        << ','
        << record.op                << ','
        << record.version           << ','
        << record.shape             << ','
        << ctx.device_type          << ','
        << ctx.gpu_name             << ','
        << ctx.driver_version       << ','
        << ctx.cuda_version         << ','
        << ctx.seed                 << ','
        << record.warmup_iters      << ','
        << record.timed_iters       << ','
        << record.stats.median_ms   << ','
        << record.stats.mean_ms     << ','
        << record.stats.std_ms      << ','
        << record.stats.min_ms      << ','
        << record.stats.max_ms      << ','
        << record.atol              << ','
        << record.rtol              << ','
        << record.correct           << '\n';
}

} // namespace benchmark_utils