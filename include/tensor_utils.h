#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace tensor_utils {

// -----------------------------------------------------------------------------
// Matrix generation
// -----------------------------------------------------------------------------

inline std::vector<float> generate_matrix(
    std::size_t rows,
    std::size_t cols)
{
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> matrix(rows * cols);
    for (float& value : matrix) {
        value = dist(rng);
    }
    return matrix;
}

// Stress-test variant: range covers overflow regime for softmax (expf overflows ~88.7)
inline std::vector<float> generate_matrix_stress(
    std::size_t rows,
    std::size_t cols)
{
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-120.0f, 120.0f);
    std::vector<float> matrix(rows * cols);
    for (float& value : matrix) {
        value = dist(rng);
    }
    return matrix;
}

// -----------------------------------------------------------------------------
// Generic tensor comparison
// Rule: abs(a[i] - b[i]) <= atol + rtol * abs(b[i])
// -----------------------------------------------------------------------------

inline bool tensors_close(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float atol = 1e-4f,
    float rtol = 1e-4f)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        float diff  = std::fabs(a[i] - b[i]);
        float limit = atol + rtol * std::fabs(b[i]);
        if (diff > limit) {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Softmax validation helpers
// -----------------------------------------------------------------------------

// Flat scan — checks every element in the tensor, no row structure
inline bool tensor_is_all_finite(
    const std::vector<float>& tensor)
{
    for (float value : tensor) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

// Accumulate in double to avoid float rounding error invalidating correct outputs
// at large col counts (e.g. cols=1024 can drift enough to fail 1e-5 in float)
inline bool row_sums_near_one(
    const std::vector<float>& tensor,
    std::size_t rows,
    std::size_t cols,
    float tolerance = 1e-5f)
{
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += static_cast<double>(tensor[row * cols + col]);
        }
        if (std::fabs(sum - 1.0) > static_cast<double>(tolerance)) {
            return false;
        }
    }
    return true;
}

} // namespace tensor_utils