#pragma once

#include <array>
#include <cstddef>

namespace benchmark_shapes {

// -------------------------
// GEMM benchmark shapes
// Square matrices: M = K = N
// -------------------------
inline constexpr std::array<std::size_t, 3> GEMM = {
    256,
    512,
    1024
};

// -------------------------
// Softmax benchmark shapes
// (rows, cols)
// -------------------------
struct SoftmaxShape {
    std::size_t rows;
    std::size_t cols;
};

inline constexpr std::array<SoftmaxShape, 2> SOFTMAX = {{
    {1024, 512},
    {1024, 1024}
}};

} // namespace benchmark_shapes