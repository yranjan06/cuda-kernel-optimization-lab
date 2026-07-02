#include "cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// -----------------------------------------------------------------------------
// CPU Softmax — reference implementation
// Numerically stable row-wise softmax over a 2D tensor (rows x cols)
//
// Three explicit passes — correctness and readability over pass-count efficiency.
// This is the ground truth; CUDA versions are validated against it.
//
// Pass 1: row max  →  prevents expf overflow (overflows at ~88.7 for float32)
// Pass 2: exp(x - max) + accumulate row sum  →  double sum avoids float drift
// Pass 3: normalize each element by row sum
//
// Initial row_max = -infinity, not 0.0f, so all-negative rows are handled correctly.
// -----------------------------------------------------------------------------

void softmax_cpu(
    const float* input,
    float*       output,
    std::size_t  rows,
    std::size_t  cols)
{
    for (std::size_t row = 0; row < rows; ++row) {

        // ---------------------------------------------------------------------
        // Pass 1: Find row maximum
        // -infinity as initial value: first real element always wins,
        // regardless of sign. 0.0f would be wrong for all-negative rows.
        // ---------------------------------------------------------------------

        float row_max = -std::numeric_limits<float>::infinity();

        for (std::size_t col = 0; col < cols; ++col) {
            row_max = std::max(row_max, input[row * cols + col]);
        }

        // ---------------------------------------------------------------------
        // Pass 2: Compute exp(x - max) and accumulate row sum
        // double accumulator: float sum of 1024 values drifts enough to fail
        // 1e-5 row-sum checks even when the softmax output is actually correct
        // ---------------------------------------------------------------------

        double row_sum = 0.0;

        for (std::size_t col = 0; col < cols; ++col) {
            float exp_val           = std::exp(input[row * cols + col] - row_max);
            output[row * cols + col] = exp_val;
            row_sum                 += static_cast<double>(exp_val);
        }

        // ---------------------------------------------------------------------
        // Pass 3: Normalize
        // ---------------------------------------------------------------------

        for (std::size_t col = 0; col < cols; ++col) {
            output[row * cols + col] /= static_cast<float>(row_sum);
        }
    }
}