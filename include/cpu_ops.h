#pragma once

#include <cstddef>

// Public interface for all CPU reference implementations.
// Each .cpp includes this header so the compiler verifies
// the definition matches the declaration at build time.

void gemm_cpu(
    const float* A,
    const float* B,
    float*       C,
    std::size_t  M,
    std::size_t  K,
    std::size_t  N);

void softmax_cpu(
    const float* input,
    float*       output,
    std::size_t  rows,
    std::size_t  cols);