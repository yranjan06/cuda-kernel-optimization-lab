#include "cpu_ops.h"

#include <cstddef>

// -----------------------------------------------------------------------------
// CPU GEMM — reference implementation
// C[M x N] = A[M x K] * B[K x N]
// Row-major flat storage: A[row*K + col], B[row*N + col], C[row*N + col]
//
// Raw pointers match the CUDA launcher signature so the benchmark driver
// calls both CPU and CUDA versions identically via vec.data().
// Allocation and size validation stay in the benchmark driver, not here.
// -----------------------------------------------------------------------------

void gemm_cpu(
    const float* A,
    const float* B,
    float*       C,
    std::size_t  M,
    std::size_t  K,
    std::size_t  N)
{
    for (std::size_t row = 0; row < M; ++row) {
        for (std::size_t col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (std::size_t k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}