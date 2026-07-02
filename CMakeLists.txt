cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimization_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# T4 on Colab = compute capability 7.5.
# If a Colab session ever allocates a different GPU (check with env_check.sh
# / nvidia-smi), update this and note it in the Environment Validation Log -
# do not silently keep building for sm_75 against a different card.
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Targets are added phase by phase. Phase 1 is now wired up with the CPU
# baseline executables; later CUDA kernels will be added as each phase lands.
# ---------------------------------------------------------------------------

add_executable(gemm_cpu_baseline
	benchmarks/benchmark_gemm.cpp
	cpu/gemm_cpu.cpp)

add_executable(softmax_cpu_baseline
	benchmarks/benchmark_softmax.cpp
	cpu/softmax_cpu.cpp)

# Phase 2 example (uncomment once kernels/gemm/gemm_naive.cu exists):
# add_executable(gemm_naive kernels/gemm/gemm_naive.cu kernels/gemm/gemm_launcher.cpp)

enable_testing()