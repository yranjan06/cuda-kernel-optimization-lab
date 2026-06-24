#!/bin/bash
# stamp_env.sh
# Prepend this to every benchmark CSV/markdown output. Outputs one CSV line:
#   gpu_name,driver_version,memory_total,cuda_version,timestamp_utc
#
# Usage:
#   bash scripts/stamp_env.sh >> benchmarks/results/gemm_2048.csv

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9.]+")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ -z "$GPU_INFO" ]; then
  echo "ERROR: nvidia-smi returned nothing - is a GPU attached to this runtime?" >&2
  exit 1
fi

echo "${GPU_INFO},${CUDA_VERSION},${TIMESTAMP}"

# Reminder: before comparing two benchmark runs (e.g. naive vs tiled GEMM),
# assert the gpu_name field matches across both. Colab can hand you a
# different physical card between sessions even on the free tier - a
# speedup number computed across two different GPUs is meaningless.
