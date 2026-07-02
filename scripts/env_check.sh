#!/bin/bash
# env_check.sh
# Run this FIRST in every new Colab session, BEFORE writing or building any kernel.
# It tells you which profiling tier is available this session so you don't plan
# around tools you can't actually use.
#
# Usage (in a Colab cell):
#   !bash scripts/env_check.sh

set -uo pipefail

echo "================================================================"
echo " GPU"
echo "================================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader 2>/dev/null \
  || echo "nvidia-smi failed - no GPU attached to this runtime?"

echo
echo "================================================================"
echo " CUDA Toolkit"
echo "================================================================"
nvcc --version 2>/dev/null || echo "nvcc not found on PATH"

echo
echo "================================================================"
echo " Nsight Systems (nsys) - usually allowed without elevated perms"
echo "================================================================"
NSYS_BIN=$(command -v nsys 2>/dev/null)
if [ -z "$NSYS_BIN" ]; then
  echo "nsys not on PATH - searching under /opt/nvidia (version-numbered paths"
  echo "change between Colab images, so this is a search, not a hardcoded path)..."
  NSYS_BIN=$(find /opt/nvidia -iname "nsys" -type f 2>/dev/null | head -1)
fi
if [ -n "$NSYS_BIN" ]; then
  echo "Found: $NSYS_BIN"
  "$NSYS_BIN" --version
  echo
  echo "Not on PATH by default. To use the bare 'nsys' command this session:"
  echo "  export PATH=\"\$PATH:$(dirname "$NSYS_BIN")\""
else
  echo "nsys not found anywhere under /opt/nvidia on this image - Tier 2 unavailable this session."
fi

echo
echo "================================================================"
echo " Nsight Compute (ncu) - requires unlocked HW perf counters"
echo "================================================================"
ncu --version 2>/dev/null
if [ $? -ne 0 ]; then
  echo "ncu binary not found - Tier 3 profiling unavailable on this image."
else
  echo
  echo "Testing counter access with a trivial no-op kernel..."
  TMPDIR=$(mktemp -d)
  cat > "$TMPDIR/ncu_probe.cu" << 'EOF'
__global__ void noop_kernel() {}
int main() {
    noop_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
  nvcc "$TMPDIR/ncu_probe.cu" -o "$TMPDIR/ncu_probe" 2>/tmp/ncu_probe_build.log
  if [ $? -ne 0 ]; then
    echo "Probe kernel failed to build - see /tmp/ncu_probe_build.log"
  else
    ncu --target-processes all "$TMPDIR/ncu_probe" 2>&1 | tee /tmp/ncu_probe_run.log
    if grep -q "ERR_NVGPUCTRPERM" /tmp/ncu_probe_run.log; then
      echo
      echo ">>> RESULT: ncu is BLOCKED on this session (ERR_NVGPUCTRPERM)."
      echo ">>> Lock profiling scope to Tier 1 (wall clock + ptxas -v) and Tier 2 (nsys)."
      echo ">>> Record this result in the Environment Validation Log in README.md."
    else
      echo
      echo ">>> RESULT: ncu appears to have counter access on this session."
      echo ">>> Tier 3 profiling (occupancy, DRAM throughput, warp stalls) is usable."
      echo ">>> Still confirm this every new session - Colab GPU allocation is not stable."
    fi
  fi
  rm -rf "$TMPDIR"
fi

echo
echo "================================================================"
echo " Compile-time register/spill check (ALWAYS available, no perms needed)"
echo "================================================================"
echo "Use this on every kernel regardless of ncu status:"
echo '  nvcc -Xptxas -v -arch=sm_75 your_kernel.cu -o your_kernel'
echo "(-arch=sm_75 is the T4 compute capability. Confirm against nvidia-smi above"
echo " if you ever run on a different Colab-allocated GPU.)"