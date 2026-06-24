# CUDA Kernel Optimization Lab

Porting GEMM and Softmax from CPU to CUDA — naive → tiled/reduction-based →
optimized — with correctness tests, environment-stamped benchmarks, and
tiered profiling, built as a learning vehicle for GPU performance engineering.

---

## 0. Status

This project is being **learned and built at the same time**, not applied
after the fact. That changes how it's built:

- Phases run **sequentially**, not in parallel — CPU baseline → naive GEMM →
  tiled GEMM → optimized GEMM → naive Softmax → optimized Softmax → benchmark
  harness polish → report.
- Every kernel goes through a forced loop before it's considered "learned":
  write `reasoning.md` (thread/block mapping, what each thread computes, what
  goes in shared memory, why the indexing is correct) **before** writing
  `attempt.cu`, and before any reference or correction is consulted.
- No resume claim or README "key learning" is written until a real benchmark
  number backs it. See [§12](#12-resume-claims-policy).

Current phase: **environment validation, pre-Phase-1.**

---

## 1. Why This Project Exists

The question a GPU/ML-infra internship recruiter actually has:

> Can this candidate write GPU code, optimize it, and reason about
> performance with evidence — not just call high-level framework ops?

This project is structured to answer that directly: CPU reference →
naive CUDA → optimized CUDA, for two operators, each backed by correctness
tests, latency benchmarks, and whatever profiling tier the environment
actually allows (see [§7](#7-profiling-tiers)).

---

## 2. Scope & Non-Goals

| In scope | Explicitly out of scope (for this version) |
|---|---|
| GEMM: CPU → naive → tiled → optimized | A 3rd/4th operator (e.g. LayerNorm, attention) |
| Softmax: CPU → naive → reduction-optimized | Tensor Core / WMMA kernels |
| Shared-memory tiling, bank-conflict awareness | Multi-GPU / distributed kernels |
| Block-level reductions (max, sum) | A general CUDA kernel library |
| Correctness tests vs CPU reference | Production-grade error handling |
| Environment-stamped, variance-aware benchmarking | Assuming dedicated/local hardware |
| Tiered profiling (wall-clock → ptxas -v → nsys → ncu if available) | Assuming Nsight Compute works without checking |

Depth over breadth: two operators done rigorously beats five done shallowly.

---

## 3. Repository Structure

```text
cuda-kernel-optimization-lab/
├── README.md
├── REPORT.md
├── CMakeLists.txt
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── optimization-notes.md
│   └── figures/
├── include/
│   ├── tensor_utils.h
│   ├── benchmark_utils.h
│   └── cuda_check.h
├── cpu/
│   ├── gemm_cpu.cpp
│   └── softmax_cpu.cpp
├── kernels/
│   ├── gemm/
│   │   ├── gemm_naive.cu
│   │   ├── gemm_tiled.cu
│   │   ├── gemm_optimized.cu
│   │   └── gemm_launcher.cpp
│   └── softmax/
│       ├── softmax_naive.cu
│       ├── softmax_optimized.cu
│       └── softmax_launcher.cpp
├── benchmarks/
│   ├── benchmark_gemm.cpp
│   ├── benchmark_softmax.cpp
│   └── results/            (gitignored - regenerated per session)
├── tests/
│   ├── test_gemm.cpp
│   ├── test_softmax.cpp
│   └── test_utils.cpp
├── profiling/
│   ├── ncu/                (only populated if Tier 3 validates - see §7)
│   ├── nsys/
│   └── notes/
└── scripts/
    ├── env_check.sh         # run first, every session
    ├── stamp_env.sh         # prepend to every benchmark output
    └── build.sh
```

Every directory above already exists in this skeleton (as `.gitkeep` files)
except actual kernel/cpp logic — that's written phase by phase, gated by
`reasoning.md` first.

---

## 4. Environment

### 4.1 Target hardware

**Google Colab, free tier — T4-class GPU.** This is a deliberate constraint,
not an oversight:

- Colab's free GPU allocation is **non-deterministic** session to session —
  re-run `env_check.sh` every session, don't assume continuity.
- Clock speeds are **not pinnable** (`nvidia-smi -lgc` is unavailable on free
  tier), so high-variance runs are expected — median + std dev (§6) exist
  because of this, not as boilerplate.
- Whether `ncu`/`nsys` have the access they need is **tested every session**,
  not assumed. Session 1 result: `ncu` validated successfully (see §13) —
  this contradicts the commonly-cited free-tier restriction, which is exactly
  why this gets tested instead of taken on faith.

Every consequence of this is handled explicitly below, not papered over.

### 4.2 Mandatory pre-Phase-1 validation

**Before writing a single line of kernel code**, run:

```bash
bash scripts/env_check.sh
```

in the first cell of a fresh Colab session. It checks, in order: GPU model
and driver, `nvcc` version, `nsys` availability, and — critically — whether
`ncu` can actually read hardware counters (it compiles and profiles a
trivial no-op kernel and checks for `ERR_NVGPUCTRPERM`).

Record the result in the [Environment Validation Log](#13-environment-validation-log)
**before** starting Phase 1. The result determines your profiling tier for
the whole project (see §7) — this is not optional bookkeeping.

### 4.3 Setup steps (Colab)

1. **Runtime → Change runtime type → T4 GPU.** Confirm it actually attached
   (`!nvidia-smi`) — free tier sometimes gives you no GPU at all if quota is
   exhausted.
2. **Persist your work.** Colab's local disk is ephemeral and resets between
   sessions. Either:
   - mount Google Drive (`from google.colab import drive; drive.mount(...)`)
     and keep the repo there, or
   - `git init` this skeleton and `git push` to a GitHub repo at the end of
     every session.
   Don't lose a session's work to a runtime reset.
3. Get this skeleton into the session (unzip into `/content/` or your
   mounted Drive path).
4. `cd cuda-kernel-optimization-lab && bash scripts/env_check.sh`
5. Record the output in §13.
6. Only then start Phase 1 — and start with `reasoning.md`, not code.

---

## 5. Build System

CMake is the project's build system, targeting `sm_75` (T4) by default —
see `CMakeLists.txt`. In Colab:

```bash
!mkdir -p build && cd build && cmake .. && make
```

For fast single-file iteration during early phases (before targets are wired
into CMake), compiling directly is fine and often faster to iterate with:

```bash
!nvcc -Xptxas -v -arch=sm_75 kernels/gemm/gemm_naive.cu kernels/gemm/gemm_launcher.cpp -o gemm_naive
```

The `-Xptxas -v` flag matters even outside CMake — it's the register/spill
count check from Tier 1 profiling, and it costs nothing to always include.

---

## 6. Benchmark Methodology

Every benchmark run must:

1. **Stamp the environment** — GPU name, driver version, total memory, CUDA
   toolkit version, timestamp (UTC) — via `scripts/stamp_env.sh`, saved
   alongside the benchmark output (CSV header or markdown table preamble).
2. **Warm up** before timing (discard the first N iterations).
3. Run **multiple timed iterations**, and report:
   - median latency (required — not average; Colab variance can be high)
   - min / max
   - standard deviation (**required, not optional** — a project whose pitch
     is "measure, don't guess" doesn't get to skip the one number that
     proves the measurement is trustworthy)
4. **Assert GPU-name match** before computing any speedup that compares two
   runs (e.g. naive vs. tiled). Colab can silently hand you a different
   physical card between sessions, even on free tier — a speedup computed
   across two different GPUs is not a real number.
5. Record the operator, version, and input shape alongside every row.

Output format — one table per operator, e.g.:

| Version | Shape | Median latency (ms) | Std dev | Speedup vs CPU | Speedup vs naive CUDA | GPU |
|---|---|---|---|---|---|---|
| CPU | 1024×1024 | — | — | 1.0× | – | — |
| CUDA naive | 1024×1024 | — | — | — | 1.0× | — |
| CUDA tiled | 1024×1024 | — | — | — | — | — |
| CUDA optimized | 1024×1024 | — | — | — | — | — |

(Filled in only once real runs exist — see `REPORT.md`.)

---

## 7. Profiling Tiers

Because `ncu` access on free Colab is unverified, profiling is tiered so the
project's evidence doesn't depend on a tool that might not work:

| Tier | Tool | Availability | What it gives you |
|---|---|---|---|
| **1 — always available** | `nvcc -Xptxas -v` | No permissions needed, works every session | Registers/thread, spill counts — direct signal for "did this optimization blow up register pressure" |
| **1 — always available** | Wall-clock benchmarking (§6) | No permissions needed | Median/std latency, real speedup numbers |
| **2 — usually available** | Nsight Systems (`nsys`) | Doesn't need counter access, typically works on Colab — **confirmed on this image, not on PATH by default; see §13** | Kernel duration timeline, H2D/D2H copy overhead, launch/sync overhead |
| **3 — bonus, gate-checked** | Nsight Compute (`ncu`) | **Only if `env_check.sh` confirms no `ERR_NVGPUCTRPERM`** — **validated working as of Session 1** | Occupancy, DRAM throughput, shared-memory usage, warp stall reasons |

Tier 3 claims (occupancy numbers, warp stall analysis) only appear in
`REPORT.md` if `ncu` validated successfully for that session — never assumed.

> **Session 1 reality check:** `ncu` (Tier 3) validated cleanly on the first
> pass. `nsys` initially reported "not found" only because it isn't on
> `PATH` — it's bundled under Nsight Compute's own install tree
> (`/opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys`).
> `nsys --version` confirmed clean once called directly / added to `PATH`.
> `env_check.sh` now auto-discovers it there if missing from `PATH`. All
> three tiers are live this session — re-verify every session regardless,
> Colab does not guarantee the same image or GPU next time.

---

## 8. Correctness Criteria

- **GEMM**: elementwise comparison against the CPU reference, tolerance
  `1e-4`–`1e-5` (float32).
- **Softmax**: row-wise comparison against CPU reference at the same
  tolerance, plus an explicit check that each output row sums to
  approximately `1.0`, and a stress test with large-magnitude inputs to
  confirm max-subtraction is actually preventing overflow (not just present
  in the code).

---

## 9. Learning Workflow (per kernel / per phase)

1. Read/discuss the concept once.
2. Close the reference.
3. Write `reasoning.md` for that kernel — in your own words, no peeking:
   - thread/block mapping
   - what each thread computes
   - what goes into shared memory and why
   - why the indexing is correct
4. Only then write `attempt.cu`.
5. Compare against feedback/reference. If `reasoning.md` was wrong, that's
   the actual learning signal — fix the reasoning, then the code.

If a kernel's logic can't be explained from a blank page without looking
anything up, it doesn't count as learned yet, regardless of whether the code
compiles and passes the correctness test.

### Phase order (strict, sequential — no parallel phases)

1. Project scaffolding + CPU baselines (this skeleton + `gemm_cpu.cpp` /
   `softmax_cpu.cpp`)
2. Naive CUDA GEMM
3. Tiled (shared-memory) CUDA GEMM
4. Optimized CUDA GEMM (register accumulation, loop unrolling — only kept if
   the benchmark actually shows a gain, see §12)
5. Naive CUDA Softmax
6. Optimized (block-reduction) CUDA Softmax
7. Benchmark harness polish + correctness test suite
8. Profiling pass (tier depends on §4.2 result) + `REPORT.md`

---

## 10. Deliverables Checklist

- [ ] CPU + CUDA source for both operators, all versions
- [ ] `README.md` (this file) — kept current as scope/findings evolve
- [ ] `REPORT.md` — filled in only with real data, phase by phase
- [ ] Correctness test suite passing for every kernel version
- [ ] Benchmark tables (environment-stamped, median+std) for every version/shape
- [ ] Profiling artifacts for whatever tier validated (§7)
- [ ] Environment Validation Log (§13) covering every session used

---

## 11. Scope Guardrails

- Don't add more than 2 operators — depth over breadth.
- Don't reach for Tensor Cores / WMMA in this version — that's a future
  extension, not the MVP.
- Don't write benchmark/profiling sections with placeholder numbers.
- Don't skip CPU baselines — without them, "speedup" claims have no anchor.
- Don't assume `ncu` works because the original plan says "use Nsight
  Compute" — verify every session.
- Don't compare benchmark runs across sessions without checking the GPU
  stamp matches.

---

## 12. Resume Claims Policy

No resume bullet or README "key learning" is written **before** the
benchmark that proves it exists. Specifically:

| Claim type | Required evidence before writing it |
|---|---|
| "Optimized X using technique Y" | A benchmark row showing Y measurably beat the prior version, on a matching GPU stamp |
| "Achieved N× speedup" | The actual median-latency ratio from a same-session, GPU-matched comparison |
| "Profiled with Nsight Compute" | A logged `ncu` session in §13 with `ERR_NVGPUCTRPERM` absent, plus a saved `.ncu-rep` |
| "Reduced global memory traffic via tiling" | The derivation (already done — see project history) *and* a benchmark showing tiled beats naive |

If an optimization (e.g. loop unrolling) gives a 2% gain instead of a real
one, the report says 2%, and the resume bullet either gets cut or gets
specific instead of vague. No claim is pre-committed before the data exists.

---

## 13. Environment Validation Log

Filled in after every `env_check.sh` run. This is what makes benchmark
comparisons across sessions trustworthy instead of assumed.

| Date (UTC) | Session # | GPU | Driver | CUDA toolkit | `ncu` status | `nsys` status | Notes |
|---|---|---|---|---|---|---|---|
| 2026-06-24 | 1 | Tesla T4, 15360 MiB | 580.82.07 | 12.8 (V12.8.93) | ✅ Validated — no `ERR_NVGPUCTRPERM`, 9-pass profile completed on noop kernel | ✅ Validated — `nsys --version` → `NVIDIA Nsight Systems version 2025.1.1.0`, located at `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`, not on PATH by default | All 3 tiers live this session. Re-verify every new session — don't assume continuity. |

---

## 14. Progress Tracker

| Phase | Status |
|---|---|
| 0. Environment validation (`env_check.sh`) | ✅ Done (Session 1, 2026-06-24) — Tier 1/2/3 all confirmed live |
| 1. Scaffolding + CPU baselines | ⬜ Not started |
| 2. Naive CUDA GEMM | ⬜ Not started |
| 3. Tiled CUDA GEMM | ⬜ Not started |
| 4. Optimized CUDA GEMM | ⬜ Not started |
| 5. Naive CUDA Softmax | ⬜ Not started |
| 6. Optimized CUDA Softmax | ⬜ Not started |
| 7. Benchmark harness + tests | ⬜ Not started |
| 8. Profiling + `REPORT.md` | ⬜ Not started |
