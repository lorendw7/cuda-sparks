# cuda-sparks

Build a real-time GPU particle system from scratch — and learn CUDA along the way.

We go in four phases, from a pure-CPU baseline all the way to **1,000,000 particles** running
smoothly on the GPU. Each phase has a working program you can run, and the parts where the real
learning happens are left for **you** to write. The framework (windowing, OpenGL rendering, build
system, timing) is provided so you can focus on the compute.

> **How the teaching works**
> - Lesson READMEs are bilingual (中文 + English) so concepts are easy to absorb.
> - Reference docs and all code comments are English-only (ASCII).
> - Three tracks are taught together as you go: **C++**, **CUDA**, and **OpenGL** (shaders + the
>   render pipeline) — the rendering layer is not a black box.
> - **You write the core logic** (the CPU update loop, the shaders, then the CUDA kernels). The
>   framework code is provided and explained.

---

## The Roadmap

| Phase | Goal | What you learn | What you write |
|-------|------|----------------|----------------|
| **1 — CPU Baseline** | 10,000 particles on the CPU, rendered with OpenGL | C++ sim loop, OpenGL points, frame timing, a reference benchmark | `ParticleSystem::update()` on the CPU |
| **2 — CUDA Migration** | Move the physics to a GPU kernel | `cudaMalloc` / `cudaMemcpy`, thread indexing, `curandState` RNG, your first speedup | the update **kernel** + init **kernel** |
| **3 — Effects** | Gravity, collisions, color fade by lifetime | `__shared__` memory, tiling, `__syncthreads()` | force & interaction kernels |
| **4 — One Million** | 1,000,000 particles at interactive frame rates | CUDA–OpenGL interop, profiling with **Nsight**, memory-layout (SoA) optimization | the optimized pipeline |

Phase 1 is in [`phases/phase1_cpu_baseline/`](phases/phase1_cpu_baseline/). Each later phase appears
as you finish the one before it.

---

## Prerequisites

| Requirement | Verify with | Notes |
|-------------|-------------|-------|
| NVIDIA GPU (Compute Capability ≥ 6.0) | `nvidia-smi` | Needed from Phase 2 on |
| CUDA Toolkit ≥ 12.8 | `nvcc --version` | You have 13.3 ✓ |
| CMake ≥ 3.18 | `cmake --version` | You have 4.4 ✓ |
| Visual Studio 2022 (MSVC) | — | The C++/CUDA host compiler |
| Internet (first build only) | — | CMake auto-downloads GLFW |

You do **not** need to install OpenGL or GLFW by hand — the build downloads and compiles GLFW
for you via CMake `FetchContent`, and a GL loader (glad) is generated at configure time.

---

## Build & Run

```bash
cmake -B build -S .
cmake --build build --parallel
```

Run Phase 1 (Windows):

```bash
.\build\phases\phase1_cpu_baseline\Release\phase1.exe
```

> The first configure downloads GLFW and glad — give it a minute. Later builds are fast.

---

## Project Layout

```
cuda-sparks/
├── CMakeLists.txt              ← root build; pulls in GLFW + glad
├── include/
│   ├── particles.h            ← Particle struct and SimParams (shared by all phases)
│   └── cuda_utils.h           ← CUDA_CHECK macro and gpu_info() (used from Phase 2)
├── docs/
│   └── cuda_reference.md      ← CUDA syntax quick reference
└── phases/
    └── phase1_cpu_baseline/
        ├── README.md          ← the lesson: read this first
        ├── CMakeLists.txt
        └── src/
            ├── main.cpp           ← app loop + timing (provided)
            ├── particle_system.h  ← YOUR work: the CPU update loop
            └── renderer.h/.cpp    ← OpenGL point renderer (you edit shaders in Level 3)
```

---

## How to Use a Phase

1. Read the phase `README.md` — it explains the concept in 中文 + English.
2. Open the file you're meant to edit and find every `// ── TODO ──` block.
3. Write your code in the marked sections only.
4. Build, run, and watch the particles (and the FPS counter in the title bar).
5. Stuck? Each README has a **Hints** section at the bottom.
