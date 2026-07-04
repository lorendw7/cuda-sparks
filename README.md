# cuda-sparks

Build a real-time GPU particle system from scratch — and learn CUDA along the way.

We go in five phases, from a pure-CPU baseline through **1,000,000 particles** running smoothly on
the GPU and on to procedurally-generated, audio-reactive sound. Each phase has a working program you
can run, and the parts where the real learning happens are left for **you** to write. The framework
(windowing, OpenGL rendering, build system, timing) is provided so you can focus on the compute.

> **How the teaching works**
> - All docs (READMEs, references) and code comments are English-only (ASCII).
> - Three tracks are taught together as you go: **C++**, **CUDA**, and **OpenGL** (shaders + the
>   render pipeline) — the rendering layer is not a black box.
> - **You write the code.** Early phases leave the core logic (the CPU update loop, the CUDA
>   kernels) for you; from Phase 3 on you write essentially everything. The framework is explained
>   as you go.

---

## The Roadmap

| Phase | Goal | What you learn | What you write |
|-------|------|----------------|----------------|
| **1 — CPU Baseline** | 10,000 particles on the CPU, rendered with OpenGL | C++ sim loop, OpenGL points, frame timing, a reference benchmark | `ParticleSystem::update()` on the CPU |
| **2 — CUDA Migration** | Move the physics to a GPU kernel | `cudaMalloc` / `cudaMemcpy`, thread indexing, `curandState` RNG, your first speedup | the update **kernel** + init **kernel** |
| **3 — Effects** | Gravity, collisions, color fade by lifetime | `__shared__` memory, tiling, `__syncthreads()` | force & interaction kernels |
| **4 — One Million** | 1,000,000 particles at interactive frame rates | CUDA–OpenGL interop, profiling with **Nsight**, memory-layout (SoA) optimization | the optimized pipeline **and the rendering layer from scratch** |
| **5 — Audio** | Procedurally-generated, audio-reactive sound | real-time audio with miniaudio, code-synthesized SFX, driving sound from live sim state | the audio layer (event → ambient → reactive) |

Phase 1 is in [`phases/phase1_cpu_baseline/`](phases/phase1_cpu_baseline/). Each later phase appears
as you finish the one before it.

> **New here? Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first** — it maps how every piece
> fits together and exactly which code is yours to write vs. provided plumbing.

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

Use the helper script — it sets up the MSVC + CUDA environment, configures with the **Ninja**
generator, and builds everything:

```bash
.\build.bat          REM build all phases
.\build.bat run1     REM build, then run Phase 1 (CPU)
.\build.bat run2     REM build, then run Phase 2 (CUDA)
.\build.bat run3     REM build, then run Phase 3 (Effects)
.\build.bat run4     REM build, then run Phase 4 (One Million)
.\build.bat clean    REM delete build/ and start fresh
```

> Why Ninja (not the Visual Studio generator)? VS 2026 does not ship the CUDA toolset integration,
> so the VS generator cannot compile `.cu` files. Ninja (bundled with VS) calls `nvcc` directly and
> needs no admin install. `build.bat` points CMake at it for you.

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
│   ├── ARCHITECTURE.md       ← how it all fits + who writes what (read first)
│   └── cuda_reference.md     ← CUDA syntax quick reference
└── phases/
    ├── phase1_cpu_baseline/
    │   ├── README.md          ← the lesson: read this first
    │   ├── CMakeLists.txt
    │   └── src/
    │       ├── main.cpp           ← app loop + timing (provided)
    │       ├── particle_system.h  ← YOUR work: the CPU update loop
    │       └── renderer.h/.cpp    ← OpenGL point renderer (you edit shaders in Level 3)
    ├── phase2_cuda_migration/
    │   ├── README.md          ← the lesson: read this first
    │   ├── CMakeLists.txt
    │   └── src/
    │       ├── main.cpp           ← app loop (identical shape to Phase 1)
    │       ├── particle_system.h  ← host-side class interface (plain C++)
    │       ├── particle_system.cu ← YOUR work: the CUDA kernel + memory plumbing
    │       └── renderer.h/.cpp    ← OpenGL point renderer (same as Phase 1)
    ├── phase3_effects/
    │   ├── README.md          ← the lesson: read this first
    │   ├── CMakeLists.txt
    │   └── src/
    │       ├── main.cpp           ← app loop (1280x1280 window)
    │       ├── particle_system.h  ← host-side class interface
    │       ├── particle_system.cu ← YOUR work: emitters + swirl + tiled N-body kernels
    │       └── renderer.h/.cpp    ← OpenGL point renderer (fully commented reference)
    └── phase4_one_million/
        ├── README.md          ← the lesson: read this first
        ├── CMakeLists.txt
        └── src/
            ├── main.cpp           ← window + main loop + preset keys + timing
            ├── renderer.h/.cpp    ← YOUR work: hand-written GL point renderer (RAII class)
            ├── particle_system.h  ← host-side class interface (SoA + interop)
            └── particle_system.cu ← YOUR work: 1M SoA sim, CUDA-GL interop, emitters/presets
```

---

## How to Use a Phase

1. Read the phase `README.md` — it explains the concept and what you'll write.
2. Open the file you're meant to edit and find every `// ── TODO ──` block.
3. Write your code in the marked sections only.
4. Build, run, and watch the particles (and the FPS counter in the title bar).
5. Stuck? Each README has a **Hints** section at the bottom.
