# cuda-sparks

Build a real-time GPU particle system from scratch — and learn CUDA along the way.

We go in four phases, from a pure-CPU baseline through **1,000,000 particles** running smoothly on
the GPU. The final phase then grows that simulation into a polished, shippable demo — with a
menu / fullscreen presentation shell and procedurally-generated, audio-reactive sound. Each phase
has a working program you can run, and the parts where the real learning happens are left for
**you** to write. The framework (windowing, OpenGL rendering, build system, timing) is provided so
you can focus on the compute.

> **How the teaching works**
> - All docs (READMEs, references) and code comments are English-only (ASCII).
> - Three tracks are taught together as you go: **C++**, **CUDA**, and **OpenGL** (shaders + the
>   render pipeline) — the rendering layer is not a black box. A fourth track, **audio**, joins in
>   Phase 4.
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
| **4 — One Million** | 1,000,000 particles at interactive frame rates, then a full demo | CUDA–OpenGL interop, profiling with **Nsight**, memory-layout (SoA) optimization | the optimized pipeline, **the rendering layer from scratch**, plus two application tracks (below) |

Phase 1 is in [`phases/phase1_cpu_baseline/`](phases/phase1_cpu_baseline/). Each later phase appears
as you finish the one before it.

> **Phase 4 is where it all comes together.** Its CUDA / performance spine (levels **L1–L7**) gets
> the 1M sim running fast. Two further **application tracks** then layer onto that *same*
> `phase4_one_million/src/` codebase (one `main.cpp` / renderer / sim — no forked phase folders)
> and are planned in their own docs:
>
> - [`PRESENTATION.md`](phases/phase4_one_million/PRESENTATION.md) — a Dear ImGui menu, real
>   fullscreen mode, a scrolling telemetry HUD, and a hands-off auto-play loop.
> - [`AUDIO.md`](phases/phase4_one_million/AUDIO.md) — the **audio track**: procedural,
>   zero-license sound driven by live simulation state (this was "Phase 5" in an earlier plan, now
>   folded in as a track on the same app).
>
> Phase 4's [`README.md`](phases/phase4_one_million/README.md) stays the CUDA/performance spine.
>
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
        ├── README.md          ← CUDA / performance spine: levels L1–L7 (read first)
        ├── PRESENTATION.md    ← app track: menu / fullscreen / auto-play (after L6)
        ├── AUDIO.md           ← app track: procedural, audio-reactive sound (last)
        ├── CMakeLists.txt
        └── src/               ← one codebase for the sim + both app tracks
            ├── main.cpp           ← window + main loop + preset keys + timing
            ├── renderer.h/.cpp    ← YOUR work: hand-written GL point renderer (RAII class)
            ├── particle_system.h  ← host-side class interface (SoA + interop)
            ├── particle_system.cu ← YOUR work: 1M SoA sim, CUDA-GL interop, emitters/presets
            ├── hud_log.h          ← Presentation track: scrolling telemetry console
            └── audio.h/.cpp       ← Audio track: miniaudio playback device + synthesized sound
```

---

## How to Use a Phase

1. Read the phase `README.md` — it explains the concept and what you'll write.
2. Open the file you're meant to edit and find every `// ── TODO ──` block.
3. Write your code in the marked sections only.
4. Build, run, and watch the particles (and the FPS counter in the title bar).
5. Stuck? Each README has a **Hints** section at the bottom.
