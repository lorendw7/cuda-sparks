# Project Architecture

Read this once and the whole project clicks. It shows **how the pieces fit**, **what data flows
where**, and **which code is yours to write** vs. which is plumbing.

---

## 1. What the program does each frame

The whole app is one loop that repeats ~60+ times per second. Every frame:

```
        ┌──────────────────────────────────────────────────────────────┐
        │                    main loop  (main.cpp)                       │
        │                                                                │
        │  1. measure dt (seconds since last frame)                      │
        │  2. sim.update(dt)      ── advance the physics                 │
        │  3. sim.to_vertices(v)  ── turn particles into [x,y,r,g,b]      │
        │  4. renderer.upload(v)  ── send that to the GPU                 │
        │  5. renderer.draw()     ── draw the points                     │
        │  6. swap buffers        ── show the frame                      │
        └──────────────────────────────────────────────────────────────┘
                 repeat forever
```

`dt` (delta-time) keeps motion at the same real speed no matter the frame rate. Steps 3–6 are the
same in every phase; only step 2 (`update`) changes — first on the CPU, then on the GPU.

---

## 2. Who owns the data

| Phase | Where particles live | How `update` runs |
|-------|----------------------|-------------------|
| 1 | CPU RAM (`std::vector<Particle>`) | a `for` loop over all particles |
| 2+ | **GPU VRAM** (`d_particles_`) + a CPU mirror (`host_`) for drawing | a **kernel** on thousands of threads; copy back to `host_` each frame |

The key Phase-2 idea: the GPU has its **own separate memory**. You `cudaMalloc` an array there,
`cudaMemcpy` data across, run the kernel, then `cudaMemcpy` the results back so the renderer (which
reads CPU memory) can draw them.

---

## 3. The components

```
   main.cpp ──────────────► ParticleSystem ──────────────► Renderer ────► screen
   (the loop, timing)       (the simulation)               (OpenGL)
                                  │
                         ┌────────┴────────┐
                         │ Phase 1: CPU     │   particle_system.h  (for-loop update)
                         │ Phase 2: CUDA    │   particle_system.cu (kernel + cudaMemcpy)
                         └─────────────────┘
```

| Component | File(s) | Role |
|-----------|---------|------|
| **App loop** | `main.cpp` | window, timing, calls update/draw each frame |
| **Simulation** | `particle_system.h` / `.cu` | holds the particles, advances the physics |
| **Renderer** | `renderer.cpp` / `.h` | turns `[x,y,r,g,b]` into glowing points via shaders |
| **Shared types** | `include/particles.h` | the `Particle` struct + `SimParams` |
| **CUDA helpers** | `include/cuda_utils.h` | `CUDA_CHECK` (error check) + `gpu_info` |

**The recurring design idea:** `main.cpp` always calls the same three methods — `update`,
`to_vertices`, `size`. It never knows or cares whether the simulation is CPU or GPU. That clean
*interface* is why Phase 2 changed everything *inside* `ParticleSystem` without touching `main.cpp`
at all. This is the most important software-design lesson in the project.

---

## 4. Who writes what

The rule: **you write the important logic; I provide the plumbing.**

| You write (the learning) | I provide (plumbing) |
|---|---|
| Physics: gravity, motion, collisions (`update`) | Window + input + OpenGL context |
| The CUDA kernel + `cudaMalloc`/`cudaMemcpy`/launch | The frame loop & timing in `main.cpp` |
| Shaders (how particles look) | Shader compile/link boilerplate, VBO/VAO setup |
| Color-over-life, forces, presets (Phase 3) | `CUDA_CHECK` macro, `gpu_info`, struct definitions |
| curand random init (Phase 2 L2) | The build system (`build.bat`, CMake, FetchContent) |

"Important" = anything that decides **how the simulation behaves or looks**. "Plumbing" = the fixed
boilerplate every such program needs. Some helpers (like the temporary `emitter_spawn`) are plumbing
*for now* and become yours later (you'll rewrite spawning with real curand RNG in Phase 2 Level 2).

---

## 5. How a phase is structured

Each phase is split into **Levels** — small, buildable steps. You finish a level, run it, see the
result, then move on. Current status:

```
Phase 1 (CPU)   ✅ L1 physics  ✅ L2 bounce  ✅ L3 shaders
Phase 2 (CUDA)  ✅ L1 kernel + memory  ✅ L2 curand  ✅ L3 measure speedup
Phase 3         ✅ L1 gravity well  ✅ L2 naive N-body (O(n^2))  ✅ L3 __shared__ tiling
Phase 4         ▶  1,000,000 particles + Nsight profiling; rendering layer built from scratch
```

---

## 6. Mental model to keep

> **CPU is the manager, GPU is the army.** The CPU (host) sets things up, hands work to the GPU
> (device), and collects results. The GPU runs one small program (the kernel) in thousands of
> identical threads, each doing one particle. Everything in Phases 2–4 is a variation on "give the
> army better instructions and stop wasting time shuffling data between them."
