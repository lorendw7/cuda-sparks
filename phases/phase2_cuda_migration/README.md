# Phase 2 — CUDA Migration

**Goal:** Move the particle physics from the CPU to the **GPU**. You'll write your first CUDA
**kernel**, learn how data travels between CPU and GPU memory (`cudaMalloc` / `cudaMemcpy`), and see
the simulation run on thousands of GPU threads at once.

This is **Level 1** of Phase 2: get the kernel running. Level 2 (curand) and Level 3 (measure the
speedup) come after.

---

## 1. The one big idea: two separate memories

Your computer has **two** separate pools of memory:

```
   CPU (host)                            GPU (device)
   ┌────────────────┐                   ┌────────────────┐
   │  normal RAM    │   cudaMemcpy      │   VRAM          │
   │  std::vector   │ ───────────────►  │  d_particles_   │
   │  host_         │ ◄───────────────  │  (kernel runs)  │
   └────────────────┘                   └────────────────┘
```

- The **CPU** can only touch CPU memory; the **GPU** can only touch GPU memory.
- To compute on the GPU, you must **copy data over** (`cudaMemcpy`), let the GPU work, then **copy
  results back**.
- A pointer to GPU memory (like `d_particles_`, the `d_` means *device*) is a real address, but the
  CPU **cannot dereference it** — `d_particles_[0]` on the CPU would crash.

This "copy over → compute → copy back" dance is the essence of CUDA programming.

---

## 2. What a kernel is

A **kernel** is a function marked `__global__`. You don't call it once — you **launch** it across
many threads with the special `<<<grid, block>>>` syntax:

```cpp
update_kernel<<<grid, block>>>(d_particles_, n_, params_, dt);
//             ▲ how many threads to start
```

The GPU then runs the **same** kernel code in thousands of threads simultaneously. Threads are
organized in a two-level hierarchy:

```
   grid
   ├── block 0   ── thread 0, 1, 2, ... 255
   ├── block 1   ── thread 0, 1, 2, ... 255
   └── block 2   ── thread 0, 1, 2, ... 255   (we use 256 threads per block)
```

Inside the kernel, three built-in variables tell each thread who it is:

| Variable | Meaning |
|----------|---------|
| `threadIdx.x` | my lane within my block (0..255) |
| `blockIdx.x`  | which block I belong to |
| `blockDim.x`  | how many threads per block (256) |

So **my global particle index** is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Analogy: think of seats in a theater. `blockDim.x` is "seats per row," `blockIdx.x` is "which row,"
and `threadIdx.x` is "which seat within the row." Row number × seats per row + seat number = a unique
seat number `i` for the whole theater. Each thread computes its own `i`, then handles only particle
number `i`.

This single line **replaces the `for` loop** from Phase 1. In Phase 1 one CPU thread walked
`i = 0, 1, 2, ...` one at a time. Here, thousands of GPU threads each grab their own `i` and run
**at the same time**. Same physics, no loop.

---

## 3. Your task

You write **all the core CUDA logic** in [`src/particle_system.cu`](src/particle_system.cu). I only
provide the RNG helper (`emitter_spawn`) and the render packing (`to_vertices`). There are two parts:

**Part 1 — the kernel** (`update_kernel`): write the whole body.

| Step | Code shape |
|------|------------|
| index | `int i = blockIdx.x * blockDim.x + threadIdx.x;` |
| guard | `if (i >= n) return;` |
| physics | your Phase 1 logic on `particles[i]` (gravity, move, 4 wall bounces, age) |
| recycle | `if (particles[i].life <= 0.0f) emitter_spawn(particles[i], i);` |

**Part 2 — the memory operations** (TODO A–F): the `cudaMalloc` / `cudaMemcpy` / launch you came to
learn. Each TODO comment shows the exact API shape — read it, then write the line.

| TODO | Where | What |
|------|-------|------|
| **A** | constructor | `cudaMalloc` — allocate the array in GPU memory |
| **B** | constructor | `cudaMemcpy` Host→Device — copy initial state up to the GPU |
| **C** | destructor | `cudaFree` — release GPU memory |
| **D** | `update()` | compute launch shape (`grid`, `block`) |
| **E** | `update()` | launch `update_kernel<<<grid, block>>>(...)` + `cudaGetLastError()` |
| **F** | `update()` | `cudaMemcpy` Device→Host — copy results back for rendering |

The file won't compile until all of it is filled — write top to bottom, then build.

### Why TODO 2 matters

We launch threads in blocks of 256. For 10,000 particles we need 40 blocks = 10,240 threads — that's
**240 more threads than particles**. Those extra threads have `i = 10000..10239`, which is past the
end of the array. Without `if (i >= n) return;` they'd read/write invalid memory and crash (or
corrupt data). This guard is in **every** CUDA kernel — make it a habit.

---

## 4. Build & run

```bash
.\build.bat run2
```

**What you should see:**
- A console line naming your GPU (e.g. `[GPU] NVIDIA GeForce ... | SM 8.9 | ...`).
- The same fountain as Phase 1, but now computed on the GPU.
- A title bar: `... | 10000 particles | <FPS> | update <ms>`.

You may notice the fountain looks a bit **repetitive / mechanical** — every particle relives the
exact same life. That's because our `emitter_spawn` fakes randomness from the particle index. **Level
2 fixes this with curand** (real per-thread random numbers). For now, repetitive is correct.

---

## 5. Feel the speedup

This is the payoff. Open `src/main.cpp`, change `kNumParticles`, rebuild, and compare the `update`
time in the title bar against your Phase 1 CPU baseline:

| Particles | Phase 1 CPU (`update` ms) | Phase 2 GPU (`update` ms) |
|-----------|---------------------------|---------------------------|
| 10,000    | (your baseline)           |                           |
| 100,000   |                           |                           |
| 1,000,000 | (probably a slideshow)    |                           |

> ⚠️ Note: Phase 2's `update` time **includes the cost of copying the data back to the CPU every
> frame** (`cudaMemcpy`). That's why the GPU isn't yet at its full potential — Phase 4 will use
> CUDA–OpenGL interop to eliminate this copy and make a million particles truly smooth. Even so, you
> should already see the GPU is clearly faster at large particle counts.

---

## 6. Checklist

- [ ] Console prints your GPU name (proves the CUDA code ran).
- [ ] The fountain animates (particles recycle, screen doesn't go empty).
- [ ] You compared GPU vs CPU `update` time at 100k or 1M particles.

Cleared? Next is **Level 2 — curand**: give every particle its own real random stream so the
fountain looks natural.

---

## Hints

<details>
<summary>Build error: "identifier i is undefined"</summary>

You haven't written TODO 1 yet. The physics code uses `i`, so the kernel won't compile until you
declare `int i = blockIdx.x * blockDim.x + threadIdx.x;`.
</details>

<details>
<summary>The window opens but is empty / crashes immediately.</summary>

Most likely TODO 2 is missing. Without `if (i >= n) return;`, the extra threads write past the array
and corrupt memory. Add the guard right after computing `i`.
</details>

<details>
<summary>The fountain runs once then the screen goes black.</summary>

That's TODO 3 — particles age to death and never come back. Add
`if (particles[i].life <= 0.0f) emitter_spawn(particles[i], i);` at the end of the kernel.
</details>

<details>
<summary>Show me the finished kernel.</summary>

```cpp
__global__ void update_kernel(Particle* particles, int n, SimParams params, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // TODO 1
    if (i >= n) return;                              // TODO 2

    particles[i].vy -= params.gravity * dt;          // (physics, pre-filled)
    particles[i].x  += particles[i].vx * dt;
    particles[i].y  += particles[i].vy * dt;
    // ... wall bounces ...
    particles[i].life -= dt;

    if (particles[i].life <= 0.0f) emitter_spawn(particles[i], i);   // TODO 3
}
```

Try it yourself first — these three lines are the core CUDA idiom you'll reuse in every kernel.
</details>
