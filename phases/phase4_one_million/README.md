# Phase 4 — One Million Particles

> **Goal:** push the particle count from ~10,000 to **1,000,000** and keep it smooth.
> In this phase you write **100% of the code from scratch** (rendering layer included);
> I only provide the spec, function signatures, and gotchas line by line.

---

## Why 10k runs fine but 1M chokes — the wall

Through Phase 3, every frame moves data like this:

```text
GPU kernel ──writes──> d_particles_ (device memory)
                          │
                          │  cudaMemcpy  (once per frame, whole array GPU→CPU)
                          ▼
                       host_ (system RAM)
                          │
                          │  to_vertices()  (CPU packs into [x,y,r,g,b])
                          ▼
                    OpenGL VBO ──> screen
```

At 10,000 particles this chain is fine. At 1,000,000 particles we **copy the entire
array from device to host every frame, pack it on the CPU, then upload it back to the
GPU** — that GPU→CPU→GPU round trip is the killer. PCIe bandwidth and the CPU packing
loop crush the frame rate.

The core of Phase 4 is to **tear out that round trip**: let the GPU write vertices
straight into the OpenGL buffer so the CPU never touches particle data.

---

## Levels

### L1 — Scale to 1M & measure the bottleneck

- Bump the particle count to 1,000,000. **Do not** change the architecture yet.
- Split the per-frame cost into three separately-timed segments (update / pack / upload)
  so you can see, not guess, where the time goes.
- Run Nsight Systems / Nsight Compute once to confirm the time sits in `cudaMemcpy` and
  the CPU pack.
- Deliverable: a "slow but correct" 1M baseline to compare every later optimization against.

### L2 — CUDA–OpenGL interop: kill the round trip

- Register the OpenGL VBO with CUDA via `cudaGraphicsGLRegisterBuffer`.
- Each frame: `cudaGraphicsMapResources` → get a device pointer → kernel **writes vertices
  directly into the VBO** → `cudaGraphicsUnmapResources`.
- Delete the `host_` mirror, delete the per-frame `cudaMemcpy`, delete `to_vertices()`.
- This is the biggest architectural change in Phase 4 and the key to the frame rate
  recovering.

### L3 — Structure of Arrays (SoA)

- Split the array-of-structs `struct Particle{ x,y,vx,vy,... }` (AoS) into **one array per
  field** (SoA): `float* x; float* y; float* vx; ...`.
- Why: when the 32 threads of a warp read `x[i]` together, SoA makes those addresses
  contiguous → **coalesced memory access**; AoS strides across memory and wastes bandwidth.
- Use Nsight to compare L2 vs L3 memory throughput / frame rate and quantify the gain.

### L4 — Nsight profiling deep-dive

- Inspect occupancy, memory throughput, warp utilization.
- Tune block size (128 / 256 / 512) for the best result; understand how registers and
  shared memory cap occupancy.
- Deliverable: a short "what I changed and how much each helped" summary at the bottom of
  this README.

---

## Files you'll write from scratch in `src/`

(names are suggestions)

| File | Responsibility |
|------|----------------|
| `renderer.{h,cpp}` | OpenGL: shaders, VAO/VBO, draw call — **hand-written from scratch this phase** |
| `particle_system.{h,cu}` | 1M-particle GPU simulation + interop + SoA |
| `main.cpp` | window + main loop + input + the three-segment timing |

> I'll give the spec, function signatures, and gotchas at the start of each file / level,
> but you type every line yourself.

---

## Progress

- [x] L0 Rendering layer from scratch — hand-built the modern OpenGL pipeline
      (VBO + VAO + vertex/fragment shaders + shader program) and drew a first
      triangle with `glDrawArrays`. The base every later level builds on.
- [ ] L1 Scale to 1M & measure the bottleneck
- [ ] L2 CUDA–OpenGL interop
- [ ] L3 SoA layout
- [ ] L4 Nsight profiling
