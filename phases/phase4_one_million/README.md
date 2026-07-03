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

**Baseline result (1,000,000 particles, scatter + gravity, RTX GPU, uncapped vsync):**

| segment   | avg ms  | what's inside |
| --------- | ------- | ------------- |
| `update`  | ~7.3    | kernel compute **+ `cudaMemcpy` D2H** (36 MB/frame back to host) |
| `pack`    | ~4.0    | CPU loop packs 1M particles into `[x,y,r,g,b]` (writes 20 MB) |
| `upload`  | ~7.7    | 20 MB host→VBO (`glBufferSubData`, CPU→GPU) |
| **total** | **~19** | **~52 FPS** |

The physics itself costs well under 1 ms — nearly the whole frame is the GPU→CPU→GPU
round trip (36 MB down + CPU repack + 20 MB up). This is the wall L2 tears out.

### L2 — CUDA–OpenGL interop: kill the round trip

- Register the OpenGL VBO with CUDA via `cudaGraphicsGLRegisterBuffer`.
- Each frame: `cudaGraphicsMapResources` → get a device pointer → kernel **writes vertices
  directly into the VBO** → `cudaGraphicsUnmapResources`.
- Delete the `host_` mirror, delete the per-frame `cudaMemcpy`, delete `to_vertices()`.
- This is the biggest architectural change in Phase 4 and the key to the frame rate
  recovering.

**Result (1,000,000 particles, same scene as L1, RTX 5070 Ti Laptop, uncapped vsync):**

| build | `update` ms | FPS | note |
| ----- | ----------- | --- | ---- |
| L1 (round trip) | ~19 (update+pack+upload) | ~52 | 36 MB D2H + CPU pack + 20 MB upload every frame |
| **L2 (interop)** | **~0.9, settling to ~0.07** | **~1000+** | kernel writes vertices straight into the VBO; **0 bytes over PCIe** |

~20–250× faster. `update` now measures only the CUDA launch + map/unmap sync — the
physics itself. The remaining per-frame cost has moved off the sim entirely (into
`draw` / buffer swap), which is what L4's Nsight pass will dissect.

> Cleanup still owed (does not affect correctness): `host_` is now only used by the
> constructor for the initial fill, and `to_vertices()` / `Renderer::upload()` are
> dead code. We keep them until L3 rewrites this layer for SoA, then delete in one pass.

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

### L5 — Effects & presets, rebuilt from scratch

Phase 3 handed you the effect system as working code. Here you **re-implement every
visual style yourself** on top of the fast 1M architecture — so you own the whole look,
not just the plumbing. Build these in order, testing each before the next:

1. **Emitters** — a data-driven `Emitter` table in `__constant__` memory; assign each
   particle to one emitter and spawn/recycle from its position / aim / spread / speed /
   color / lifetime.
2. **Staggered lifetimes** — give the initial population *randomised* remaining life so
   particles don't all die on the same frame. Without this the cloud pulses in waves;
   with it you get one continuous stream. (One line, big visual difference.)
3. **Physics forces** — write each force as its own clearly-named term in the update
   kernel and understand what shape it produces:
   - `gravity` (global up/down),
   - `nbodyStrength` (mutual attraction — collapses toward the center),
   - `swirl` (tangential/vortex force around the origin — makes particles *orbit* and
     wind into spiral arms; the key that mutual gravity alone can't produce).
4. **Presets** — bundle an emitter table + those physics knobs into a `Preset` struct,
   hold a table of them, and switch with number keys (re-upload the emitter table via
   `cudaMemcpyToSymbol` on each switch).
5. **Design your own look** — invent at least one preset that is *not* a port of
   fireworks / fire / nebula (e.g. a galaxy, a fountain, rain, an explosion). Pick the
   forces and emitter layout that produce it. This is the "you can now build any style"
   checkpoint.

### L6 — Realistic simulation & randomness

Phase 3's presets look "busy" because they use one shortcut model: a fixed pool of
particles that **instantly respawns at a fixed emitter the moment it dies** — a permanent
fountain, never a real event. Here you replace that with genuine, per-style physical
models and make every style feel alive with randomness. This is where the sim stops
looking procedural and starts looking real.

1. **Episodic *shell* bursts (real fireworks)** — the headline change. Instead of
   continuous emission:
   - Split the particle pool into **shells** (groups that share one explosion).
   - A shell's particles are all born **together, at one random point, at one instant**,
     fly outward, then die together — after which the shell stays **dark for a random
     gap** before re-launching somewhere new. Bursts appear, expand, fade, vanish, pause.
   - Needs either a small per-shell state array (center + countdown timer, advanced by a
     tiny kernel each frame) or a stateless analytic model (each particle computes which
     burst it belongs to from time + shell id). Decide the trade-off yourself.
   - Add **air drag** (`damping`: `v *= damping` each step, ~0.98) so sparks expand fast
     then decelerate and arc down under gravity — the classic firework trajectory.

2. **Per-style realistic models** — give each look its own honest physics, not just
   re-skinned emitters:
   - **Fire** — buoyant rise + turbulence (a curl/noise force that wavers left/right),
     color cooling from white→orange→red→smoke as the particle ages.
   - **Rain / snow** — steady downward field with wind, size/speed variation, splash on
     the floor.
   - **Galaxy** — swirl + weak inward pull tuned so orbits are stable (spiral arms that
     persist instead of collapsing).
   - **Explosion / smoke** — one impulsive radial burst + drag + rising, expanding,
     fading smoke.

3. **Randomness everywhere** — replace the remaining hard-coded, index-derived "fake"
   randomness with real per-particle RNG so no two runs (and no two particles) look
   identical: jittered spawn position, launch angle, speed, lifetime, size, and color;
   randomised burst timing and location. Understand why a *per-particle* `curandState`
   stream is what makes a million particles look organic rather than mechanical.

> Reference only if stuck: Phase 3's `particle_system.cu` has a working version of the
> *continuous* model (emitters, staggered life, gravity/nbody/swirl). L5/L6 go beyond it —
> try each step from the spec first, peek afterward to compare.

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
- [x] Renderer refactored into a reusable `Renderer` class (RAII: init/upload/
      draw + destructor, non-copyable) drawing `GL_POINTS` from interleaved
      `[x,y,r,g,b]` vertices. `main.cpp` wires it up and draws 4 test points to
      verify the pipeline before the particle system exists.
- [x] L1 Scale to 1M & measure the bottleneck — naive GPU→CPU→GPU baseline
      (`particle_system.{h,cu}` + three-segment timing in `main.cpp`). Result:
      update ~7.3 ms / pack ~4.0 ms / upload ~7.7 ms, ~52 FPS. Round trip
      confirmed as the bottleneck; physics itself is <1 ms. Default look for now =
      full-screen scatter + gravity (throwaway scaffolding, replaced by L2's rewrite).
- [x] L2 CUDA–OpenGL interop — kernel writes vertices straight into the VBO
      (`register_vbo` once + per-frame map/get-pointer/launch/unmap in `update`).
      Deleted the per-frame `cudaMemcpy` round trip. Result: `update` ~19 ms → ~0.9 ms
      (settling to ~0.07 ms), ~52 FPS → ~1000+ FPS. `host_`/`to_vertices`/`upload` left
      as dead code, deleted in L3's SoA rewrite.
- [ ] L3 SoA layout
- [ ] L4 Nsight profiling
- [ ] L5 Effects & presets from scratch (emitters, staggered life, gravity/nbody/swirl, your own look)
- [ ] L6 Realistic simulation & randomness (episodic shell bursts, per-style physics, full per-particle RNG)
