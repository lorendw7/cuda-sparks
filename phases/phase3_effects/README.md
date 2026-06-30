# Phase 3 — Effects

**Goal:** Turn the fountain into a force field. Phase 3 adds *effects* — forces that change how
particles move and look — and works up to the headline CUDA optimization of the course: **shared
memory + tiling**.

Phase 3 starts as a copy of the Phase 2 baseline (the curand fountain) and grows level by level.
From this phase on, **you write all of the code** — the kernels, the launches, and the wiring.

---

## The levels

| Level | Effect | New idea |
|-------|--------|----------|
| **1 — Gravity well** ✅ | Particles are pulled toward an attractor point | a force toward a point: direction × magnitude, softening |
| **2 — Naive N-body** ✅ | Every particle attracts every other particle | an `O(n^2)` loop in the kernel; splitting force/integrate kernels to avoid a data race |
| **3 — Shared memory** ✅ | The same N-body, but fast | `__shared__` memory + tiling + `__syncthreads()` |
| **4 — Multiple emitters** ✅ | Several independent fountains, each its own style | `__constant__` memory + a data-driven `Emitter` table |
| **5 — Effect presets** ✅ | Switch fireworks / fire / nebula with keys 1/2/3 | bundling emitters + physics into a `Preset`; re-uploading `__constant__` at runtime; GLFW key input |

---

## Level 1 — Gravity well

A fixed attractor point at `(wellX, wellY)` pulls every particle toward it each frame. The force is
**direction × magnitude**:

- direction = the unit vector from the particle toward the well, `(dx, dy) / dist`
- magnitude = a constant `strength`
- softening: `dist = sqrtf(dx*dx + dy*dy + 0.01f)` — the `+0.01f` prevents divide-by-zero when a
  particle sits exactly on the well.

```cpp
float dx = wellX - particles[i].x;
float dy = wellY - particles[i].y;
float dist = sqrtf(dx*dx + dy*dy + 0.01f);
particles[i].vx += (dx / dist) * strength * dt;
particles[i].vy += (dy / dist) * strength * dt;
```

The well position and strength are passed to the kernel as arguments, so they are easy to change
(and could later follow the mouse).

---

## Level 2 — Naive N-body

Now every particle is pulled by **every other particle**, so each thread loops over all `n`
particles and sums the gravitational pull. Real gravity falls off with the square of distance, so
each contribution is `(dx, dy) / dist^3` — the inverse-square magnitude (`1/dist^2`) times the unit
direction (`(dx, dy)/dist`):

```cpp
for (int j = 0; j < n; j++) {
    float dx = particles[j].x - xi, dy = particles[j].y - yi;
    float dist2 = dx*dx + dy*dy + 0.01f;     // softening
    float inv = rsqrtf(dist2);               // 1/dist  (fast reciprocal sqrt)
    float inv3 = inv * inv * inv;            // 1/dist^3
    ax += dx * inv3;  ay += dy * inv3;
}
```

With `n` threads each doing `n` iterations this is **`O(n^2)`** work per frame — the cost we cut in
Level 3. Bump `kNumParticles` in `main.cpp` and watch the `update` time in the title bar climb
steeply: doubling the particle count roughly **quadruples** the time.

### The data-race trap → two kernels

Thread `i` reads every `particles[j].x`, while thread `j` may be writing `particles[j].x` in the
same launch — a **data race**. The fix is to split the work into **two kernel launches**:

1. **`nbody_force_kernel`** — reads all positions, writes **only velocities**. No thread writes a
   position, so the reads are safe.
2. **`update_kernel`** — now that all velocities are final, integrate positions, bounce, age, and
   recycle.

The boundary between the two launches is a **global synchronization point** — a core CUDA idea.

---

## Level 3 — Shared memory

The naive N-body re-reads all `n` particle positions from slow global memory, for every one of the
`n` threads — `n^2` global reads. **Tiling** keeps the same `O(n^2)` math but changes *where* the
positions are read from: each block cooperatively loads a 256-wide **tile** of positions into fast
on-chip `__shared__` memory **once**, then all 256 threads reuse it. The outer loop marches the tile
across all particles in steps of `blockDim.x`, cutting global reads by roughly the block size.

```cpp
__shared__ float2 tile[256];

for (int t = 0; t < n; t += blockDim.x) {
    int j = t + tid;
    tile[tid] = (j < n) ? make_float2(particles[j].x, particles[j].y)
                        : make_float2(0.0f, 0.0f);
    __syncthreads();                          // tile fully loaded before anyone reads

    int valid = min((int)blockDim.x, n - t);  // real particles in this tile
    for (int k = 0; k < valid; k++) {
        float dx = tile[k].x - xi, dy = tile[k].y - yi;
        float dist2 = dx*dx + dy*dy + 0.01f;
        float inv = rsqrtf(dist2);
        float inv3 = inv * inv * inv;
        ax += dx * inv3;  ay += dy * inv3;
    }
    __syncthreads();                          // everyone done before tile is reused
}
```

Two ideas make or break this kernel:

- **Both `__syncthreads()` are mandatory.** The first stops a thread from reading `tile[]` before
  it is fully loaded; the second stops a fast thread from overwriting `tile[]` for the next step
  while a slow thread is still reading it. Every thread in the block must reach each barrier, so the
  `i < n` guard at the top **returns nothing** — out-of-range threads still loop and sync.
- **The padding trap.** `kNumParticles` (10000) is not a multiple of 256, so the last tile is only
  partly real; the rest is `(0,0)`. Looping the full `blockDim.x` would let those `(0,0)` "ghost"
  particles add a fake pull toward the origin. `valid` bounds the inner loop to the real entries.

This is the payoff of Phase 3: same physics as Level 2, but the per-frame `update` time drops because
the expensive global-memory traffic is replaced by shared-memory reuse.

---

## Level 4 — Multiple emitters

Until now every particle was born at the same hard-coded fountain mouth. Level 4 makes spawning
**data-driven**: a table of `Emitter` recipes (position, aim `angle`, `spread`, `baseSpeed`, color,
`lifetime`) decides where and how each particle is born. Particle `i` belongs to emitter
`i % numEmitters`, so the work spreads evenly across the fountains.

### The new idea — `__constant__` memory

The emitter table is read by **every thread** and **never written** during a kernel — the exact
shape `__constant__` memory is built for. It is a small (64 KB), GPU-cached, read-only region:

```cpp
#define MAX_EMITTERS 8
__constant__ Emitter d_emitters[MAX_EMITTERS];   // a fixed global symbol — NOT cudaMalloc'd
__constant__ int      d_numEmitters;
```

When the 32 threads of a **warp** read the same address, the hardware fetches it once and broadcasts
it to all 32 — so a shared lookup table is essentially free. You fill it once from the host by
**symbol name** (constant memory has no ordinary pointer):

```cpp
cudaMemcpyToSymbol(d_emitters, emitters.data(), numEmitters * sizeof(Emitter));
cudaMemcpyToSymbol(d_numEmitters, &numEmitters, sizeof(int));
```

### Two traps caught along the way

- **The host cannot read `__constant__` memory.** The one-time initial fill runs on the CPU, so
  `emitter_spawn` takes the emitter **by value** from a host-side `std::vector<Emitter>`; only the
  device-side `respawn_rng` reads `d_emitters[e]` directly.
- **Spawn direction is now per-emitter.** The old fountain forced every particle upward
  (`fabsf(sin) * speed + 0.4f`). That hard-coded "up" is gone — direction comes from each emitter's
  `angle` ± `spread`, so a fountain can aim anywhere (needed for Level 5 presets). Want an upward
  "boost"? Raise `baseSpeed` or add a constant *along the aim*, not a hard-coded `vy` push.

Per-particle color now flows through new `cr/cg/cb` fields on `Particle`, copied from the emitter at
birth and faded by remaining life in `to_vertices`.

### Tuning the look

A clump that won't spread is usually the **mutual gravity** (`nbodyStrength` in the
`nbody_force_tiled` launch) pulling the fountains back together — set it to `0.0f` for clean separate
fountains. To make emitters feel independent: push their `x` apart, aim them apart (`angle`), and
narrow `spread`.

---

## Level 5 — Effect presets

A **preset** bundles a whole "look" into one struct: an emitter table *plus* the physics that goes
with it (`gravity`, `nbodyStrength`). Three presets — **fireworks / fire / nebula** — live in a
static table, and number keys `1` / `2` / `3` switch between them at runtime.

```cpp
struct Preset {
    Emitter emitters[MAX_EMITTERS];  // this look's spawn sources (fixed-size slots)
    int     numEmitters;             // how many slots are used
    float   gravity;                 // physics for this look ...
    float   nbodyStrength;           // ... bundled in with the emitters
};
```

### The key idea — switching = re-uploading `__constant__`

The device already reads spawn settings from `__constant__ d_emitters` when it recycles particles
(Level 4). So switching presets is just **uploading a different table** with the same
`cudaMemcpyToSymbol` calls, plus copying the preset's physics into the live `params_`:

```cpp
void ParticleSystem::set_preset(int i) {
    const Preset& ps = presets[i];                 // presets[] lives on the HOST
    params_.gravity       = ps.gravity;            // CPU -> CPU
    params_.nbodyStrength = ps.nbodyStrength;
    cudaMemcpyToSymbol(d_emitters, ps.emitters, ps.numEmitters * sizeof(Emitter));  // CPU -> GPU
    cudaMemcpyToSymbol(d_numEmitters, &ps.numEmitters, sizeof(int));
}
```

Existing particles keep their old look and only adopt the new preset as they die and respawn, so a
switch **cross-fades** instead of snapping. The constructor just calls `set_preset(0)` to boot.

### Two things worth remembering

- **Host vs. device, again.** `presets[]` and `params_` live in CPU memory; `d_emitters` /
  `d_particles_` live on the GPU. The CPU never reads `d_emitters` directly — even the initial fill
  reads the host-side `presets[0]`. Crossing the boundary always means a `cudaMemcpy*`.
- **Polled input fires every frame.** `glfwGetKey` returns `GLFW_PRESS` for as long as the key is
  held, so holding `2` calls `set_preset(1)` many times — harmless, because re-selecting the same
  preset just re-uploads the same table. Edge-triggered "next preset" cycling would need to remember
  the previous frame's key state.

---

## Build & run

```bash
.\build.bat run3
```

It boots on the **fireworks** preset. Press `1` / `2` / `3` to switch between fireworks, fire, and
nebula. Tune the look by editing:

- the `presets[]` table in `particle_system.cu` (emitter rows + per-preset `gravity` /
  `nbodyStrength`),
- `kNumParticles` in `main.cpp` (raise it to feel the `O(n^2)` slowdown),
- the color line in `to_vertices` (per-particle `cr/cg/cb` faded by lifetime `t`).
