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
| **3 — Shared memory** ▶ | The same N-body, but fast | `__shared__` memory + tiling + `__syncthreads()` |

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

## Level 3 — Shared memory (next)

The naive N-body re-reads all `n` particle positions from slow global memory, for every one of the
`n` threads — `n^2` global reads. **Tiling** has each block cooperatively load a small tile of
positions into fast on-chip `__shared__` memory, which all 256 threads in the block reuse, cutting
global reads by roughly the block size. `__syncthreads()` coordinates the load. This is the payoff
of Phase 3.

---

## Build & run

```bash
.\build.bat run3
```

You should see a glowing cyan-to-magenta cloud of particles clumping under their mutual gravity.
Tune the look by editing:

- `strength` in the `update()` launch (N-body force; start around `0.0001f`),
- `params.gravity` in `main.cpp` (set to `0.0f` to drop the downward pull and let N-body dominate),
- `kNumParticles` in `main.cpp` (raise it to feel the `O(n^2)` slowdown),
- the color line in `to_vertices` (interpolated by lifetime `t`).
