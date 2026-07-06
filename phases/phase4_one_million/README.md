# Phase 4 — One Million Particles

> **Goal:** push the particle count from ~10,000 to **1,000,000** and keep it smooth.
> In this phase you write **100% of the code from scratch** (rendering layer included);
> I only provide the spec, function signatures, and gotchas line by line.
>
> **Documents.** This README is the CUDA / performance spine (levels **L1–L7**). Two
> further tracks that turn the finished sim into a polished demo live in their own files,
> layered onto the **same** `src/` codebase (no forked folders):
> **[PRESENTATION.md](PRESENTATION.md)** (menu / fullscreen / auto-play) and
> **[AUDIO.md](AUDIO.md)** (procedural, zero-license sound). See
> *[Beyond the CUDA levels](#beyond-the-cuda-levels--application-tracks)* below.

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

> Cleanup status: L3 deleted `host_` and `to_vertices()`; the now-dead
> `Renderer::upload()` (unused once the L2 interop path wrote the VBO directly) was
> **deleted** in the cleanup pass, along with its `#include <vector>`. The renderer's
> only data path is now interop.

### L3 — Structure of Arrays (SoA)

- Split the array-of-structs `struct Particle{ x,y,vx,vy,... }` (AoS) into **one array per
  field** (SoA): `float* x; float* y; float* vx; ...`.
- Why: when the 32 threads of a warp read `x[i]` together, SoA makes those addresses
  contiguous → **coalesced memory access**; AoS strides across memory and wastes bandwidth.
- Use Nsight to compare L2 vs L3 memory throughput / frame rate and quantify the gain.

**Result (1,000,000 particles, RTX 5070 Ti Laptop, uncapped):** wall-clock is
**unchanged from L2** — ~0.08 ms steady-state `update`, ~1040 FPS. This is the
expected outcome, not a failure: `update_kernel` reads almost every field
(x,y,vx,vy,life,cr,cg,cb), so AoS wasn't wasting much of each 36-byte fetch —
there was little coalescing headroom for SoA to reclaim. SoA pays off most when a
kernel touches *few* fields; here it doesn't. Whether coalescing efficiency
actually improved (even without a frame-time win) is what L4's Nsight pass will
measure. Also cleaned up in this pass: deleted `host_`, `to_vertices()`.

### L4 — Nsight profiling deep-dive

- Inspect occupancy, memory throughput, warp utilization.
- Tune block size (128 / 256 / 512) for the best result; understand how registers and
  shared memory cap occupancy.
- Deliverable: a short "what I changed and how much each helped" summary at the bottom of
  this README.

**Result — Nsight Compute on `update_kernel` (1M particles, block=256, RTX 5070 Ti Laptop):**

Profiled with `ncu --replay-mode application` (kernel replay can't save/restore the
CUDA-GL interop VBO). Added a `SPARKS_MAX_FRAMES` env-var escape hatch so the app
terminates for application replay. Kernel duration ~164 µs.

| Section | Metric | Value | Reading |
| ------- | ------ | ----- | ------- |
| Speed of Light | Memory Throughput | **93.9%** | memory-bound |
| Speed of Light | Compute (SM) | **14.2%** | compute nearly idle |
| Speed of Light | L2 Cache Throughput | 93.9% | **L2 is the actual limiter** (DRAM only 62.9%) |
| Memory | Max Bandwidth | 93.9% (326 GB/s) | memory subsystem ~saturated |
| Memory | L1 / L2 Hit Rate | 69% / 64% | caches absorb most traffic → L2-bound, not DRAM |
| Occupancy | Achieved | **89.3%** (42.8/48 warps) | excellent; not the bottleneck |

**Conclusion:** the kernel is memory-bandwidth-bound (L2) with compute idle and
occupancy already near-max. This explains L3's null result: the wall is *total
bytes moved*, and SoA only rearranges bytes — it doesn't move fewer when the kernel
reads every field. Real speedups here would come from moving **less** data (fewer
fields, smaller/half-precision storage), not from layout or block-size tuning.
Block-size sweep (128/256/512) is expected to be marginal since we're bandwidth-
bound, not occupancy-bound.

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

**Progress:** Steps 1–5 DONE. The `Emitter` table lives in `__constant__ d_emitters`,
uploaded via `upload_emitter()` / `cudaMemcpyToSymbol` (added `SimParams.numEmitters`).
`spawn()` births particle `i` from emitter `i % numEmitters`: polar `(angle ± spread,
baseSpeed)` → Cartesian velocity, emitter position/colour, staggered lifetime so the
stream is continuous, not pulsed. `update_kernel` sums four named forces —
`gravity` / central attractor (`nbodyStrength`) / `swirl` / `wind` — into an `(ax, ay)`
accumulator, then integrates with semi-implicit Euler. A `Preset` table (Jia /
fireworks / fire / galaxy / rain) bundles an emitter table with those physics knobs;
`set_preset(i)` clamps, re-uploads the emitters, and copies the knobs into `params_`. The
constructor boots preset 1 (fireworks) and hotkeys **J** (Jia) / **1 / 2 / 3 / 4**
(fireworks / fire / galaxy / rain) switch live — `GLFW_KEY_*` is the physical key, so J is
case-insensitive (the look fades in over ~1 lifetime as particles recycle). **Step 5 DONE** — two original presets designed from
scratch: **galaxy** (an orange nucleus parked at the origin, where the swirl/nbody forces
vanish so it stays tight, plus two offset blue arms the vortex winds into spirals) and
**Jia** (two point-symmetric pink + gold jets braided by a gentle swirl). Default particle
count is **30k** (sparse enough to stay crisp; overridable at runtime via `SPARKS_PARTICLES`).

**Real randomness (pulled forward from L6):** the index-derived `frac(i * golden)`
jitter formed visible *ripples* at low counts (a low-discrepancy sequence shows
structure), so `spawn()` now draws from a **per-particle `curandState`**: a `d_rng_`
array seeded once by an `init_rng` kernel (`curand_init(seed, i, 0, …)` — subsequence
`i` decorrelates each stream), then `curand_uniform()` for angle/speed/lifetime. Every
spawn *and* rebirth draws fresh values, so no repeating paths. (Cheap here — `curandState`
is ~48 B/particle, only a bandwidth concern at 1M; at 10k–30k it's free.)

**Look tuning:** presets ported to Phase 3's values (spread-out fast fireworks, narrow
fire jets, two-source swirl nebula). Renderer matched to Phase 3 (`gl_PointSize 12`,
alpha blend, soft round dot). **Key fix:** the sim used a *fixed* `dt = 0.016` while the
frame rate was uncapped (thousands of FPS) → physics ran ~50× too fast (a chaotic churn
that read as "blur"). Now `dt` is the **real elapsed frame time** (clamped to 0.05 s), so
the sim is frame-rate-independent and runs at real-time speed.

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
   > **Bandwidth watch (ties back to L4):** the RNG *state* is the biggest byte-mover
   > you'll add. `curandStateXORWOW` is ~48 B/particle — bigger than all 8 current
   > fields (32 B) combined — and the kernel reads+writes it every frame. Storing 1M of
   > them roughly *triples* per-frame traffic. This choice (small state like Philox, or a
   > *stateless* counter-based hash from `(id, frame)` that stores nothing) is a far
   > larger bandwidth lever than any FP16 packing in L7. Decide it deliberately.

> Reference only if stuck: Phase 3's `particle_system.cu` has a working version of the
> *continuous* model (emitters, staggered life, gravity/nbody/swirl). L5/L6 go beyond it —
> try each step from the spec first, peek afterward to compare.

**Progress:**

- [x] **Air drag** (groundwork for shell bursts, helps every preset). Wired the unused
  `SimParams.damping` into `update_kernel`: after integrating velocity, `v *= drag` where
  `drag = powf(damping, dt * 60.0f)` — frame-rate-independent (`damping` is defined per
  60 fps frame; the `dt*60` exponent rescales it to the real frame length, same fix as
  `dt` itself). Added `damping` to the `Preset` struct so each look tunes its own drag
  (fireworks 0.97 strong / fire 0.99 / galaxy 1.0 frictionless / Jia ~0.9996).
- [x] Episodic shell bursts (real fireworks) — the headline change; replaces the L5
  continuous fountain. Implemented as approach A (per-shell state array). A `Shell[numShells]`
  array + per-shell RNG; **2a** seeds every shell dark with a staggered countdown; **2b**
  `advance_shells` (one thread/shell, run *before* `update_kernel`) runs each shell's
  state machine — `live`→`dark` on expiry (random gap), `dark`→relaunch (new random
  center + palette color + `launch` pulse); **2c** `update_kernel` branches on a per-preset
  `useShells` flag: on `launch` all of a shell's particles `spawn_burst` together at the
  center (random full-circle direction, shell color, life = burst time), and the vertex is
  written off-screen while the shell is `dark` (born-together / die-together / dark gap).
  Only `fireworks` sets `useShells`; other presets keep the continuous model. Air drag
  (above) + gravity give the fly-out-then-arc trajectory.
- [ ] Per-style realistic models (fire turbulence + color cooling, rain, galaxy stable
  orbits, explosion/smoke).
- [ ] Randomness everywhere (already partly done — L5.5 moved spawn to per-particle
  `curandState`; extend to burst timing/location, size, color-over-life).

**Shell design decisions (L6):**

- **Scale target 620k particles**; `numShells` will be **auto = `n / 2500`** (≈2500
  particles per burst) rather than hardcoded, so shell count scales with `n`
  (30k→~12, 620k→~250). Currently `numShells = 16` in `main.cpp` for the 30k default.
- **Shells are for *episodic* styles only** (fireworks, future explosion) — **not**
  continuous ones. Fire / rain / galaxy get their *own* randomness instead (fire:
  turbulence + color cooling; rain: wind/size jitter + splash; galaxy: orbit jitter).
- **Timing = independent staggered cycling** is the default (continuous show). A
  synchronized-batch "finale/volley" (all launch → all die → full dark → repeat) is an
  optional future toggle, not default (avoids a full-screen pulse).
- **Preset switching stays fade-in** (old particles finish their life as the new look
  fades in). No hard reset-on-switch.
- **Anti-chaos at high counts = structure, not fewer particles:** grouping (shells),
  one color per burst, negative space (only ~10–30% of shells live at once via the
  live/dark ratio), smaller points (2–5px) + fade-by-life, and revisiting additive
  blend (the right tool at high density, where overlap should read as a brightness
  gradient — it only looked hazy at *low* count).

> **Known limitation (L5 model, fixed by shell bursts above):** `fireworks` still emits
> *continuously* from 3 fixed points, so sparks "spread in place" and fall near the
> emitter rather than launching as a discrete, born-together burst. This is the exact
> artifact the shell-burst step removes.
>
> **Input — key debounce (DONE):** the hotkey handling was table-driven (a `kBinds`
> array of `{key, preset}` rows, looped over so the logic is written once) and
> **edge-detected**: each key's previous-frame state is kept in `prevState[]`, and
> `set_preset` fires only on the `RELEASE -> PRESS` transition (`now == PRESS &&
> prev == RELEASE`). One physical press = one switch + one `cudaMemcpyToSymbol`, instead
> of thousands per second while a key is held.

**Visual polish — firework shapes, color & effects (planned, fireworks branch of L6):**

The whole point after "it bursts" is to make it *beautiful and varied*. Three tracks,
all small additions on top of the working shell system:

1. **Burst shapes** — the shape is set *entirely* by `spawn_burst`'s initial-velocity
   distribution (today: random full-circle direction + random speed `0..0.8` = a filled
   disk). Add an `int type` to `Shell`, picked at random on relaunch (like the color),
   and branch `spawn_burst` on it. Menu by difficulty:
   - ⭐ **ring / sphere** — fixed speed (drop the `0..max` randomness) → all sparks share
     one radius = a hollow expanding ring.
   - ⭐⭐ **star / petals** — speed modulated by angle, `s = base*(1 + 0.3*cos(5*a))` → spikes.
   - ⭐⭐ **willow** — low speed + strong gravity + long life → rise then droop.
   - ⭐⭐ **double-ring / layered** — split by `i` into inner-slow / outer-fast groups, a
     different color per layer.
   - ⭐⭐⭐ **heart / letter / arbitrary pattern** — use `i` as a parameter into a shape
     equation and aim the velocity along it.
   - ⭐⭐⭐ **rising tail** — add a "rising" shell state (rocket climbs + trails) that only
     explodes at apex.
2. **Color** — beyond one palette hue per burst: **color-over-life** (add an end-color,
   lerp birth→ember by life fraction: white→hue→dark red = the classic cooling spark;
   shared with the Phase-A2 optimization idea), per-layer colors, brighter/HDR palette
   for additive blend.
3. **Visual-effect optimization** — point **size-by-life** (twinkle: big-bright when young,
   small-dim when dying), per-particle size jitter, a brightness-over-life curve, glow,
   and the additive-blend-at-high-density revisit (see anti-chaos above). Preset struct
   grows a few fields (`colorEnd`, size params); small refactor.

Recommended first shape lesson: **hollow ring + per-burst random `Shell.type`** — tiny
change, big visual gain, and it teaches "one field drives many behaviors."

### L7 — Precision & bandwidth *(optional / stretch)*

Turns L4's *diagnosis* ("the wall is total bytes moved") into a *measured result*. Do
this only after L6, once the field set is final (L6's RNG state is the dominant term —
see the bandwidth watch above). Not required for a working, good-looking sim; its value
is pedagogical (learn `__half` / `__half2`) and closing L4's scientific loop.

- Move perceptually-tolerant fields (`cr,cg,cb`, `life`) to `__half` — ~25% less traffic,
  zero visible precision loss.
- Optionally pack `(x,y)` and `(vx,vy)` as `__half2`, **computing in FP32 registers** and
  storing FP16 — FP16 *integration* stalls (a slow particle's `vx*dt` is below the FP16
  ULP near 1.0 and gets swallowed), so never integrate in half.
- Re-run the L4 Nsight pass and check the hypothesis: halving bytes ≈ halving the ~164 µs.

---

## Beyond the CUDA levels — application tracks

L1–L7 above are the CUDA / performance spine of Phase 4. Two further tracks turn the
finished simulation into a polished, shippable demo. They are **layered onto the same
`phase4_one_million/src` codebase** (no forked folders — a menu or a sound engine does not
justify re-copying the whole 1M sim) but documented in their own files so the CUDA story
here stays clean. Build them in this order:

1. **[Presentation & UX](PRESENTATION.md)** — an on-screen **Dear ImGui** menu, a real
   **fullscreen** mode (square sim area on the shorter screen edge + a hacker-style
   telemetry HUD in the leftover strip), and a hands-off **auto-play** loop. Pure
   C++/GLFW/OpenGL app-shell work — **no kernel changes**. Do it **after L6** (browse
   *good-looking* presets, not ugly ones).
2. **[Audio](AUDIO.md)** — **procedurally generated, zero-license** sound: event SFX →
   per-preset ambient beds → an **audio-reactive** layer driven by a scalar sampled from
   the live GPU sim. Do it **after Presentation** — it plugs straight into the menu
   (volume / mute) and auto-play (advance on the beat).

*(L7 is an optional stretch and can be done any time after L6; it does not block these
tracks.)*

---

## Files you'll write from scratch in `src/`

(names are suggestions)

| File | Responsibility |
|------|----------------|
| `renderer.{h,cpp}` | OpenGL: shaders, VAO/VBO, draw call — **hand-written from scratch this phase** |
| `particle_system.{h,cu}` | 1M-particle GPU simulation + interop + SoA |
| `main.cpp` | window + main loop + input + the three-segment timing |

> The Presentation and Audio tracks add their own files to this same `src/` (e.g. a
> `hud.*` / `audio.*`); see their docs. I'll give the spec, function signatures, and
> gotchas at the start of each file / level, but you type every line yourself.

---

## Progress

- [x] L0 Rendering layer from scratch — hand-built the modern OpenGL pipeline
      (VBO + VAO + vertex/fragment shaders + shader program) and drew a first
      triangle with `glDrawArrays`. The base every later level builds on.
- [x] Renderer refactored into a reusable `Renderer` class (RAII: init/upload/
      draw + destructor, non-copyable) drawing `GL_POINTS` from interleaved
      `[x,y,r,g,b]` vertices. At that milestone `main.cpp` drew 4 test points to
      verify the pipeline before the particle system existed (it now runs the full
      1M draw loop).
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
- [x] L3 SoA layout — split AoS `Particle*` into 8 `float*` field arrays
      (`ParticleSoA`); initial fill moved to an on-GPU `init_kernel` (no host
      mirror, no upload). Wall-clock unchanged from L2 (~0.08 ms, ~1040 FPS) —
      expected, since the kernel reads nearly every field so AoS had little waste
      for coalescing to reclaim. Deleted `host_` / `to_vertices()`. Nsight (L4)
      will measure whether coalescing efficiency improved regardless.
- [x] L4 Nsight profiling — `ncu` (application replay; added `SPARKS_MAX_FRAMES`
      escape hatch) on `update_kernel`: Memory 93.9% vs Compute 14.2% → memory-bound,
      specifically L2-limited (DRAM 62.9%); Achieved Occupancy 89.3% (not the
      bottleneck). Confirms L3's null result — the wall is total bytes moved, so SoA
      (same bytes, rearranged) can't help; only moving *less* data would.
- [x] L5 Effects & presets from scratch — **DONE** (emitters in
      `__constant__` + `cudaMemcpyToSymbol` upload; `spawn` births from emitter with
      polar→Cartesian launch + staggered life; three named forces —
      gravity / central attractor / swirl — in an `(ax,ay)` accumulator + semi-implicit
      Euler; `Preset` table + `set_preset` + number-key **1/2/3/4** switching, ctor boots
      preset 0). Step 5 — two original presets: **galaxy** (origin nucleus + two
      swirl-wound arms) and **Jia** (symmetric pink/gold braided jets). Default count 30k.
- [ ] L6 Realistic simulation & randomness (episodic shell bursts, per-style physics, full per-particle RNG)
  - [x] L6 **rain** preset (key **4**) — a data-driven `useRain` mode. `spawn_rain` births
        drops at a random x across the top; a per-drop `depth` fakes parallax (near drops fall
        faster **and** brighter, far ones slower/dimmer — both derived from one RNG draw). Drops
        **land and linger as a puddle**: a large `life` sentinel doubles as a "still falling" flag
        vs. a short puddle-dwell countdown, so the 3-state cycle (fall → puddle → respawn) needs
        no extra per-particle field. A `wind` knob adds slant (a constant horizontal accel; 0 for
        every non-rain preset, so it needs no branch). Hue reads from the one rain emitter row —
        data-driven, so the sea-water color is just data.
  - [x] L6 **smoke** preset (key **5**) — a buoyant, turbulent plume from one narrow bottom
        vent. Adds a `turbulence` knob (last field of `Preset`/`SimParams`, so the other five
        presets zero-fill it automatically). A new force in `update_kernel` gives each particle
        a fresh **random horizontal kick** every frame; integrated into `vx` it makes the sideways
        velocity a **random walk / Brownian motion**, so the rising column widens and curls into a
        diffusing plume. Buoyancy (negative gravity) drives the rise; light drag sets a terminal
        velocity that, over the 8 s life, carries the plume to the **top wall — where the non-rain
        bounce path makes it pool under a "ceiling"** (a kept happy accident). The kick is scaled by
        **`/ sqrtf(dt)`** so the diffusion rate is **frame-rate-independent**: a random walk's spread
        grows as √(steps), and over a fixed second there are `1/dt` steps of size ~`dt`, so a constant
        coefficient would give spread ~√dt (less diffusion at higher FPS) — the same class of bug the
        `dt` clamp and `powf(damping, dt*60)` drag already fixed elsewhere.
  - [x] L6 **curl-noise flow field** preset (key **6**) — a data-driven `useFlow` mode; the
        coherent-field upgrade of smoke's white noise. A scalar potential `psi(x,y,t)` (a couple
        of drifting sine octaves) is turned into a **divergence-free flow vector** by taking its
        2D **curl** (`fx = ∂ψ/∂y`, `fy = −∂ψ/∂x`, estimated by central finite differences) — added
        as force #6, gated by a `curl` knob (0 for every other preset, so no branch). A new
        `time` field in `SimParams`, accumulated per frame in `update()`, drifts the field so the
        eddies never repeat. `spawn_scatter` seeds particles across the **whole screen** (a
        per-position field is invisible from a point emitter), and strong damping (0.75) makes
        velocity track the field (near-advection). **Key insight:** because the field is
        divergence-free, density stays uniform → the swirls are invisible from dot positions;
        they are made visible by **coloring each particle by its velocity direction**
        (`atan2f(vy,vx)` → three sines 120° apart = a seamless rainbow hue wheel), so same-eddy
        particles share a hue and each eddy reads as a rotating color patch. Tuned to `curl 1.2`,
        `damping 0.75`, `F 6.12`.
  - [x] L6 **strange attractor** preset (key **7**) — a data-driven `useAttractor` mode, the first
        **velocity-field** (not force) style: `dp/dt = f(p)`, so velocity is read straight from the
        Lorenz ODE and position is integrated directly (no acceleration, no stored `v`). Needs a 3rd
        coordinate `z` (added to `ParticleSoA`) because a 2D autonomous flow can't be chaotic
        (Poincaré–Bendixson); the butterfly is drawn by projecting `(x, z)` to the screen.
        `lorenz()` velocity field (σ=10, ρ=28, β=8/3) + `spawn_attractor()` seeds the Lorenz state
        space (x,y ±20, z 0..50). The `update_kernel` `useAttractor` branch runs **first and early-
        returns**, keeping the raw ±20..50 state clear of the force-model pipeline (whose wall-clamp
        would pin it into a 1×1 box and destroy the butterfly). It integrates with forward Euler
        **substepped** for stability — an **adaptive** step count (`sub = ceil(dt·timeScale / 0.006)`)
        holds every step ≤ 0.006 at any FPS, so a long frame can't extrapolate the fast Lorenz
        velocity into a NaN blow-up — then reseeds dead particles via `spawn_attractor` (refreshing
        the register copies so the vertex is drawn from the *new* seed) and projects `(x, z)` to the
        screen (x already centered → scale by 1/25; z spans 0..50 → shift −25, then the same 1/25 so
        the aspect ratio stays true). Key **7** is bound in `main.cpp` (the `kBinds` table now
        auto-sizes via `nBinds = sizeof(kBinds)/sizeof(kBinds[0])`, so adding a preset is one row).
        Coloring by speed/z + trail persistence are deferred to the optional per-style polish below.
- [ ] L7 Precision & bandwidth *(optional)* — FP16/`__half2` on tolerant fields, re-run L4 Nsight to confirm "fewer bytes ≈ less time"

### Application tracks (layered on the finished sim — separate docs)

- [ ] Presentation & UX — see **[PRESENTATION.md](PRESENTATION.md)** (auto-play +
      manual toggle, fullscreen square viewport + telemetry HUD, Dear ImGui menu). After L6.
- [ ] Audio — see **[AUDIO.md](AUDIO.md)** (procedural event SFX → ambient beds →
      audio-reactive). After Presentation. (Was "Phase 5" in the roadmap.)

### Per-style advanced polish (OPTIONAL — LAST, after Presentation & Audio)

The looks are "good enough" once L6 is done; these are the *final, optional* per-style
refinements, deliberately parked until the app shell (Presentation) and Audio tracks ship.
Several depend on one enabling renderer feature, so build that first if you pick them up:

- [ ] **Shared renderer upgrades** (unlock several items below):
  - Per-vertex point **size** — add a `size` attribute to the VBO + `gl_PointSize = size`
    in the vertex shader. Unlocks rain near-big/far-small, twinkle, size-by-life.
  - **Streak / motion-blur** rendering — draw fast particles as short lines / trails.
  - **Additive blend** revisited at high density (was reverted at low count for haze).
- [ ] **Fireworks** — burst **SHAPES** via `spawn_burst`'s velocity distribution + a random
  `Shell.type`: hollow ring/sphere (fixed speed), star/petals (`s=base*(1+k*cos(mθ))`),
  willow/垂柳 (low speed + gravity droop + long life), double-ring (split `i` into layers),
  heart/letter (`i` → shape equation), rising-tail rocket (climb, explode at apex). Plus
  secondary "cracker" bursts (a dying spark spawns children) and color-over-life (bright→ember).
- [ ] **Fire** — curl-noise turbulence (smooth field, not white noise), color cooling
  white→orange→red→smoke over life, flicker, smoke hand-off at the top.
- [ ] **Galaxy** — orbit radius/phase jitter, color by radius (hot core / cool arms),
  disk confinement, dust lanes.
- [ ] **Rain** — **near-big/far-small** (point size by `depth`, needs per-vertex size),
  velocity **streaks**, puddle-dwell jitter + landing ripples, gusting (time-varying) wind,
  splash-up droplets on landing, depth fog.
- [ ] **Smoke** — denser curl-noise turbulence, grow+fade curve tuning, buoyancy variation,
  light/dark shading by local density.
- [ ] **Curl-Noise flow field** — color by velocity direction/magnitude, multi-octave noise,
  faster field evolution.
- [ ] **Strange Attractor** — color by speed/position, trail persistence, multiple attractors.
