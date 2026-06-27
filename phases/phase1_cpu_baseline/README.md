# Phase 1 — CPU Baseline

**Goal:** Run **10,000 particles** entirely on the CPU, draw them with OpenGL, and record how
fast it goes. This is your *reference benchmark* — every later phase is measured against the numbers
you see here.

---

## 1. The big picture

A particle system is just a big array of tiny objects, each with a position, a velocity, and a
lifetime. Every frame you do the same three things:

```
   for each particle:
       move it a little
       age it a little
       if it died, restart it
```

That `for each particle` loop is the heart of everything. Notice that **each particle is completely
independent** — particle #5 never needs to look at particle #6. Hold onto that idea: it is exactly
*why* a GPU can later run all 10,000 (or 1,000,000) of them at the same time.

---

## 2. What's already built for you

| File | What it does | Edit it? |
|------|--------------|----------|
| `src/main.cpp` | Opens the window, runs the loop, measures FPS + update time | ❌ no |
| `src/renderer.cpp/.h` | Draws particles as colored OpenGL points | ❌ no |
| `src/particle_system.h` | The simulation. `reset()`, `respawn()`, `to_vertices()` are done | ✅ **yes — one function** |
| `include/particles.h` | The `Particle` struct and `SimParams` | ❌ no (read it) |

You only write **one function**: `ParticleSystem::update(float dt)`.

---

## 3. Your task

Open [`src/particle_system.h`](src/particle_system.h) and find the `── TODO ──` block inside
`update(float dt)`. Write a loop over all `params_.n` particles. For each particle `i`:

| Step | Code shape |
|------|------------|
| 1. Gravity | `particles_[i].vy -= params_.gravity * dt;` |
| 2. Move | `particles_[i].x += particles_[i].vx * dt;` (and `y`) |
| 3. Age | `particles_[i].life -= dt;` |
| 4. Recycle | `if (particles_[i].life <= 0.0f) respawn(i);` |

That's the entire physics for Phase 1. Keep each iteration self-contained.

`dt` (delta-time) is the real number of seconds since the last frame, measured by `main.cpp`. Using
it instead of a fixed step means the animation runs at the same *speed* whether your machine does
60 or 6000 FPS.

---

## 4. Build & run

From the project root:

```bash
.\build.bat run1
```

`build.bat` configures (with the Ninja generator), builds every phase, and runs Phase 1. To build
without running, just `.\build.bat`.

> The first build automatically downloads and compiles GLFW, so it needs an internet connection —
> be patient the first time. After that it's quick.

**What you should see:** a dark window with a fountain of warm-colored points arcing up from the
bottom center and falling under gravity, looping forever. The title bar shows
`10000 particles | <FPS> | update <ms>`.

---

## 5. Record your baseline

Write these down — you'll compare against them in Phase 2:

- **FPS** at 10,000 particles: ________
- **`update` time** (ms per frame): ________

Curious? Bump `kNumParticles` in `main.cpp` to `100000`, rebuild, and watch the `update` time climb.
The CPU does every particle one after another, so the cost grows linearly. That wall is what the GPU
is about to smash through.

---

## 6. Checklist

- [ ] The fountain animates smoothly and loops (particles respawn, never disappear forever).
- [ ] The title bar shows a stable FPS and an `update` time.
- [ ] You recorded your baseline FPS and update-time numbers.

When all three are done, you've cleared **Level 1**. Before Phase 2, try **Level 2** below — more
code to write, more C++ to learn.

---

## Level 2 — Bounce off the walls

Right now particles that fly off the edge just keep going until they die. Let's make them **bounce**
off the four walls instead, like a ball in a box. This is more practice with `if` and a new C++
idea: the **unary minus** (flipping a sign).

### The physics

A wall sits at each edge: the world spans `-bound … +bound` on both x and y (`bound` is in
`SimParams`, value `1.0`). When a particle crosses a wall, two things happen:

1. **Flip the velocity** in that direction — moving right becomes moving left. In C++ you flip a
   sign with a minus in front: `vx = -vx;`  (this is the *unary minus* — it negates one value).
2. **Lose a little energy** — multiply by `restitution` (0.8) so each bounce is a bit weaker:
   `vx = -vx * params_.restitution;`

### New C++ concept: unary minus

```cpp
float a = 5.0f;
float b = -a;     // b is now -5.0  —— a single minus in front means "negate"
```

You've seen `-` as subtraction (`a - b`). Here it has **one** operand and just flips the sign. A
velocity of `+0.3` (moving right) becomes `-0.3` (moving left) — a bounce.

### Your task

In `update()`, **after** the move step (after you update `x` and `y`), add four `if` checks — one
per wall. Here's the **right wall** done for you as the model:

```cpp
// hit the RIGHT wall? (x went past +bound)
if (particles_[i].x > params_.bound) {
    particles_[i].x  = params_.bound;                          // put it back on the wall
    particles_[i].vx = -particles_[i].vx * params_.restitution; // bounce: flip + lose energy
}
```

Now you write the other three, following the same shape:

| Wall | Condition | Which velocity flips |
|------|------|------|
| Right (done) | `x > bound` | `vx` |
| Left | `x < -bound` | `vx` |
| Top | `y > bound` | `vy` |
| Bottom | `y < -bound` | `vy` |

> Hint: the left-wall condition is `x < -params_.bound` (note the unary minus shows up again in
> `-params_.bound`!). The top and bottom walls change `y` and `vy` — the logic is exactly the same
> as left and right.

Build, run, and watch particles ping around inside the box, each bounce a little gentler until they
settle. When that works, you've cleared Level 2.

---

## Level 3 — OpenGL shaders

So far you only touched the CPU (the physics). Level 3 is your first **OpenGL** lesson: you edit the
two tiny GPU programs — *shaders* — that decide how each particle is drawn. They live at the top of
[`src/renderer.cpp`](src/renderer.cpp).

### The render pipeline

The CPU hands the GPU a list of points (position + color). For every point the GPU runs two shaders:

```
your points  ->  ① Vertex Shader        ->  ② Fragment Shader       ->  screen
(x,y,color)      "where on screen?"          "what color is each pixel?"
```

| Concept | In `renderer.cpp` | Plain words |
|---------|-------------------|----------------------|
| **Shader** | `kVertexShader` / `kFragmentShader` strings | GPU programs, written in GLSL (C-like) |
| **VBO** | `glBufferData` | a shelf in GPU memory holding all the points |
| **VAO** | `glVertexAttribPointer` | the layout: 5 floats per point = 2 pos + 3 color |
| **Draw call** | `glDrawArrays(GL_POINTS, ...)` | "draw them all" |

### What you build

Turn the flat square points into **round, glowing dots**. Three coordinated edits:

1. **Vertex shader** — make points bigger so the shape is visible: `gl_PointSize = 12.0;`
2. **Fragment shader** — clip the square into a circle and fade the glow toward the edge:
   ```glsl
   void main() {
       vec2  d    = gl_PointCoord - vec2(0.5);   // offset from the point's center
       float dist = length(d);                    // 0 = center, 0.5 = edge
       if (dist > 0.5) discard;                    // drop pixels outside the circle -> round
       float glow = 1.0 - dist * 2.0;              // 1 at center, 0 at the edge
       FragColor  = vec4(vColor * glow, glow);     // fade color and alpha outward
   }
   ```
3. **Enable blending** (C++ side, in `init()`) so the soft transparent edges blend with the
   background instead of overwriting it:
   ```cpp
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   ```

### New ideas you met

- **GLSL `vec2/vec3/vec4`** — built-in bundles of 2/3/4 floats (a struct you already understand).
- **`gl_PointCoord`** — where inside the point the current pixel is (center = 0.5, 0.5).
- **`discard`** — "don't draw this pixel" (this is what carves the square into a circle).
- **Blending** — every GL call must happen *while the context is alive*: between `init()` and
  cleanup, **never after** `glfwTerminate()`. A GL call after teardown silently does nothing.

Clear Level 3 and you've touched all three layers — **C++**, the **render pipeline**, and (next)
**CUDA**. You're ready for **Phase 2 — CUDA Migration**, where the Level-1 loop moves onto the GPU.

---

## Hints

<details>
<summary>I get a black/empty window.</summary>

Did you write the `update()` loop? With an empty `update()`, every particle stays at the emitter
point and nothing moves. Also confirm the build picked up your edit (rebuild).
</details>

<details>
<summary>Particles fly away and never come back.</summary>

Make sure step 4 calls `respawn(i)` when `life <= 0`. Without it, particles age past zero, keep
drifting, and eventually leave the screen for good.
</details>

<details>
<summary>Everything falls instantly / looks frozen.</summary>

Check the order: apply gravity to `vy` *before* moving, and multiply by `dt` everywhere
(`* dt`). Forgetting `dt` makes the motion depend on frame rate and usually looks far too fast.
</details>

<details>
<summary>Show me the answer.</summary>

```cpp
void update(float dt) {
    for (int i = 0; i < params_.n; ++i) {
        particles_[i].vy   -= params_.gravity * dt;        // 1. gravity
        particles_[i].x    += particles_[i].vx * dt;       // 2. move
        particles_[i].y    += particles_[i].vy * dt;
        particles_[i].life -= dt;                          // 3. age
        if (particles_[i].life <= 0.0f) respawn(i);        // 4. recycle
    }
}
```

Try writing it yourself first — the muscle memory matters when this becomes a CUDA kernel in Phase 2.
</details>
