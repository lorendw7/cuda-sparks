# The Mathematics Behind the Sim — Advanced-Learning Roadmap

Every preset and level in this project quietly stands on a *real* branch of math, physics,
or CS. Building the sim, you have already been **using** numerical integration, vector
calculus, stochastic processes, chaos theory, procedural noise, and GPU architecture —
without formally studying any of them. Each was "a line that works."

This document is a **study map**, not a code deliverable. For each theory domain it gives:

- **what it is** — the idea in one breath;
- **where it already lives in your code** — the exact kernel / line you wrote;
- **a picture-first path** — a video to build intuition, then a book to make it rigorous;
- **a hands-on exercise** — usually one that drops into this same GLFW/OpenGL + CUDA
  framework, so theory turns back into something you can run.

Pick domains in **any order**. A [priority ladder](#the-shortest-high-yield-path) at the end
suggests one if you want a route. Two [capstone exercises](#capstone-exercises) turn the most
abstract ideas (chaos, fractals) into pictures *and* double as clean CUDA practice.

> **Scope.** This is optional, beyond the build itself. None of it is needed to *finish* or
> *extend* the sim — it is here to convert "it works" into "I understand why," for when you
> want the theory under the code you already wrote.

---

## The map at a glance

Each domain first shows up at a specific point in the build. The whole roadmap, in order of
first appearance:

| # | Domain | First appears | The "line that works" you already wrote |
|---|--------|---------------|------------------------------------------|
| [1](#1-numerical-methods-for-odes--how-every-particle-moves) | Numerical methods for ODEs | Phase 1 update loop | semi-implicit Euler: update `v`, *then* `x` |
| [2](#2-vector-calculus--field-theory--curl-noise-swirl-potentials) | Vector calculus & field theory | Phase 3 swirl → Phase 4 curl-noise | curl of a scalar potential `(∂ψ/∂y, −∂ψ/∂x)` |
| [3](#3-dynamical-systems--chaos--the-lorenz-butterfly) | Dynamical systems & chaos | Phase 4 L6 Lorenz preset | the Lorenz strange attractor |
| [4](#4-probability--stochastic-processes--smoke-rng-randomness-that-looks-organic) | Probability & stochastic processes | Phase 4 smoke + RNG | `curandState` → random walk (`/ sqrtf(dt)`) |
| [5](#5-procedural-noise--coherent-fields-instead-of-confetti) | Procedural noise | Phase 4 curl-noise preset | a smooth potential `ψ(x,y,t)` from sine octaves |
| [6](#6-parallel-computing--gpu-architecture--why-1m-is-memory-bound) | Parallel computing & GPU architecture | Phase 2 → Phase 4 L4 | SoA / coalescing / the roofline you measured |
| [7](#7-computer-graphics--how-points-become-pixels) | Computer graphics | Phase 1 L3 shaders | the point-sprite + blending pipeline |

---

## Prerequisites

Three things unlock everything below. If any feel shaky, fill them first — they pay for
themselves immediately.

- **Single-variable calculus** — a derivative is a *rate of change*. You already use `dp/dt`
  every frame.
- **Basic linear algebra** — vectors, dot product, the 2×2 rotation matrix, eigenvalues.
- **Reading a first-order ODE** — comfort with `dx/dt = f(x)` as "the rule that says how the
  state changes," which is literally what every force term computes.

Fastest fills: MIT OCW **18.01** (calculus) and **18.06** (linear algebra); 3Blue1Brown's
*Essence of Calculus* and *Essence of Linear Algebra* for the intuition in a weekend.

---

## 1. Numerical methods for ODEs — *how every particle moves*

- **In your code:** the `(ax, ay)` accumulator + **semi-implicit (symplectic) Euler** (update
  `v` first, then `x`) in `update_kernel`; the frame-rate-independent `dt` (clamped, and the
  `powf(damping, dt*60)` drag rescale); the attractor branch's **forward Euler with adaptive
  substepping** and the `|1 − kh| > 1` blow-up you traced when a long frame over-extrapolated
  the fast Lorenz velocity.
- **Theory:** truncation error and the *order* of a method; **stability** as a function of
  step size; explicit vs implicit vs symplectic integrators (why semi-implicit Euler roughly
  conserves energy while plain explicit Euler pumps it and orbits spiral out); Runge–Kutta
  (RK4); adaptive step-size control.
- **Learn:** 3Blue1Brown *Differential Equations* series → *Numerical Recipes* ch. 16, or
  Hairer, *Solving Ordinary Differential Equations I*. Video search: "RK4 explained".
- **Do:** add an **RK4** integrator option to the Lorenz branch and compare how much larger a
  step it stays stable at, vs your substepped Euler — you will *see* order-4 accuracy buy you
  a coarser `dt` for the same-looking butterfly.

## 2. Vector calculus & field theory — *curl-noise, swirl, potentials*

- **In your code:** `swirl` (a tangential vector field wound around the origin); the
  curl-noise style's scalar **potential** `ψ(x,y,t)` and its **2-D curl**
  `(∂ψ/∂y, −∂ψ/∂x)` estimated by central finite differences; and the fact that this curl is
  **divergence-free by construction** (the mixed second partials cancel) — which is exactly
  why the curl-noise flow keeps density uniform and the swirls stay invisible until you color
  by velocity direction.
- **Theory:** gradient / divergence / curl and their physical meaning (source–sink vs
  rotation); scalar and vector potentials; the Helmholtz decomposition; why `div(curl) = 0`;
  finite-difference approximations of derivatives.
- **Learn:** 3Blue1Brown *Divergence and curl* + Khan Academy *Multivariable calculus*
  (grad/div/curl) → *Div, Grad, Curl and All That* (Schey), or MIT OCW **18.02**.
- **Do:** render `div(ψ)` and `curl(ψ)` of your own potential as color fields side by side and
  confirm the curl-noise flow's divergence stays ≈ 0 everywhere — the numerical proof of the
  "invisible swirl" you had to color-code around.

## 3. Dynamical systems & chaos — *the Lorenz butterfly*

- **In your code:** the `lorenz()` velocity field (σ = 10, ρ = 28, β = 8/3); the strange
  **attractor** it traces; **sensitive dependence on initial conditions** (nearby seeds smear
  across the whole butterfly); **Poincaré–Bendixson** (why the model needed a 3rd coordinate
  `z` — a 2-D autonomous flow *cannot* be chaotic); and the *manifold* / *fractal-dimension*
  words that describe the resulting set.
- **Theory:** phase space; fixed points and their **eigenvalue stability**; limit cycles;
  **bifurcations** (the period-doubling route to chaos); **Lyapunov exponents** (the rate of
  that exponential smearing); strange attractors; **fractal / Hausdorff dimension** (why the
  Lorenz set is ≈ 2.06-dimensional).
- **Learn:** 3Blue1Brown *chaos* video + Veritasium *logistic map* → **Strogatz, *Nonlinear
  Dynamics and Chaos*** (early chapters + the Lorenz chapter — the canonical, genuinely
  readable text; his Cornell lectures are free on YouTube).
- **Do:** both [capstone sketches](#capstone-exercises) — the logistic bifurcation diagram and
  the Mandelbrot set — are pure dynamical-systems pictures you can render in this framework.

## 4. Probability & stochastic processes — *smoke, RNG, randomness that looks organic*

- **In your code:** the per-particle **`curandState`** stream (seed / subsequence / offset,
  so each particle's randomness is decorrelated); smoke's per-frame random horizontal kick
  that, integrated into `vx`, becomes a **random walk / Brownian motion**; and the
  **`/ sqrtf(dt)`** fix that keeps the diffusion rate frame-rate-independent — because
  variances add, so spread grows as **√(steps)**, not linearly.
- **Theory:** pseudo-random generators (XORWOW vs counter-based **Philox** — ties straight
  back to L6's bandwidth note, since the RNG *state* is your biggest byte-mover); uniform vs
  normal draws; expectation and **variance**; the random-walk → **diffusion-equation** limit;
  why independent white noise ≠ spatially-coherent noise (→ domain 5).
- **Learn:** 3Blue1Brown *Central Limit Theorem* + *But what is a random walk?* → any intro
  probability text (Blitzstein, *Introduction to Probability*; the Harvard **Stat 110** videos
  are free).
- **Do:** log your smoke particles' spread vs time and confirm it grows as **√t** (not `t`);
  then delete the `/ sqrtf(dt)` and watch the diffusion rate change with frame rate — the bug
  made visible.

## 5. Procedural noise — *coherent fields instead of confetti*

- **In your code:** the curl-noise `ψ` built from a couple of drifting sine octaves; the
  principle that turbulence has to be a **smooth field sampled by position**, not per-particle
  white noise — the difference between smoke (independent kicks, domain 4) and the curl-noise
  flow (a coherent field every particle in a region shares).
- **Theory:** value noise vs gradient (**Perlin**) noise vs **simplex** noise; **fractal
  Brownian motion** (summing octaves at halving amplitude / doubling frequency); and turning
  any smooth potential into a divergence-free flow by taking its curl — **"curl noise"**
  (Bridson 2007), the exact technique your preset uses.
- **Learn:** *The Book of Shaders* (the noise chapters, interactive) → Ken Perlin's original
  talks; Bridson, *Curl-Noise for Procedural Fluid Flow* (a short, readable paper).
- **Do:** replace the two-sine `ψ` with real gradient (Perlin) noise summed over a few
  **octaves** (FBM) and watch the flow gain multi-scale detail — big eddies with small ones
  riding on them.

## 6. Parallel computing & GPU architecture — *why 1M is memory-bound*

- **In your code:** SoA vs AoS; **coalesced** memory access (a warp's 32 threads reading
  contiguous addresses); the **roofline** you measured in L4 (Memory 93.9% vs Compute 14.2%
  ⇒ bandwidth-bound, and specifically L2-limited); occupancy; warps / SIMT; `__constant__`
  broadcast memory for the emitter table; and CUDA–OpenGL interop killing the PCIe round trip.
- **Theory:** the **roofline model** (arithmetic intensity decides compute- vs memory-bound,
  and explains L3's null SoA result — same bytes, rearranged, can't beat a byte-count wall);
  the memory hierarchy and bandwidth; warp divergence; occupancy vs latency hiding; parallel
  **reductions** and atomics (needed for a GPU stat read-back and the audio-reactive scalar).
- **Learn:** the NVIDIA **CUDA C++ Programming Guide** + **Best Practices Guide** →
  *Programming Massively Parallel Processors* (Kirk & Hwu — the standard text).
- **Do:** finish **L7** (FP16 packing on tolerant fields) as the *measured* payoff of the L4
  diagnosis, and add a parallel **reduction** that computes total kinetic energy on-GPU — the
  same scalar the audio-reactive layer wants (see [AUDIO.md](../phases/phase4_one_million/AUDIO.md)).

## 7. Computer graphics — *how points become pixels*

- **In your code:** the hand-written OpenGL pipeline (VBO / VAO / shaders / program);
  **normalized device coordinates**; point sprites (`gl_PointSize`) and the fragment-shader
  disc that rounds each square point; and **alpha vs additive blending** (the high-density
  additive plan, where overlap should read as a brightness gradient).
- **Theory:** the rasterization pipeline; homogeneous / clip coordinates; the blending
  equation `src·factor + dst·factor`; gamma vs linear color; premultiplied alpha.
- **Learn:** **LearnOpenGL.com** (whose concepts you have already been using) →
  *Real-Time Rendering* (Akenine-Möller) for the deep version.
- **Do:** the per-vertex point **`size`** attribute + additive-blend renderer upgrade (listed
  under Phase 4's per-style polish) — it unlocks the twinkle / size-by-life / streak effects
  that several presets are waiting on.

---

## Capstone exercises

Both of these are **embarrassingly parallel** (every point / pixel is independent, zero
cross-talk) and drop straight into the existing GLFW/OpenGL + CUDA framework — ideal
self-contained kernels that make *fractal*, *bifurcation*, and *attractor* tangible.

### Logistic-map bifurcation diagram *(do this first)*

The cleanest "order → chaos" picture in all of math, and a perfect first standalone CUDA +
rendering warm-up.

- **The map:** iterate `x → r·x·(1 − x)` (a population model). For a fixed `r`, from almost
  any start `x₀ ∈ (0, 1)`, `x` settles onto an **attractor**: a single fixed point for small
  `r`, then a 2-cycle, 4-cycle, 8-cycle… (**period-doubling**), then chaos.
- **The plot:** x-axis = `r` (sweep ≈ 2.5 → 4.0 across the screen width); y-axis = the settled
  `x` values. For each `r`: iterate ~1000 steps to discard the **transient**, then plot the
  next ~200–500 `x` values as points. One image column per `r`.
- **CUDA shape:** one **thread per `r`-column** — each independently iterates its own `x` and
  writes its settled points; zero cross-talk, a perfect first parallel kernel. Map `r` →
  screen-x and `x` → screen-y into the existing `[x, y, r, g, b]` VBO and reuse the `Renderer`
  as-is (`GL_POINTS`). Write a slow **CPU version first** to verify the shape, then move the
  per-`r` loop into a kernel.
- **What to look for:** the **period-doubling cascade**; the onset of chaos near `r ≈ 3.57`;
  the **white non-chaotic windows** inside chaos (notably the period-3 window near
  `r ≈ 3.83`); and **Feigenbaum's constant** δ ≈ 4.669 in the shrinking spacing of successive
  forks — measure the fork `r`-values, take ratios, and watch a genuinely *universal* number
  fall out of your own picture.
- **Stretch:** color by local density (how often each `x` is visited); an interactive zoom
  into one fork; a live `r`-sweep animation.

### Mandelbrot / Julia set

Iterate `z → z² + c` per pixel, color by escape time, zoom into infinite self-similar detail.
The canonical fractal — and one **thread per pixel**, so it maps to CUDA even more directly
than the bifurcation diagram.

### *(stretch)* Double pendulum or a 2-body → 3-body sim

Chaos and sensitive dependence you can *feel* by nudging the initial conditions — the physical
cousin of the Lorenz butterfly.

---

## The shortest high-yield path

If you want a route rather than a buffet:

1. **Shore up the [prerequisites](#prerequisites)** — calculus, linear algebra, reading an ODE.
2. **Domain 1 (integration)** and **domain 3 (chaos)** — they underpin the most presets, and
   almost every force term is one or the other.
3. **Code the [logistic bifurcation diagram](#logistic-map-bifurcation-diagram-do-this-first)
   + [Mandelbrot set](#mandelbrot--julia-set)** — theory becomes a picture *and* CUDA practice
   in one move.
4. **Branch into whichever domain pulls you next** — fields (2), stochastic (4), noise (5),
   GPU (6), or graphics (7).

After that, none of the words under these presets are vocabulary anymore — they are things you
can derive, plot, and tune.

---

## Only if you want the formal rigor

The precise definitions behind the vocabulary are **graduate-flavored and not needed** to
understand or extend the sim — reach for them only out of curiosity:

- **Manifold** — a space with locally-flat coordinates, like the Earth's surface — lives in
  differential geometry / topology (Milnor, *Topology from the Differentiable Viewpoint*;
  do Carmo, *Differential Geometry of Curves and Surfaces*).
- **Hausdorff dimension** — the measure-theoretic reason the Lorenz set is ≈ 2.06-D — lives in
  geometric measure theory.

Everything you actually need to keep building is in domains 1–7 above.
