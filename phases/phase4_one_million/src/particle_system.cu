#include "particle_system.h"
#include "cuda_utils.h" // CUDA_CHECK macro + gpu_info()

#include <cuda_runtime.h>
#include <cmath> // floorf

#include <glad/gl.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

__constant__ Emitter d_emitters[MAX_EMITTERS];

// ===========================================================================
// presets  --  the effect library (L5-4).  One Preset bundles an emitter table
// with the physics knobs (gravity / nbodyStrength / swirl / damping) that shape it.
// ===========================================================================
// Field order matches the struct: { emitters[MAX_EMITTERS], numEmitters, gravity,
// nbodyStrength, swirl, damping }. damping (L6-1) is air drag: v *= damping each
// step -- 1.0 frictionless, <1 decelerates. Only the first numEmitters rows are used; the rest are
// zero-filled by the aggregate initializer and never uploaded. Add/remove a whole
// Preset here and numPresets (below) + the number keys in main.cpp pick it up.
//   0 fireworks -- scattered full-circle bursts, positive gravity (fall)
//   1 fire      -- bottom row aimed up, negative gravity (buoyant rise), warm
//   2 galaxy    -- warm orange core + two cool arms; swirl winds them into a spinning disk
//   3 Jia       -- two diagonal jets (pink + gold), gentle swirl curves them into
//                  winding arms, no gravity
// ===========================================================================
static const Preset presets[] = {
    // ---- 0: fireworks -- 3 full-circle bursts, fast launch, slight swirl, fall under gravity ----
    {
        {
            //  x     y   angle  spread  baseSpd  r    g    b    life
            {-0.5f, 0.3f, 0.0f, 6.2832f, 1.2f, 1.0f, 0.2f, 0.2f, 4.0f}, // red,  upper-left
            {0.5f, 0.4f, 0.0f, 6.2832f, 1.2f, 0.2f, 0.6f, 1.0f, 4.0f},  // blue, upper-right
            {0.0f, -0.2f, 0.0f, 6.2832f, 1.2f, 1.0f, 0.9f, 0.3f, 4.0f}, // gold, center
        },
        3,         // numEmitters
        0.45f,     // gravity
        0.000012f, // nbodyStrength
        0.4f,      // swirl
        0.97f,     // damping -- strong drag. NOTE this is still the L5 CONTINUOUS-fountain
                   //   model: 3 fixed points emit non-stop, so sparks puff out & fall near
                   //   the emitter ("spread in place") rather than a real fly-out-and-arc
                   //   burst. Raise toward 0.99 to let them travel farther; the true
                   //   born-together / explode / die-together look arrives with L6 shell bursts.
        1,
    },
    // ---- 1: fire -- 3 narrow upward jets along the bottom, buoyant (negative gravity), warm ----
    {
        {
            {-0.15f, -0.9f, 1.5708f, 0.28f, 0.9f, 1.0f, 0.3f, 0.05f, 3.0f}, // deep red
            {0.0f, -0.9f, 1.5708f, 0.28f, 1.0f, 1.0f, 0.6f, 0.1f, 3.0f},    // orange
            {0.15f, -0.9f, 1.5708f, 0.28f, 0.9f, 1.0f, 0.85f, 0.2f, 3.0f},  // yellow
        },
        3,
        -0.5f,
        0.000006f,
        0.0f,
        0.99f, // damping -- light drag so the flames settle instead of shooting off
        0,
    },
    // ---- 2: galaxy -- warm orange core + two cool arms, wound by swirl into a spinning disk ----
    // The orange source sits AT the origin (0,0): swirl/nbody are both centered there, so a
    // particle at r=0 feels no sideways push and stays put -> a tight bright nucleus. The two
    // blue sources are offset (x = +/-0.5), so swirl winds them into two spiral arms. gravity 0 keeps
    // the disk round. (Was "nebula" -- an orange core + a warm color turned the cloud into a galaxy.)
    {
        {
            //  x     y   angle  spread  baseSpd  r    g    b     life
            {-0.5f, 0.0f, 0.0f, 6.2832f, 0.25f, 0.3f, 0.4f, 1.0f, 9.0f}, // cool arm #1 (blue)
            {0.0f, 0.0f, 0.0f, 6.2832f, 0.06f, 1.0f, 0.6f, 0.15f, 3.5f}, // warm orange core
            {0.5f, 0.0f, 0.0f, 6.2832f, 0.25f, 0.5f, 0.6f, 1.0f, 9.0f},  // cool arm #2 (lighter blue)
        },
        3,        // numEmitters
        0.0f,     // gravity        -- 0: keep the disk round, not squashed downward
        0.00003f, // nbodyStrength  -- inward pull that tightens the disk (balance this vs swirl)
        1.2f,     // swirl          -- orbital spin; winds the two offset arms into spirals
        1.0f,     // damping        -- 1.0 = frictionless, so orbits stay stable & long-lived
        0,
    },
    // ---- 3: Jia -- two diagonal jets (pink + gold) bent into arms by a gentle swirl ----
    // Each source fires a directional fan (spread = 1.8 rad, ~103 deg) aimed diagonally,
    // so here the angle field DOES steer the stream (unlike the full-circle presets above).
    // Placed on opposite corners (upper-left / lower-right); the +swirl vortex curves the
    // two jets so the pink and gold streams wind around the center instead of flying off.
    {
        {
            //  x     y     angle    spread  baseSpd  r    g     b     life
            {-0.6f, 0.8f, -0.785f, 1.8f, 0.2f, 1.0f, 0.71f, 0.76f, 8.0f}, // light pink, upper-left
            {0.6f, -0.8f, 2.356f, 1.8f, 0.2f, 1.0f, 0.84f, 0.0f, 8.0f},   // gold, lower-right
        },
        2,        // numEmitters
        0.0f,     // gravity        -- 0: keep the spiral round, not squashed downward
        0.00002f, // nbodyStrength  -- a light inward leash so the arms don't fly to the walls
        0.62f,    // swirl          -- gentle counter-clockwise vortex braids the two colors
        0.99962f, // damping        -- very light drag keeps the braided jets tight
        0,
    }};

static const int numPresets = sizeof(presets) / sizeof(presets[0]);

// ===========================================================================
// set_preset  --  switch the whole look to preset i.  (ctor boot + number keys)
// ===========================================================================
// Clamp i into range, then (1) re-upload that preset's emitter table into
// __constant__ memory via upload_emitter (which also sets params_.numEmitters),
// and (2) copy its physics knobs (gravity/nbodyStrength/swirl/damping) + the
// useShells mode flag (L6-2) into params_ so the next update_kernel
// launch uses them. Cheap: no realloc, no init_kernel -- the 1M particles already
// alive keep flying and only adopt the new look as they recycle, so a switch
// fades in over roughly one lifetime (a feature, not a bug).
// ===========================================================================
void ParticleSystem::set_preset(int i)
{
    if (i < 0)
    {
        i = 0;
    }

    if (i >= numPresets)
    {
        i = numPresets - 1;
    }

    const Preset &pr = presets[i];
    upload_emitter(pr.emitters, pr.numEmitters);

    params_.gravity = pr.gravity;
    params_.nbodyStrength = pr.nbodyStrength;
    params_.swirl = pr.swirl;
    params_.damping = pr.damping;
    params_.useShells = pr.useShells;
}

// A small palette of saturated firework colors; advance_shells picks one per burst.
// __device__ = lives in GPU memory, readable by kernels. Colors are yours to pick;
// keep them BRIGHT & saturated (real fireworks are single vivid hues, never muddy).
__device__ const float palette[][3] = {
    {1.0f, 0.3f, 0.3f}, // red
    {1.0f, 0.8f, 0.2f}, // gold
    {0.3f, 0.6f, 1.0f}, // blue
    {0.5f, 1.0f, 0.4f}, // green
    {1.0f, 0.4f, 0.9f}, // magenta
    {0.4f, 1.0f, 1.0f}, // cyan
};

// ===========================================================================
// spawn_burst  --  born particle i from a SHELL, not an emitter.  (L6-2)
// ===========================================================================
// The shell-mode counterpart of spawn(): update_kernel calls this on the frame a
// shell relaunches (shell.launch==1), so all of a shell's particles appear together
// at its center. Unlike spawn() (emitter aim + spread), a burst fires in a random
// FULL-CIRCLE direction with a random speed from 0 (so the bloom is a filled disk,
// not a hollow ring), wears the shell's single color, and takes the shell's remaining
// burst time as its life so it fades in step with the shell.
// ===========================================================================
__device__ inline void spawn_burst(ParticleSoA p, int i, Shell sh, curandState *st)
{
    float a = curand_uniform(st) * 6.2832f; // random direction over the full circle (2*pi)
    float s = curand_uniform(st) * 0.8;     // random speed 0..0.8 -> a filled disk of sparks
    p.vx[i] = cosf(a) * s;                  // polar (a, s) -> Cartesian velocity
    p.vy[i] = sinf(a) * s;
    p.x[i] = sh.cx; // born at the shell's burst center
    p.y[i] = sh.cy;
    p.cr[i] = sh.cr; // wear the shell's one burst color
    p.cg[i] = sh.cg;
    p.cb[i] = sh.cb;
    p.life[i] = sh.timer; // life = shell's remaining burst time -> fades with the shell
}

// ===========================================================================
// spawn  --  write particle i's initial/reborn state into the SoA arrays.
// ===========================================================================
// Device-only (L3): both callers -- init_kernel (initial fill) and update_kernel
// (recycle a dead particle) -- run on the GPU, so there is no host path left.
// (L5) A particle is now BORN FROM AN EMITTER: particle i is assigned emitter
// (i % numEmitters), whose recipe is read from the __constant__ d_emitters table,
// and takes that emitter's position / aim / colour / lifetime. The jitter (angle
// within spread, speed, lifetime) now comes from this particle's own curandState
// (curand_uniform) -- so each spawn, and each later rebirth, draws fresh randoms.
// ===========================================================================
__device__ inline void spawn(ParticleSoA p, int i, int numEmitters, curandState *st)
{
    // Three fresh randoms in (0,1]. curand_uniform advances *st, so each rebirth
    // of this particle draws NEW values -> no repeating path, no ripple structure.
    float f1 = curand_uniform(st);
    float f2 = curand_uniform(st);
    float f3 = curand_uniform(st);

    // Pick this particle's emitter and copy its recipe out of constant memory.
    int e = i % numEmitters;    // round-robin: assign particle i to one emitter
    Emitter em = d_emitters[e]; // broadcast read from the __constant__ table

    // Launch direction: the emitter's central aim, jittered within its spread.
    // (f1-0.5) is in [-0.5,0.5), so a lands in [angle - spread/2, angle + spread/2).
    float a = em.angle + (f1 - 0.5f) * em.spread;

    // Launch speed: baseSpeed scaled by a per-particle factor in [0.5,1.0).
    float s = em.baseSpeed * (0.5f + 0.5f * f2);

    // Polar (direction a, magnitude s) -> Cartesian velocity components.
    p.vx[i] = cosf(a) * s;
    p.vy[i] = sinf(a) * s;

    // Born at the emitter's mouth, wearing the emitter's colour.
    p.x[i] = em.x;
    p.y[i] = em.y;

    p.cr[i] = em.r;
    p.cg[i] = em.g;
    p.cb[i] = em.b;

    // Staggered lifetime (0.7..1.0 of the emitter's lifetime) so particles from
    // one emitter don't all die on the same frame -> a continuous stream, not a pulse.
    p.life[i] = em.lifetime * (0.7f + 0.3f * f3);
}

// ===========================================================================
// init_rng  --  seed one curandState per particle.  (runs ONCE, before init_kernel)
// ===========================================================================
// Same seed, but subsequence = i, so every particle gets its own decorrelated
// random stream. Must run BEFORE init_kernel, since spawn() now draws from it.
// ===========================================================================
__global__ void init_rng(curandState *rng, int n, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    curand_init(seed, i, 0, &rng[i]);
}

// ===========================================================================
// advance_shells  --  run each shell's state machine one step.  (every frame)
// ===========================================================================
// L6-2 heart. One GPU thread per shell (s = numShells, ~16). Each shell is a tiny
// two-phase state machine advanced by its own countdown `timer`:
//   live=1 (exploding): timer counts down the burst's lifetime; on expiry -> go
//           dark (live=0) and set a random dark-gap countdown.
//   live=0 (dark):      timer counts down the dark gap; on expiry -> RELAUNCH:
//           pick a new random center + a palette color, set a random burst
//           lifetime, live=1, and raise launch=1 for THIS frame only.
// `launch` is the one-frame pulse update_kernel (2c) reads to (re)spawn this
// shell's particles at the new center. It is cleared at the TOP every frame, so
// it is 1 for exactly the relaunch frame and 0 otherwise -- which is why this
// kernel MUST run BEFORE update_kernel each frame (so the pulse is fresh). One
// RNG per shell (rng[i]) draws the center / color / timings.
// ===========================================================================
__global__ void advance_shells(Shell *shells, int s, float dt, curandState *rng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s)
    {
        return;
    }

    shells[i].launch = 0;  // default: not a launch frame (may be re-raised below)
    shells[i].timer -= dt; // advance the current phase's countdown

    if (shells[i].timer <= 0.0f) // current phase finished -> switch state
    {
        if (shells[i].live) // ---- was exploding -> go dark ----
        {
            shells[i].live = 0;
            shells[i].timer = 0.5f + curand_uniform(&rng[i]) * 2.0f; // random dark gap 0.5..2.5s
        }
        else // ---- dark gap over -> relaunch somewhere new ----
        {
            shells[i].live = 1;
            shells[i].launch = 1;                                     // pulse: tells update_kernel to respawn this shell's particles
            shells[i].cx = ((curand_uniform(&rng[i]) - 0.5f) * 1.6f); // new center x in (-0.8,0.8]
            shells[i].cy = ((curand_uniform(&rng[i]) - 0.5f) * 1.6f); // new center y in (-0.8,0.8]

            // Pick one palette color for the whole burst (one vivid hue per firework).
            int np = sizeof(palette) / sizeof(palette[0]); // color count (sizeof idiom)
            int c = (int)(curand_uniform(&rng[i]) * np);   // random index [0, np)
            if (c >= np)                                   // curand_uniform is (0,1] -> can hit 1.0
            {
                c = np - 1; // clamp the 1.0 case out of bounds
            }

            shells[i].cr = palette[c][0];
            shells[i].cg = palette[c][1];
            shells[i].cb = palette[c][2];
            shells[i].timer = 1.5f + curand_uniform(&rng[i]) * 1.0f; // random burst lifetime 1.5..2.5s
        }
    }
}

// ===========================================================================
// init_shells  --  seed the shell state machine.  (runs ONCE, at startup)
// ===========================================================================
// Start EVERY shell DARK with a staggered countdown (curand_uniform * 3s), so
// they don't all launch on the same frame -> bursts appear at different times.
// center/color are left 0 here; they get their real values at the first relaunch
// (in advance_shells, 2b), when the dark timer runs out. One RNG per shell (rng[i]).
// ===========================================================================
__global__ void init_shells(Shell *shells, int s,
                            curandState *rng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s)
        return;
    shells[i].live = 0;                               // dark: not exploding yet
    shells[i].launch = 0;                             // not a launch frame
    shells[i].timer = curand_uniform(&rng[i]) * 3.0f; // staggered dark countdown (0..3s)
    shells[i].cx = 0.0f;                              // center/color are chosen at relaunch, not here
    shells[i].cy = 0.0f;
    shells[i].cr = 0.0f;
    shells[i].cg = 0.0f;
    shells[i].cb = 0.0f;
}

// ===========================================================================
// init_kernel  --  build the whole initial population on the GPU.  (runs ONCE)
// ===========================================================================
// One thread per particle calls spawn(), drawing from its own curandState (so
// init_rng MUST have run first). The whole initial population is built directly in
// device memory -- no CPU arrays, no host->device upload (that's why L3 has no host_ mirror).
// ===========================================================================
__global__ void init_kernel(ParticleSoA p, int n, int numEmitters, curandState *rng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    spawn(p, i, numEmitters, &rng[i]); // forward the emitter count so spawn does i % numEmitters
}

// ===========================================================================
// update_kernel  --  advance one particle AND write its vertex.  (every frame)
// ===========================================================================
// One GPU thread per particle: sum three forces (gravity + central attractor +
// swirl) into an acceleration, integrate velocity, apply air drag (L6-1), move
// the position, bounce off the 4 walls, then age/respawn -- which now BRANCHES on
// params.useShells (L6-2): shell presets (fireworks) respawn a particle only when
// its shell relaunches (shell.launch) and hide it while its shell is dark; other
// presets keep the L5 continuous model (respawn the moment life hits 0). Finally
// (L2 interop) it packs [x,y,r,g,b] straight into the OpenGL VBO via `vbo` (or, for
// a dark shell, writes the vertex off-screen so GL clips it). No copy to the CPU.
// params is passed BY VALUE (a copy per thread) -- use params.gravity (dot),
// never params_ (that member is invisible here). `shells` is the per-shell state
// array, already advanced this frame by advance_shells (must run first).
// ===========================================================================
__global__ void update_kernel(ParticleSoA p, float *vbo, int n, SimParams params, float dt, curandState *rng, Shell *shells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread id = particle index
    if (i >= n)                                    // grid usually rounds up past n:
        return;                                    // extra threads must do nothing

    // --- Force accumulator (L5-3) ---------------------------------------------
    // Newton: a force produces an ACCELERATION. We sum every force's acceleration
    // into (ax, ay) first, then apply it once. No dt here -- dt is one "per second"
    // scale factor applied at the very end (a is "velocity change per second").
    float ax = 0.0f, ay = 0.0f;
    float px = p.x[i], py = p.y[i]; // read position once, reuse below

    // 1) Gravity: a constant downward acceleration.
    ay -= params.gravity;

    // 2) Central attractor -- the O(n) stand-in for O(n^2) mutual N-body at 1M
    //    scale. Pull every particle toward the origin (0,0):
    //      px*px + py*py = r^2  (squared distance to origin, Pythagoras)
    //      rsqrtf(r^2)   = 1/r  (fast reciprocal-sqrt hardware instruction)
    //      (px,py)*invr  = the UNIT vector origin->particle (length 1, direction only)
    //    Negate it to point particle->origin, scale by nbodyStrength. Because the
    //    vector is normalized, the inward pull is the same strength at any distance.
    //    +0.01f is "softening": stops 1/r blowing up when a particle sits at r=0.
    float invr = rsqrtf(px * px + py * py + 0.01f);
    ax -= px * invr * params.nbodyStrength;
    ay -= py * invr * params.nbodyStrength;

    // 3) Swirl: a tangential (vortex) force around the origin. Rotating the radius
    //    vector (px,py) by +90 degrees gives the tangent (-py, px); pushing along it
    //    makes particles ORBIT the center. No divide by r, so outer particles move
    //    faster (rigid rotation) and the cloud winds into spiral arms -- the shape
    //    mutual gravity alone can never make (that only collapses to a point).
    ax -= py * params.swirl;
    ay += px * params.swirl;

    // --- Integrate (semi-implicit Euler) --------------------------------------
    // Velocity first, THEN use the NEW velocity to move the position. This
    // ordering (not position-first) keeps orbits stable, which swirl needs.
    p.vx[i] += ax * dt; // v += a*dt   (acceleration changes velocity)
    p.vy[i] += ay * dt;

    // Air drag (L6-1): scale velocity down a hair each step -> exponential decay,
    // so a fast spark decelerates and arcs instead of flying straight to the wall.
    // damping is defined per 60fps-frame; powf(damping, dt*60) rescales it to THIS
    // frame's real length so the drag-per-second is identical at any FPS (same
    // frame-rate-independence fix as dt itself -- a plain v*=damping would drag
    // ~50x harder at thousands of FPS).
    float drag = powf(params.damping, dt * 60.0f);
    p.vx[i] *= drag;
    p.vy[i] *= drag;

    p.x[i] += p.vx[i] * dt; // x += v*dt   (velocity changes position)
    p.y[i] += p.vy[i] * dt;

    // Bounce off each of the 4 walls: clamp to the wall, flip velocity, scale by
    // restitution (energy kept per bounce). Clamp value must match the wall tested.
    if (p.x[i] > params.bound)
    {
        p.x[i] = params.bound;
        p.vx[i] = -p.vx[i] * params.restitution;
    }
    if (p.x[i] < -params.bound)
    {
        p.x[i] = -params.bound;
        p.vx[i] = -p.vx[i] * params.restitution;
    }
    if (p.y[i] > params.bound)
    {
        p.y[i] = params.bound;
        p.vy[i] = -p.vy[i] * params.restitution;
    }
    if (p.y[i] < -params.bound)
    {
        p.y[i] = -params.bound;
        p.vy[i] = -p.vy[i] * params.restitution;
    }

    // --- Age / respawn: two models, chosen per preset (L6-2) ------------------
    if (params.useShells) // shell model (fireworks): the shell drives birth/death
    {
        int sh = i % params.numShells; // which shell this particle belongs to
        if (shells[sh].launch)         // shell relaunched THIS frame -> born together at its center
        {
            spawn_burst(p, i, shells[sh], &rng[i]);
        }
        p.life[i] -= dt; // still age, for the color fade below (visibility gated on shell.live)
    }
    else // continuous model (L5): each particle respawns itself the instant it dies
    {
        p.life[i] -= dt;
        if (p.life[i] <= 0.0f)
        {
            spawn(p, i, params.numEmitters, &rng[i]);
        }
    }

    // --- Write this particle's vertex straight into the VBO (must be AFTER the
    // recycle above, so a just-reborn particle emits its NEW position/color, not
    // the stale dead one). fade dims the color toward death; clamp to [0,1]
    // because a freshly spawned particle's life can exceed 1.
    float fade = p.life[i];
    if (fade < 0.0f)
    {
        fade = 0.0f;
    }
    if (fade > 1.0f)
    {
        fade = 1.0f;
    }

    // Visibility gate (L6-2): a particle whose shell is DARK is hidden. Non-shell
    // presets are always visible. (Hidden = written off-screen below, not colored
    // black -- a black dot would still draw a visible dark disc under alpha blend.)
    int visible = 1;
    if (params.useShells && shells[i % params.numShells].live == 0)
    {
        visible = 0;
    }

    float *v = &vbo[i * 5]; // this particle's 5 floats inside the VBO
    if (visible)
    {
        v[0] = p.x[i];         // attribute 0 (pos.x) -- shader reads offset 0
        v[1] = p.y[i];         // attribute 0 (pos.y)
        v[2] = p.cr[i] * fade; // attribute 1 (color.r) -- shader reads offset 8
        v[3] = p.cg[i] * fade; // attribute 1 (color.g)
        v[4] = p.cb[i] * fade; // attribute 1 (color.b)
    }
    else
    {
        v[0] = -2.0f;
        v[1] = -2.0f;
        v[2] = 0.0f;
        v[3] = 0.0f;
        v[4] = 0.0f;
    }
}

// ===========================================================================
// Constructor  --  allocate the 8 SoA arrays, then fill them on the GPU.
// ===========================================================================
// L3: one cudaMalloc per field (8 total). No host mirror, no upload -- the fill
// runs on the device via init_kernel. Each malloc MUST pair with a cudaFree in
// the destructor; miss one field and the kernel writes a wild pointer / leaks.
// ===========================================================================
ParticleSystem::ParticleSystem(const SimParams &p) // definition of the ctor declared in the header
    : params_(p), n_(p.n)                          // member-init list: fill params_ and n_ first
{
    gpu_info(); // print which GPU we're on (from cuda_utils.h)

    size_t bytes = (size_t)n_ * sizeof(float); // one field's array = n floats
    CUDA_CHECK(cudaMalloc(&d_.x, bytes));
    CUDA_CHECK(cudaMalloc(&d_.y, bytes));
    CUDA_CHECK(cudaMalloc(&d_.vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_.vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_.life, bytes));
    CUDA_CHECK(cudaMalloc(&d_.cr, bytes));
    CUDA_CHECK(cudaMalloc(&d_.cg, bytes));
    CUDA_CHECK(cudaMalloc(&d_.cb, bytes));
    CUDA_CHECK(cudaMalloc(&d_rng_, (size_t)n_ * sizeof(curandState)));
    // L6-2: the shell state array + its own (smaller) RNG pool, one per shell.
    CUDA_CHECK(cudaMalloc(&d_shells_, (size_t)params_.numShells * sizeof(Shell)));
    CUDA_CHECK(cudaMalloc(&d_shell_rng_, (size_t)params_.numShells * sizeof(curandState)));

    // Boot the default look (preset 0). set_preset uploads its emitter table into
    // __constant__ memory and sets params_.numEmitters -- so it MUST run before
    // init_kernel below, which reads both to build the initial 1M population.
    set_preset(0);
    int block = 256, grid = (n_ + block - 1) / block;
    init_rng<<<grid, block>>>((curandState *)d_rng_, n_, 1025ULL);
    CUDA_CHECK(cudaGetLastError());
    init_kernel<<<grid, block>>>(d_, n_, params_.numEmitters, (curandState *)d_rng_); // params_.numEmitters was set by set_preset(0) -> upload_emitter above
    CUDA_CHECK(cudaGetLastError());

    // L6-2: seed the shell RNG (reuse init_rng, different seed) then init_shells to
    // put every shell in the DARK/staggered start state. Separate grid sized to
    // numShells (128), not n. Order doesn't depend on the particle init above.
    int sblock = 256, sgrid = (params_.numShells + sblock - 1) / sblock;
    init_rng<<<sgrid, sblock>>>((curandState *)d_shell_rng_, params_.numShells, 2049ULL);
    CUDA_CHECK(cudaGetLastError());
    init_shells<<<sgrid, sblock>>>(d_shells_, params_.numShells, (curandState *)d_shell_rng_);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// register_vbo  --  hand the OpenGL VBO to CUDA.  (called ONCE, at startup)
// ===========================================================================
// One-time handshake: CUDA asks the GL driver for the VBO's physical address,
// size and format, and records them in vbo_resource_ -- a proxy handle for this
// GL buffer in the CUDA world. Registration is expensive (cross-driver
// negotiation), so it lives here, never in the per-frame loop.
//
// Must run AFTER renderer.init(): the VBO has to already exist (glGenBuffers +
// glBufferData with non-zero size) or this fails with cudaErrorInvalidValue.
// WriteDiscard = "the kernel overwrites every vertex each frame and never reads
// the old ones", letting the driver skip preserving previous contents.
// ===========================================================================
void ParticleSystem::register_vbo(unsigned int vbo_id)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &vbo_resource_,                       // out: proxy handle written here
        vbo_id,                               // in:  the GL buffer id to share
        cudaGraphicsRegisterFlagsWriteDiscard // hint: full overwrite, no read-back
        ));
}

// ===========================================================================
// upload_emitter  --  push an emitter table into __constant__ memory. (ctor + set_preset)
// ===========================================================================
// cudaMemcpyToSymbol copies host bytes into the file-scope __constant__ d_emitters
// BY SYMBOL NAME (not a pointer, unlike cudaMemcpy). Also records how many emitters
// are live in params_.numEmitters, which spawn()/init_kernel read to do i % numEmitters.
// ===========================================================================
void ParticleSystem::upload_emitter(const Emitter *e, int count)
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_emitters, e, count * sizeof(Emitter)));
    params_.numEmitters = count;
}

// ===========================================================================
// Destructor  --  release everything we acquired.  (RAII: automatic at scope end)
// ===========================================================================
// Every acquire needs a matching release: unregister pairs with register_vbo,
// cudaFree pairs with cudaMalloc. Unregister FIRST (undo interop) and only if we
// actually registered -- the null guard makes the destructor safe even if the
// object is torn down before register_vbo ran. Note we only unregister, never
// glDeleteBuffers: CUDA borrowed the VBO, it never owned it -- the Renderer does.
// ===========================================================================
ParticleSystem::~ParticleSystem()
{
    if (vbo_resource_) // only if register_vbo succeeded
    {
        cudaGraphicsUnregisterResource(vbo_resource_); // undo the interop registration
    }

    cudaFree(d_.x);
    cudaFree(d_.y);
    cudaFree(d_.vx);
    cudaFree(d_.vy);
    cudaFree(d_.life);
    cudaFree(d_.cr);
    cudaFree(d_.cg);
    cudaFree(d_.cb);
    cudaFree(d_rng_);
    cudaFree(d_shells_);    // L6-2: shell state array
    cudaFree(d_shell_rng_); // L6-2: shell RNG pool
}

// ===========================================================================
// update  --  map the VBO, run the shell + particle kernels into it, unmap.  (per frame)
// ===========================================================================
// The L2 heart. No cudaMemcpy, no host_ mirror, no CPU pack: the kernel writes
// vertices directly into the OpenGL VBO. map/unmap are the per-frame handshake
// that hands the buffer GL -> CUDA and back, and also synchronise the two so the
// kernel's writes are done before draw() reads them.
// ===========================================================================
void ParticleSystem::update(float dt)
{
    // 1) Borrow the VBO: GL -> CUDA. Waits for prior GL work on it to finish.
    CUDA_CHECK(cudaGraphicsMapResources(1, &vbo_resource_, 0));

    // 2) Get this frame's raw device pointer into the VBO (re-fetch every frame;
    //    valid only between map and unmap). bytes = its size, which we don't need.
    float *d_vbo = nullptr;
    size_t bytes = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_vbo, &bytes, vbo_resource_));

    // 3) Launch TWO kernels, in order (L6-2): advance_shells FIRST (advances the
    //    per-shell state machine so this frame's launch/live flags are fresh), then
    //    update_kernel, which reads those flags to spawn/hide particles + writes the
    //    vertices straight into d_vbo. Order matters -- update_kernel must see the
    //    shell state advance_shells just produced.
    int sblock = 256, sgrid = (params_.numShells + sblock - 1) / sblock;
    advance_shells<<<sgrid, sblock>>>(d_shells_, params_.numShells, dt, (curandState *)d_shell_rng_);
    CUDA_CHECK(cudaGetLastError());
    int block = 256;
    int grid = (n_ + block - 1) / block;
    update_kernel<<<grid, block>>>(d_, d_vbo, n_, params_, dt, (curandState *)d_rng_, d_shells_);
    CUDA_CHECK(cudaGetLastError());

    // 4) Return the VBO: CUDA -> GL. Waits for the kernel to finish, so the
    //    buffer is safe for draw() the moment this returns. draw() must come AFTER.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource_, 0));
}
