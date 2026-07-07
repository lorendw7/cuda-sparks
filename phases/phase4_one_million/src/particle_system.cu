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
// nbodyStrength, swirl, damping, useShells, useRain, useFlow, wind, turbulence, curl, useAttractor }. damping (L6-1)
// is air drag: v *= damping each step -- 1.0 frictionless, <1 decelerates. turbulence is smoke's
// random-walk kick; curl (L6) is the curl-noise flow-field strength -- both 0 for every other preset. Only the first numEmitters rows are used; the rest are
// zero-filled by the aggregate initializer and never uploaded. Add/remove a whole
// Preset here and numPresets (below) + the number keys in main.cpp pick it up.
//   0 Jia       -- two diagonal jets (pink + gold), gentle swirl curves them into
//                  winding arms, no gravity (the default look; J key)
//   1 fireworks -- scattered full-circle bursts, positive gravity (fall)
//   2 fire      -- bottom row aimed up, negative gravity (buoyant rise), warm
//   3 galaxy    -- warm orange core + two cool arms; swirl winds them into a spinning disk
//   4 rain      -- drops spawn at a random x across the top and fall under gravity,
//                  land and linger on the floor as a puddle, then respawn up top
//                  (useRain + spawn_rain -> a full-width curtain with a wet floor)
//   5 smoke     -- one narrow upward vent; buoyancy lifts it, turbulence random-walks
//                  it sideways into a widening plume that bounces under the top wall
//   6 curl-noise-- particles seeded across the whole screen (spawn_scatter, useFlow),
//                  advected by a smooth divergence-free curl-noise flow field (curl>0)
//   7 attractor -- Lorenz strange attractor (useAttractor). A velocity-field, not a
//                  force model: (x,y,z) hold the 3D Lorenz state, update_kernel reads
//                  velocity straight from the ODE and projects (x,z) -> twin butterfly wings
// ===========================================================================
static const Preset presets[] = {
    // ---- 0: Jia -- two diagonal jets (pink + gold) bent into arms by a gentle swirl ----
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
        0,        // useShells
        0,        // useRain
        0,        // useFlow
        0.0f,     // wind
        0.0f,     // turbulence
        0.0f,     // curl           -- flow field off
        0,        // useAttractor   -- Lorenz off
    },
    // ---- 1: fireworks -- 3 full-circle bursts, fast launch, slight swirl, fall under gravity ----
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
        1,         // useShells
        0,         // useRain
        0,         // useFlow
        0.0f,      // wind
        0.0f,      // turbulence
        0.0f,      // curl           -- flow field off
        0,         // useAttractor   -- Lorenz off
    },
    // ---- 2: fire -- 3 narrow upward jets along the bottom, buoyant (negative gravity), warm ----
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
        0,     // useShells
        0,     // useRain
        0,     // useFlow
        0.0f,  // wind
        0.0f,  // turbulence
        0.0f,  // curl           -- flow field off
        0,     // useAttractor  -- Lorenz off
    },
    // ---- 3: galaxy -- warm orange core + two cool arms, wound by swirl into a spinning disk ----
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
        0,        // useShells
        0,        // useRain
        0,        // useFlow
        0.0f,     // wind
        0.0f,     // turbulence
        0.0f,     // curl           -- flow field off
        0,        // useAttractor   -- Lorenz off
    },
    // ---- 4: rain -- a full-width falling curtain that puddles on the floor (useRain=1). ----
    // update_kernel spawns each drop via spawn_rain (random x, depth-scaled downward speed),
    // so only the FIRST emitter row is read here -- just for the drop's base COLOR (r,g,b). Its
    // x/y/angle/speed AND the life column are ignored: spawn_rain sets position, per-drop speed,
    // brightness and a huge life sentinel itself. The physics knobs still apply: gravity pulls
    // drops down, swirl/nbody off. A drop falls, LANDS on the floor and lingers as a puddle for
    // a few seconds, then respawns up top -> an endless curtain.
    {
        {
            //  x      y     angle    spread baseSpd  r    g    b    life
            {-0.8f, 0.95f, -1.5708f, 0.15f, 0.25f, 0.15f, 0.55f, 0.62f, 2.5f}, // -1.5708 = -pi/2 -> straight down
        },
        1,     // numEmitters
        0.7f,  // gravity       -- positive: accelerate the drops downward
        0.0f,  // nbodyStrength -- off: no center attraction
        0.0f,  // swirl         -- off: rain doesn't spin
        0.99f, // damping       -- light drag so drops keep speeding up (never 0 = would freeze them)
        0,     // useShells     -- continuous model, no shell burst
        1,     // useRain
        0,     // useFlow
        0.3,   // wind
        0.0f,  // turbulence
        0.0f,  // curl          -- flow field off
        0,     // useAttractor  -- Lorenz off
    },
    // ---- 5: smoke -- one narrow upward vent, buoyant rise + turbulent diffusion. ----
    // A single bottom-center emitter aims a slow, gray, long-lived stream straight up
    // (1.5708 = pi/2). Negative gravity is BUOYANCY (gentler than fire): the column
    // drifts upward. The turbulence knob adds a per-frame random horizontal kick, so
    // the rising velocity does a random walk -> the plume widens and curls as it climbs
    // (see force 5 in update_kernel). climb ~ (buoyancy/drag) * life: buoyancy -0.20 +
    // light damping 0.996 give a fast enough terminal rise that over its 8 s life the
    // plume reaches the TOP wall and, since this is the non-rain path, BOUNCES and pools
    // under it -- reading as smoke hitting a ceiling (a happy accident we kept). Tune it
    // DOWN (weaker gravity / stronger damping / shorter life) if you want it to fade in
    // open air instead. swirl/nbody/wind off, continuous model (useShells/useRain both 0).
    {
        {
            //  x     y     angle    spread baseSpd  r     g     b    life
            {0.0f, -0.7f, 1.5708f, 0.35f, 0.25f, 0.55f, 0.55f, 0.6f, 8.0f}, // slow gray upward vent
        },
        1,      // numEmitters
        -0.20f, // gravity       -- negative: gentle buoyant rise (softer than fire's -0.5)
        0.0f,   // nbodyStrength -- off
        0.0f,   // swirl         -- off
        0.996f, // damping       -- light drag: the plume slows as it rises
        0,      // useShells     -- continuous model
        0,      // useRain
        0,      // useFlow
        0.0f,   // wind          -- off (add a touch for a leaning column)
        0.3f,   // turbulence    -- random horizontal walk -> widening, curling diffusion
        0.0f,   // curl          -- flow field off (smoke uses white-noise turbulence, not a coherent field)
        0,      // useAttractor  -- Lorenz off
    },
    // ---- 6: curl-noise -- a screen-filling cloud advected by a smooth flow field. ----
    // Particles are seeded EVERYWHERE (spawn_scatter, useFlow=1) so the per-position flow
    // field is visible, then pushed by force #6 (the curl of psi). gravity/swirl/nbody are
    // all off -- the flow field IS the motion. Strong damping (0.75) kills momentum so
    // velocity tracks the field each frame -> clean streamlines instead of inertial smear.
    // The eddies are made VISIBLE by coloring each particle by its velocity DIRECTION (see
    // the useFlow branch in update_kernel's vertex write) -- density stays uniform (the field
    // is divergence-free), so the swirls only show up through direction-hue, not dot spacing.
    // The emitter row's color is thus unused here; only its life is read. params.time
    // (accumulated in update()) drifts the field so the flow never repeats.
    {
        {
            //  x    y   angle spread baseSpd  r    g    b   life
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3, 0.8, 0.9, 8.6f}, // scatter ignores pos/aim; row gives color + life
        },
        1,     // numEmitters
        0.0f,  // gravity        -- off (the flow field drives all motion)
        0.0f,  // nbodyStrength  -- off
        0.0f,  // swirl          -- off (the curl field provides the swirl)
        0.75f, // damping        -- strong: velocity tracks the field (near-advection)
        0,     // useShells
        0,     // useRain
        1,     // useFlow        -- ON: births via spawn_scatter (seed the whole screen)
        0.0f,  // wind
        0.0f,  // turbulence
        1.2f,  // curl           -- flow-field strength (turns force #6 on)
        0,     // useAttractor   -- Lorenz off
    },
    // ---- 7: strange attractor -- Lorenz butterfly, a VELOCITY-FIELD (not force) model. ----
    // The particle's (x,y,z) hold the Lorenz STATE, not screen coords. update_kernel's
    // useAttractor branch reads velocity straight from the Lorenz ODE (lorenz()), integrates
    // the 3D state, and projects (x,z) to the screen -> the twin butterfly wings. Every force
    // knob is OFF -- the field alone drives the motion; this preset just flips useAttractor on.
    // The emitter row is read only for base COLOR + life; spawn_attractor sets the position.
    {
        {
            //  x    y   angle spread baseSpd  r    g    b   life
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 0.8f, 1.0f, 12.0f}, // scatter seeds ignore pos/aim; row gives color + life
        },
        1,    // numEmitters
        0.0f, // gravity        -- off (the Lorenz field IS the motion)
        0.0f, // nbodyStrength  -- off
        0.0f, // swirl          -- off
        1.0f, // damping        -- 1.0 frictionless: velocity IS the field, no drag wanted
        0,    // useShells
        0,    // useRain
        0,    // useFlow
        0.0f, // wind
        0.0f, // turbulence
        0.0f, // curl
        1,    // useAttractor   -- ON: the whole reason this preset exists
    }};

static const int numPresets = sizeof(presets) / sizeof(presets[0]);

// ===========================================================================
// set_preset  --  switch the whole look to preset i.  (ctor boot + number keys)
// ===========================================================================
// Clamp i into range, then (1) re-upload that preset's emitter table into
// __constant__ memory via upload_emitter (which also sets params_.numEmitters),
// and (2) copy its physics knobs (gravity/nbodyStrength/swirl/damping) + the
// useShells / useRain mode flags into params_ so the next update_kernel
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
    params_.useRain = pr.useRain;
    params_.wind = pr.wind;
    params_.turbulence = pr.turbulence;
    params_.curl = pr.curl;
    params_.useFlow = pr.useFlow;
    params_.useAttractor = pr.useAttractor;
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
// psi  --  a smooth scalar "potential" field: one number at every (x, y, t).  (L6 curl-noise)
// ===========================================================================
// The raw material for the curl-noise flow field (style #6). Think of it as a
// time-varying terrain HEIGHT map: one value per point, smooth everywhere. 6b takes
// its curl to get a divergence-free flow vector -- particles slide along psi's
// contour lines (never up/down them), so they swirl without ever bunching up.
//   * SMOOTH across space  -> neighbors read near values -> they flow together
//     (this is the coherent field that white-noise turbulence, in smoke, is not).
//   * DRIFTS over time      -> the +/- t phase morphs the terrain, so the eddies
//     keep reorganizing and the flow never repeats.
// Built from two "octaves" (sine-wave layers): the first is the big, coarse swell;
// the second, at 2x frequency and 0.5x amplitude, adds finer detail on top. F is the
// base spatial frequency = eddy size (bigger F -> smaller, tighter eddies). This is a
// cheap analytic stand-in for Perlin/simplex noise -- swappable later without
// touching 6b, since the curl there samples psi as a black box.
// ===========================================================================
__device__ inline float psi(float x, float y, float t)
{
    float F = 6.12f;                                                 // base spatial frequency (eddy size); larger = smaller eddies
    return sinf(F * x + t) * cosf(F * y - t)                         // octave 1: coarse swell
           + 0.5f * sinf(2.0f * F * y - t) * cosf(2.0f * F * x + t); // octave 2: finer detail (2x freq, 0.5x amp)
}

// ===========================================================================
// curl_noise  --  the divergence-free flow vector at (x, y, t).  (L6 curl-noise, 6b)
// ===========================================================================
// Takes the 2D CURL of the scalar potential psi: rotate its gradient 90 degrees so
// the flow runs ALONG psi's contour lines, never up/down them -> particles swirl but
// never pile up (divergence = 0, guaranteed by construction: div of a curl is the
// difference of the two equal mixed partials, which cancels for ANY smooth psi).
// The partials are estimated by CENTRAL FINITE DIFFERENCES (sample psi a tiny step
// eps to each side, rise / run), so psi stays a black box -- swap in Perlin noise
// later and this code is unchanged. Two outputs, written back through the fx/fy
// POINTERS the caller supplies (the void-return "return two values" idiom).
// ===========================================================================
__device__ inline void curl_noise(float x, float y, float t, float *fx, float *fy)
{
    float eps = 0.01f;                                                       // finite-difference step: small enough to be local, big enough to dodge float cancellation
    float dpsi_dx = ((psi(x + eps, y, t) - psi(x - eps, y, t))) / (2 * eps); // d(psi)/dx: right minus left, over the span
    float dpsi_dy = ((psi(x, y + eps, t) - psi(x, y - eps, t))) / (2 * eps); // d(psi)/dy: up minus down
    *fx = dpsi_dy;                                                           //  along-contour x-component  =  d(psi)/dy
    *fy = -dpsi_dx;                                                          //  along-contour y-component  = -d(psi)/dx  (the 90-degree rotation of the gradient)
}

// ===========================================================================
// lorenz  --  the Lorenz strange-attractor VELOCITY field.  (L6 #7, 7a)
// ===========================================================================
// A velocity-field model, NOT a force model: this returns the velocity DIRECTLY
// as a function of the 3D state (x,y,z) -- dp/dt = f(p) -- so update_kernel
// integrates position in ONE step (x += f*dt), with no acceleration and no
// stored velocity. Chaotic (the butterfly) at sigma=10, rho=28, beta=8/3; it
// NEEDS 3 dimensions because a 2D autonomous flow can't be chaotic (its
// trajectories can't cross, so Poincare-Bendixson traps them onto a limit cycle).
// Three outputs written back through the dx/dy/dz pointers (same idiom as curl_noise).
// beta = 8/3, written 2.6667f (NOT 8/3, which is integer division = 2).
// ===========================================================================
__device__ inline void lorenz(float x, float y, float z, float *dx, float *dy, float *dz)
{
    float sigma = 10.0f, rho = 28.0f, beta = 2.6667f;
    *dx = sigma * (y - x);   // x is pulled toward y
    *dy = x * (rho - z) - y; // z feeds back on y (the xz nonlinearity = chaos source)
    *dz = x * y - beta * z;  // xy pumps energy in, -beta*z dissipates it
}
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
// spawn_rain  --  born particle i as a raindrop.  (useRain)
// ===========================================================================
// The rain-mode counterpart of spawn(): a drop is born at a RANDOM x across the
// full width, up at the top of the screen (y = 1). A random `depth` (0 far, 1 near)
// fakes parallax: near drops fall FASTER and are BRIGHTER, far ones slower/dimmer --
// vy and the color's brightness both derive from that one draw. The base hue comes
// from d_emitters[0] (the single rain emitter row), so it stays data-driven. life is
// set to a HUGE sentinel here so the fall never times out -- the drop is instead
// recycled by update_kernel when it LANDS on the floor and finishes its puddle dwell.
// ===========================================================================
__device__ inline void spawn_rain(ParticleSoA p, int i,
                                  curandState *st)
{
    p.x[i] = (curand_uniform(st) - 0.5f) * 2.0f;
    p.y[i] = 1.0f;
    p.vx[i] = 0.0f;
    float depth = curand_uniform(st);
    p.vy[i] = -(0.6f + 1.8f * depth);
    float b = 0.4f + 0.6f * depth;
    p.cr[i] = d_emitters[0].r * b;
    p.cg[i] = d_emitters[0].g * b;
    p.cb[i] = d_emitters[0].b * b;
    p.life[i] = 999.9f;
}

__device__ inline void spawn_scatter(ParticleSoA p, int i, curandState *st)
{
    p.x[i] = (curand_uniform(st) - 0.5f) * 2.0f;
    p.y[i] = (curand_uniform(st) - 0.5f) * 2.0f;
    p.vx[i] = 0.0f;
    p.vy[i] = 0.0f;
    p.cr[i] = d_emitters[0].r;
    p.cb[i] = d_emitters[0].b;
    p.cg[i] = d_emitters[0].g;
    p.life[i] = d_emitters[0].lifetime * (0.7f + 0.3f * curand_uniform(st));
}

// ===========================================================================
// spawn_attractor  --  seed particle i somewhere in Lorenz STATE space.  (L6 #7)
// ===========================================================================
// The attractor counterpart of spawn_scatter(): (x,y,z) here are the 3D Lorenz
// STATE, not screen coords, so we scatter across the attractor's own range
// (roughly x,y in +/-20, z in 0..50). Chaos (sensitive dependence) then smears
// these seeds across the whole butterfly within a second or two. Velocity is
// left 0 -- it isn't stored state here; update_kernel reads it fresh from
// lorenz() each frame. Color/life come from the single emitter row, like scatter.
// ===========================================================================
__device__ inline void spawn_attractor(ParticleSoA p, int i, curandState *st)
{
    p.x[i] = (curand_uniform(st) - 0.5f) * 40.0f;
    p.y[i] = (curand_uniform(st) - 0.5f) * 40.0f;
    p.z[i] = curand_uniform(st) * 50.0f;
    p.vx[i] = 0.0f;
    p.vy[i] = 0.0f;
    p.cr[i] = d_emitters[0].r;
    p.cb[i] = d_emitters[0].b;
    p.cg[i] = d_emitters[0].g;
    p.life[i] = d_emitters[0].lifetime * (0.7f + 0.3f * curand_uniform(st));
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
// One GPU thread per particle: sum four forces (gravity + central attractor +
// swirl + wind) into an acceleration, integrate velocity, apply air drag (L6-1), move
// the position, bounce off the 4 walls, then age/respawn -- which now BRANCHES on
// params.useShells (L6-2): shell presets (fireworks) respawn a particle only when
// its shell relaunches (shell.launch) and hide it while its shell is dark; other
// presets keep the L5 continuous model (respawn the moment life hits 0), where
// useRain further picks spawn_rain (random x at the top) over spawn (emitter). Finally
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
    // --- Strange attractor (#7): a VELOCITY-FIELD path -- handled FIRST, then RETURN --
    // Every style BELOW is a force model (sum accelerations -> integrate v -> move).
    // This one is not: the Lorenz ODE gives velocity DIRECTLY as a function of the 3D
    // state, dp/dt = f(p), so there is no (ax,ay) accumulator, no stored velocity, no
    // drag. Crucially, (x,y,z) here hold the Lorenz STATE in its OWN range (x,y ~ +/-20,
    // z ~ 0..50), NOT screen coordinates -- so this state must never reach the
    // wall-clamp / drag / fade code below, all of which assume p.x/p.y ARE the on-screen
    // position in [-1,1] (the wall clamp would pin the +/-20 state into a 1x1 box and
    // destroy the butterfly). A single early return keeps the raw state clear of that
    // whole pipeline; we project it to the screen ourselves at the end.
    if (params.useAttractor)
    {
        // The 3D Lorenz STATE (not screen coords). Held in registers across the substep
        // loop so we touch global memory only once to read and once to write.
        float x = p.x[i], y = p.y[i], z = p.z[i];

        // Integrate dp/dt = lorenz(p) with forward Euler, SUBSTEPPED for stability.
        // We advance dt*timeScale of "Lorenz time" this frame -- scaling by the REAL dt
        // makes the evolution speed frame-rate independent (timeScale is the visual-speed
        // knob). One big Euler step of that whole span would extrapolate the large,
        // fast-changing Lorenz velocity too far and diverge to NaN (the butterfly
        // vanishes); splitting it into `sub` small steps of size h, RE-SAMPLING lorenz()
        // at the new point each step, keeps every jump tiny so the straight segments hug
        // the curved trajectory.
        float timeScale = 1.0f; // Lorenz-time advanced per real second = how fast the butterfly evolves
        // ADAPTIVE substep count: pick JUST enough steps that each one is <= 0.006 (the
        // forward-Euler stability/accuracy ceiling for this field), and no more.
        // dt*timeScale/0.006 = the exact fractional steps needed; ceilf rounds UP so h can
        // never exceed 0.006; max(1,..) floors it at one step (guards a dt that rounds to 0).
        // So h auto-tracks ~0.006 at ANY frame rate: ~1 step at high FPS (tiny dt, no wasted
        // work), more steps when a slow frame would otherwise take a dangerously big jump.
        int sub = max(1, (int)ceilf(dt * timeScale / 0.006f));
        float h = dt * timeScale / sub; // one substep's size (<= 0.006 by construction)
        for (int k = 0; k < sub; k++)
        {
            float dx, dy, dz;
            lorenz(x, y, z, &dx, &dy, &dz); // fresh velocity at the CURRENT point
            x += dx * h;                    // forward Euler on each of the 3 state coords
            y += dy * h;
            z += dz * h;
        }

        // Write the evolved state back into the SoA arrays for next frame's read.
        p.x[i] = x;
        p.y[i] = y;
        p.z[i] = z;

        // Age + reseed. Chaos collapses every seed onto the SAME butterfly manifold, so
        // without rebirth the cloud would slowly stop looking alive; spawn_attractor
        // scatters a dead particle back across the whole state space, and fresh points
        // spiral back in -- keeping the motion perpetually varied.
        p.life[i] -= dt;
        if (p.life[i] <= 0)
        {
            spawn_attractor(p, i, &rng[i]);
            // Refresh the register copies: spawn_attractor just wrote a NEW seed into
            // p.x/y/z, but the projection below reads x/z -- without this it would draw the
            // stale pre-death position for one frame (the "emit the vertex AFTER respawn,
            // from fresh state" rule, the same ordering the continuous model relies on).
            x = p.x[i];
            y = p.y[i];
            z = p.z[i];
        }

        // PROJECT the 3D state onto the 2D screen: draw the (x, z) plane -- the plane the
        // butterfly's two wings live in -- and drop y (it becomes the depth we look
        // through; y is still fully simulated above, just not drawn). x is already
        // centered on 0 so it only scales; z spans 0..50, so shift by -25 to center it
        // first. /25 maps each ~+/-25 half-range into [-1,1] with a little margin.
        float sx = x / 25.0f, sy = (z - 25.0f) / 25.0f;
        float *v = &vbo[i * 5]; // this particle's 5-float slice [x,y,r,g,b] of the VBO
        v[0] = sx;              // screen x  <- state x
        v[1] = sy;              // screen y  <- state z
        v[2] = p.cr[i];         // flat emitter hue for now (coloring by speed/z is 7d)
        v[3] = p.cg[i];
        v[4] = p.cb[i];
        return; // done -- skip the entire force model below
    }
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

    // 4) Wind (rain slant): a constant horizontal push. 0 for every non-rain preset,
    //    so no useRain guard is needed -- adding 0 is a no-op. It's an ACCELERATION
    //    (not a one-off initial velocity), so drag can't erase it: drops settle into a
    //    slanted terminal fall instead of drifting straight again.
    ax += params.wind;

    // 5) Turbulence (smoke): a fresh random HORIZONTAL kick every frame. Each frame
    //    draws (curand_uniform - 0.5) in [-0.5,0.5) -> a symmetric +/- push (mean 0,
    //    unlike wind's fixed direction), scaled by turbulence. 0 for every non-smoke
    //    preset, so like wind it needs no branch -- adding 0 is a no-op. Integrated
    //    into vx below, these independent kicks make the horizontal velocity a RANDOM
    //    WALK (Brownian motion): no preferred direction, but the sideways SPREAD grows
    //    ~sqrt(time), so the rising column widens and curls into a diffusing plume
    //    instead of a straight jet. Drag (below) keeps the walk from running away.
    //    Draws from this particle's OWN curandState, so no two smoke trails wander
    //    alike. Only ax here -- buoyancy (negative gravity) already owns the rise.
    //    Frame-rate independence (the `/ sqrtf(dt)`): a random walk's spread grows as
    //    sqrt(number of steps). Over a fixed second there are 1/dt steps each of size
    //    ~dt, so a CONSTANT coefficient gives spread ~ sqrt(1/dt)*dt = sqrt(dt) -> LESS
    //    diffusion at higher FPS (same class of bug as raw dt or a raw v*=damping).
    //    Dividing by sqrtf(dt) makes each kick ~sqrt(dt), cancelling it so the spread
    //    tracks sqrt(time) only -- identical diffusion at any frame rate. (dt is the
    //    real frame time, clamped to (0, 0.05] upstream, so it is never zero here.)
    ax += (curand_uniform(&rng[i]) - 0.5f) * params.turbulence / sqrtf(dt);

    // 6) Curl-noise flow field (style #6): a smooth, divergence-free vector field
    //    sampled at THIS particle's own position (px, py). Because the field is smooth,
    //    neighbors read nearby vectors and swirl together in coherent eddies -- the
    //    coherent-field upgrade of smoke's per-particle white-noise kick (force 5). It's
    //    divergence-free (built as the curl of a scalar potential), so particles never
    //    pile up or leave holes -- they just flow. params.time (accumulated per frame in
    //    update()) drifts the field so the eddies keep reorganizing and never repeat.
    //    Gated by params.curl (0 for every other preset -> adding 0 is a no-op, no
    //    branch needed, exactly like wind and turbulence). Added as an acceleration into
    //    the accumulator; strong damping in this preset makes velocity track the field
    //    (near-advection). A true velocity-field model (overwrite v, not add to a) is
    //    the strange attractor's job (#7).
    float fx, fy;
    curl_noise(px, py, params.time, &fx, &fy); // sample the flow vector here; written into fx/fy
    ax += fx * params.curl;
    ay += fy * params.curl;

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

    // How a particle meets the world edge now depends on the mode:
    //  - non-rain: bounce off all 4 walls -- clamp to the wall, flip velocity, keep
    //    `restitution` of the speed (clamp value must match the wall tested).
    //  - rain: no bounce. When a drop reaches the FLOOR it lands and stays there as a
    //    puddle (pinned, velocity zeroed) until it respawns up top. See the else below.
    if (!params.useRain)
    {
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
    }
    else // rain: land on the floor and puddle instead of bouncing
    {
        if (p.y[i] < -params.bound) // reached the floor -> pin the drop as a puddle
        {
            // Pin to the floor EVERY frame: gravity re-pulls the drop under the bound
            // each step, so re-clamping here holds it flat instead of letting it sink.
            p.y[i] = -params.bound;
            p.vx[i] = 0.0f;
            p.vy[i] = 0.0f;

            // The landing itself happens only ONCE. While falling, life is a huge
            // sentinel (set by spawn_rain), so life > t is true only on the FIRST frame
            // on the floor: start the puddle's dwell countdown (life = t) and give the
            // drop a tiny one-time horizontal splash. On later puddle frames life <= t,
            // so this block is skipped and the drop just sits (aging in the branch below).
            float t = 3.0f; // puddle dwell: seconds a drop lingers before it respawns
            if (p.life[i] > t)
            {
                p.life[i] = t;
                p.x[i] += (curand_uniform(&rng[i]) - 0.5) * 0.05f;
            }
        }
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
            // Same aging/gate for both looks; only the birth recipe differs:
            // rain drops are reborn at a random x up top (spawn_rain), everything
            // else at its emitter mouth (spawn).
            if (params.useRain)
            {
                spawn_rain(p, i, &rng[i]);
            }
            else if (params.useFlow)
            {
                spawn_scatter(p, i, &rng[i]);
            }

            else
            {
                spawn(p, i, params.numEmitters, &rng[i]);
            }
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
        v[0] = p.x[i]; // attribute 0 (pos.x) -- shader reads offset 0
        v[1] = p.y[i]; // attribute 0 (pos.y)
        if (params.useFlow)
        {
            // FLOW COLORING (显影): color by velocity DIRECTION so the otherwise-invisible
            // divergence-free field shows up. The field keeps density uniform (nothing to
            // see from dot positions), but particles in the same eddy move alike -> share a
            // hue -> the eddy reads as a rotating color patch. atan2(vy, vx) is the movement
            // angle (-pi..pi); three sines 120 deg apart map it onto a seamless rainbow wheel
            // (R/G/B peak at different angles, so each direction gets its own vivid hue).
            float ang = atan2f(p.vy[i], p.vx[i]);             // movement direction (note: y first, then x)
            v[2] = (0.5f + 0.5f * sinf(ang)) * fade;          // R
            v[3] = (0.5f + 0.5f * sinf(ang + 2.094f)) * fade; // G, phase +120 deg (2pi/3)
            v[4] = (0.5f + 0.5f * sinf(ang + 4.189f)) * fade; // B, phase +240 deg (4pi/3)
        }
        else
        {
            v[2] = p.cr[i] * fade; // attribute 1 (color.r) -- shader reads offset 8
            v[3] = p.cg[i] * fade; // attribute 1 (color.g)
            v[4] = p.cb[i] * fade; // attribute 1 (color.b)
        }
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
    CUDA_CHECK(cudaMalloc(&d_.z, bytes)); // L6 #7: the Lorenz attractor's 3rd coordinate
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
    // L6 #7: zero the z array up front. z is only used by the Lorenz attractor, but the
    // sim boots as fireworks; when the user switches TO attractor, particles still carrying
    // a positive life from the old look run Lorenz integration before their first respawn.
    // If z were uninitialized device garbage, that integration could blow up to NaN/inf.
    // z=0 is a safe Lorenz start (spirals out naturally); spawn_attractor seeds it properly
    // on respawn. cudaMemset fills BYTES, and 0.0f's bit pattern is all-zero, so this is 0.0f.
    CUDA_CHECK(cudaMemset(d_.z, 0, bytes));

    // Boot the default look (preset 1, fireworks). set_preset uploads its emitter table
    // into __constant__ memory and sets params_.numEmitters -- so it MUST run before
    // init_kernel below, which reads both to build the initial 1M population.
    set_preset(1);
    int block = 256, grid = (n_ + block - 1) / block;
    init_rng<<<grid, block>>>((curandState *)d_rng_, n_, 1025ULL);
    CUDA_CHECK(cudaGetLastError());
    init_kernel<<<grid, block>>>(d_, n_, params_.numEmitters, (curandState *)d_rng_); // params_.numEmitters was set by set_preset(1) -> upload_emitter above
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
    cudaFree(d_.z);         // L6 #7: Lorenz attractor's 3rd coordinate
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
    params_.time += dt;
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
