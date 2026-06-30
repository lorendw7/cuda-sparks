#pragma once

// ---------------------------------------------------------------------------
// Particle  --  one element in the simulation
// ---------------------------------------------------------------------------
// Memory layout: Array of Structures (AoS)
//   All fields for particle i live at particles[i].x, particles[i].y, etc.
//   Simple to index; every kernel sees the same layout.
//   Lesson 5 discusses when SoA (Structure of Arrays) can be faster.
// ---------------------------------------------------------------------------
struct Particle {
    float x,  y;   // position   (world coordinates, typical range [-1, 1])
    float vx, vy;  // velocity   (world units per second)
    float r;        // radius     (used for display and collision)
    float life;     // lifetime   [1.0 = fully alive, 0.0 = dead]
    float cr, cg, cb;   // birth color, copied from this particle's emitter
};

// ---------------------------------------------------------------------------
// SimParams  --  simulation configuration passed into every physics kernel
// ---------------------------------------------------------------------------
struct SimParams {
    float dt;           // time step in seconds (e.g. 0.016 for 60 fps)
    float gravity;      // downward acceleration (subtracted from vy each step)
    float damping;      // velocity scale per step; 1.0 = no drag, 0.99 = slight drag
    float restitution;  // wall bounce coefficient: 1.0 = perfect elastic, 0.0 = dead stop
    float bound;        // world boundary half-size; positions clamp to [-bound, bound]
    int   n;            // total number of particles
};

// ---------------------------------------------------------------------------
// Emitter  --  one spawn source (Phase 3 Level 4)
// ---------------------------------------------------------------------------
// A small, data-driven recipe for how a particle is born. The simulation keeps
// a table of these in __constant__ GPU memory; each particle is assigned to one
// emitter (i % numEmitters) and spawns/recycles from its settings. Swapping the
// whole table is how Level 5 presets (fireworks / fire / nebula) will work.
// ---------------------------------------------------------------------------
struct Emitter {
    float x, y;        // spawn position (world coordinates)
    float angle;       // central aim direction in radians (0 = +x, pi/2 = up)
    float spread;      // angular randomness around angle (radians); 0 = a tight beam
    float baseSpeed;   // base launch speed (jittered per particle)
    float r, g, b;     // color given to this emitter's particles
    float lifetime;    // how long its particles live (seconds)
};
