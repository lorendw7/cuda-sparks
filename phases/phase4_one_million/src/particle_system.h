#pragma once
#include "particles.h" // Particle, SimParams (shared include/ folder)

// ===========================================================================
// ParticleSystem (Phase 4, L5)  --  fast 1M-particle GPU simulation, SoA layout.
// ===========================================================================
// Particles live entirely in GPU memory as a Structure of Arrays (one device
// array per field, see ParticleSoA). update() maps the OpenGL VBO, runs one
// kernel that both advances the physics and writes vertices straight into the
// VBO, then unmaps -- no per-frame CPU round trip (that was L1; L2 removed it).
//
// Effects (L5) are data-driven: an Emitter table in __constant__ memory plus a
// small library of Presets (emitter table + physics knobs) switched at runtime
// via set_preset(). The table and presets live in particle_system.cu.
//
// Plain C++ header (no CUDA <<<>>> syntax) so the host compiler can include it;
// the actual kernels live in particle_system.cu.
// ===========================================================================
// Forward declaration: vbo_resource_ is a POINTER, so the compiler only needs to
// know the name here, not the full type (that lives in <cuda_gl_interop.h>, .cu only).
struct cudaGraphicsResource;

// Structure of Arrays: one device array per particle field. Defined ABOVE the
// class because ParticleSystem holds it BY VALUE (member d_) -- a value member
// needs the complete type/size at that point, unlike the pointer above.
struct ParticleSoA
{
    float *x, *y, *vx, *vy, *life, *cr, *cg, *cb;
};

class ParticleSystem
{
private:
    SimParams params_;                        // simulation config (dt, gravity, bound, n...)
    int n_ = 0;                               // total particle count
    ParticleSoA d_;                           // the 8 device field-arrays (allocated in the ctor)
    cudaGraphicsResource *vbo_resource_ = nullptr; // interop handle for the renderer's VBO
    void *d_rng_ = nullptr;                    // device curandState[n] (one RNG per particle);
                                               // void* so this host header needs no CUDA headers
public:
    // Constructor: cudaMalloc the 8 SoA arrays and fill them on the GPU via
    // init_kernel. explicit forbids an implicit SimParams->ParticleSystem.
    explicit ParticleSystem(const SimParams &p);

    // Destructor: cudaFree all 8 arrays + unregister the VBO (RAII -- automatic).
    ~ParticleSystem();

    // Particle count. const: reads a member, modifies nothing.
    int size() const { return n_; }

    // Advance the whole system by dt: map the VBO, launch the kernel (physics +
    // vertex write), unmap. Not const -- it mutates device state and the VBO.
    void update(float dt);

    // One-time: register the renderer's VBO with CUDA so the kernel can write it.
    void register_vbo(unsigned int vbo_id);

    // Push an emitter table into __constant__ memory (cudaMemcpyToSymbol) and set
    // params_.numEmitters. Called by the ctor and by set_preset on every switch.
    void upload_emitter(const Emitter*, int count);

    // Switch to effect preset i (clamped): re-upload its emitters + copy its physics
    // knobs into params_. Cheap -- no realloc; the look fades in as particles recycle.
    // The number keys in main.cpp call this at runtime.
    void set_preset(int i);
};
