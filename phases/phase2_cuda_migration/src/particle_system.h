#pragma once
#include "particles.h"     // Particle, SimParams  (shared include/ folder)
#include <vector>

// ===========================================================================
// ParticleSystem (Phase 2)  --  the GPU simulation.
// ===========================================================================
// Notice this header looks almost identical to Phase 1's: same methods, same
// names. main.cpp does not change at all. What changed is INSIDE -- the
// particles now live in GPU memory and update() runs a CUDA kernel.
//
// This file is plain C++ (no CUDA syntax) so the host compiler can read it.
// The actual CUDA code lives in particle_system.cu.
//
// The class manages TWO copies of the particle array:
//   - d_particles_ : on the GPU (device). The kernel works on this one.
//   - host_        : a CPU-side copy we fill each frame so the renderer can draw.
// ===========================================================================
class ParticleSystem {
public:
    explicit ParticleSystem(const SimParams& p);   // allocates GPU memory
    ~ParticleSystem();                             // frees GPU memory

    int  size() const { return n_; }

    // Advance the whole system by dt: launch the kernel, then copy results back.
    void update(float dt);

    // Pack the latest particle data into interleaved [x,y,r,g,b] vertices.
    void to_vertices(std::vector<float>& out) const;

private:
    SimParams              params_;
    int                    n_ = 0;
    Particle*              d_particles_ = nullptr;   // GPU pointer (device memory)
    std::vector<Particle>  host_;                    // CPU mirror for rendering
    void*                  d_rng_ = nullptr;         // device curandState array (one RNG
                                                     // per particle). Kept as void* so this
                                                     // host-readable header needs no CUDA
                                                     // headers; the .cu casts it to curandState*.
};
