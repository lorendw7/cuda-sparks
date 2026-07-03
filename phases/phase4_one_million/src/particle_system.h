#pragma once
#include "particles.h" // Particle, SimParams (shared include/ folder)
#include <vector>

// ===========================================================================
// ParticleSystem (Phase 4, Level 1)  --  naive 1M-particle GPU simulation.
// ===========================================================================
// This is the "slow but correct" baseline. The particles live in GPU memory;
// update() runs a CUDA kernel, then copies the whole array back to the CPU so
// the renderer can pack and draw it. That per-frame GPU->CPU->GPU round trip is
// exactly the bottleneck L1 measures and L2 (CUDA-GL interop) tears out.
//
// Plain C++ header (no CUDA syntax) so the host compiler can include it; the
// actual kernels live in particle_system.cu.
// ===========================================================================
struct cudaGraphicsResource;

class ParticleSystem // NOTE: rename to ParticleSystem to match project convention
{
private:
    SimParams params_;                // simulation config (dt, gravity, bound, n...)
    int n_ = 0;                       // total particle count
    Particle *d_particles_ = nullptr; // GPU pointer (device memory) -- the kernel works on this
    std::vector<Particle> host_;      // CPU mirror, refilled every frame for rendering
    cudaGraphicsResource *vbo_resource_ = nullptr;

public:
    // Constructor: cudaMalloc the device array, build the initial state on the
    // CPU, and upload it to the GPU. explicit forbids implicit SimParams->System.
    explicit ParticleSystem(const SimParams &p);

    // Destructor: cudaFree the device array (RAII -- no manual cleanup at call site).
    ~ParticleSystem();

    // Particle count. const: reads a member, modifies nothing.
    int size() const { return n_; }; // BUG: missing ';' -> should be { return n_; }

    // Advance the whole system by dt: launch the kernel, then cudaMemcpy the
    // results back into host_. Not const -- it mutates device + host state.
    void update(float dt);

    // Pack the latest host_ data into interleaved [x,y,r,g,b] vertices in out.
    // const: only reads our members; writes solely to the out-parameter.
    void to_vertices(std::vector<float> &out) const;

    void register_vbo(unsigned int vbo_id);
};
