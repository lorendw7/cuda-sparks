#include "particle_system.h"
#include "cuda_utils.h"     // CUDA_CHECK macro + gpu_info()  (shared include/ folder)

#include <cuda_runtime.h>
#include <cmath>

// ===========================================================================
// emitter_spawn  --  put a particle back at the fountain mouth.  (HELPER, given)
// ===========================================================================
// __host__ __device__ = callable from BOTH CPU (initial fill) and GPU (kernel).
// The "randomness" is faked from the index i for now -- Level 2 swaps in curand.
// You do not need to edit this.
// ===========================================================================
__host__ __device__ inline void emitter_spawn(Particle& p, int i) {
    float f1 = i * 0.6180340f;  f1 -= floorf(f1);
    float f2 = i * 0.7548777f;  f2 -= floorf(f2);

    float angle = f1 * 6.2831853f;          // 0 .. 2*pi
    float speed = 0.4f + 0.4f * f2;
    p.x    = 0.0f;                            // emitter at screen center
    p.y    = -0.6f;
    p.vx   = cosf(angle) * speed;
    p.vy   = fabsf(sinf(angle)) * speed + 0.4f;  // bias upward
    p.r    = 0.01f;
    p.life = 0.5f + 0.5f * f1;                // 0.5 .. 1.0
}

// ===========================================================================
//  *** YOUR CODE ***  The update kernel -- runs ON THE GPU, once per particle.
// ===========================================================================
// A kernel is a function marked __global__. We launch it across thousands of
// threads; each thread handles ONE particle. Write the whole body:
//
//   1. int i = blockIdx.x * blockDim.x + threadIdx.x;   // which particle am I?
//   2. if (i >= n) return;                              // extra threads do nothing
//   3. physics on particles[i] (your Phase 1 logic):
//        - gravity:  vy -= params.gravity * dt
//        - move:     x += vx*dt ;  y += vy*dt
//        - bounce:   4 walls, e.g.
//            if (particles[i].x >  params.bound) { particles[i].x =  params.bound;
//                                                  particles[i].vx = -particles[i].vx * params.restitution; }
//          ...and the other three walls (x<-bound, y>bound, y<-bound)
//        - age:      life -= dt
//   4. recycle:  if (particles[i].life <= 0.0f) emitter_spawn(particles[i], i);
//
// Note: inside the kernel use params.gravity (a dot), NOT params_.gravity --
// params is the COPY passed in as an argument, not the class member.
// ===========================================================================
__global__ void update_kernel(Particle* particles, int n, SimParams params, float dt) {

    // -- write the kernel body here --
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    particles[i].vy -= params.gravity * dt;
    particles[i].x += particles[i].vx * dt;
    particles[i].y += particles[i].vy * dt;

    if(particles[i].x > params.bound) {
        particles[i].x = params.bound;
        particles[i].vx = -particles[i].vx * params.restitution;
    }
    if(particles[i].x < -params.bound) {
        particles[i].x = -params.bound;
        particles[i].vx = -particles[i].vx * params.restitution;
    }
    if(particles[i].y > params.bound) {
        particles[i].y = params.bound;
        particles[i].vy = -particles[i].vy * params.restitution;
    }
    if(particles[i].y < -params.bound) {
        particles[i].y = -params.bound;
        particles[i].vy = -particles[i].vy * params.restitution;
    }

    particles[i].life -= dt;
    if(particles[i].life <= 0.0f) {
        emitter_spawn(particles[i], i);
    }
}

// ===========================================================================
//  *** YOUR CODE ***  GPU memory management.
// ===========================================================================
// This is the cudaMalloc / cudaMemcpy you came to learn. The CPU and GPU have
// separate memory; we allocate on the GPU, copy data across, and free it.
// ===========================================================================

ParticleSystem::ParticleSystem(const SimParams& p)
    : params_(p), n_(p.n) {
    gpu_info();   // (given) print which GPU we are on

    // ---- TODO A: allocate the particle array IN GPU MEMORY ------------------
    // cudaMalloc takes the ADDRESS of the pointer (&d_particles_) and a size in
    // BYTES. The size of the whole array is n_ * sizeof(Particle).
    // Wrap the call in CUDA_CHECK( ... ) so any error aborts with a message.
    //
    //   CUDA_CHECK(cudaMalloc(&d_particles_, (size_t)n_ * sizeof(Particle)));
    //
    // <write TODO A here>
    CUDA_CHECK(cudaMalloc(&d_particles_, (size_t)n_*sizeof(Particle)));

    // ---- build the starting state on the CPU (given) ------------------------
    host_.resize(n_);
    for (int i = 0; i < n_; ++i) emitter_spawn(host_[i], i);

    // ---- TODO B: copy the array UP to the GPU (Host -> Device) ---------------
    // cudaMemcpy(destination, source, bytes, direction).
    //   source      = host_.data()  (CPU)
    //   bytes       = (size_t)n_ * sizeof(Particle)
    //   direction   = cudaMemcpyHostToDevice
    // Wrap in CUDA_CHECK.
    //
    // <write TODO B here>
    CUDA_CHECK(cudaMemcpy(d_particles_, host_.data(), (size_t)n_ * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
    // ---- TODO C: free the GPU memory you allocated in the constructor -------
    // Every cudaMalloc needs a matching cudaFree, or the GPU leaks memory.
    //   cudaFree(d_particles_);
    //
    // <write TODO C here>
    cudaFree(d_particles_);
}

void ParticleSystem::update(float dt) {
    // ---- TODO D: choose the launch shape ------------------------------------
    // Threads are grouped into blocks. Use 256 threads per block, and enough
    // blocks to cover all n_ particles (round UP so the last few aren't lost):
    //   int block = 256;
    //   int grid  = (n_ + block - 1) / block;
    //
    // <write TODO D here>
    int block = 256;
    int grid = (n_ + block - 1) / block;

    // ---- TODO E: launch the kernel ------------------------------------------
    // The <<<grid, block>>> syntax says how many threads to start. Pass the GPU
    // pointer and the params/dt:
    //   update_kernel<<<grid, block>>>(d_particles_, n_, params_, dt);
    // Then check the launch succeeded:
    //   CUDA_CHECK(cudaGetLastError());
    //
    // <write TODO E here>
    update_kernel<<<grid, block>>>(d_particles_, n_, params_, dt);
    CUDA_CHECK(cudaGetLastError());

    // ---- TODO F: copy results back DOWN to the CPU (Device -> Host) ----------
    // The kernel updated GPU memory; the renderer reads CPU memory. Copy back:
    //   cudaMemcpy(host_.data(), d_particles_, (size_t)n_ * sizeof(Particle),
    //              cudaMemcpyDeviceToHost)
    // Wrap in CUDA_CHECK. (This per-frame copy is the slow part Phase 4 removes
    // with CUDA-OpenGL interop.)
    //
    // <write TODO F here>
    CUDA_CHECK(cudaMemcpy(host_.data(), d_particles_, (size_t)n_ * sizeof(Particle), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// to_vertices  --  pack particles into [x,y,r,g,b] for the renderer.  (given)
// ---------------------------------------------------------------------------
void ParticleSystem::to_vertices(std::vector<float>& out) const {
    out.resize((size_t)n_ * 5);
    for (int i = 0; i < n_; ++i) {
        const Particle& p = host_[i];
        float t = p.life;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float* v = &out[(size_t)i * 5];
        v[0] = p.x;
        v[1] = p.y;
        v[2] = 1.0f;                 // red
        v[3] = 0.4f + 0.6f * t;      // green: bright when young
        v[4] = 0.2f * t;             // blue
    }
}
