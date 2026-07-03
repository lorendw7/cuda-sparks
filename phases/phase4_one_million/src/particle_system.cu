#include "particle_system.h"
#include "cuda_utils.h" // CUDA_CHECK macro + gpu_info()

#include <cuda_runtime.h>
#include <cmath> // floorf

// ===========================================================================
// spawn  --  build one particle from its index i.  (host + device)
// ===========================================================================
// __host__ __device__: compiled for BOTH sides. The constructor calls it on the
// CPU for the initial fill; update_kernel calls it on the GPU when a particle
// dies and must be recycled. No curand in L1 -- randomness is faked from i.
// ===========================================================================
__host__ __device__ inline void spawn(Particle &p, int i, float bound)
{
    // Three pseudo-random values in [0,1): index * irrational, keep the fraction.
    float f1 = i * 0.6180340f; // golden ratio
    f1 -= floorf(f1);          // fractional part -> [0,1)
    float f2 = i * 0.7548777f;
    f2 -= floorf(f2);
    float f3 = i * 0.9541230f;
    f3 -= floorf(f3);

    p.x = (f1 * 2.0f - 1.0f) * bound; // map [0,1)->[-bound,bound): scatter across x
    p.y = (f2 * 2.0f - 1.0f) * bound; // scatter across y
    p.vx = 0.0f;                      // start at rest; gravity takes over
    p.vy = 0.0f;
    p.r = 0.01f; // radius (unused this level, placeholder)

    p.cr = 0.3f + 0.7f * f1; // per-particle color from index, so the field isn't flat
    p.cg = 0.3f + 0.7f * f2;
    p.cb = 1.0f;

    p.life = 1.0f + f3 * 3.0f; // staggered lifetime 1..4s -> not all die on one frame
}

// ===========================================================================
// update_kernel  --  advance one particle per thread.  (runs every frame)
// ===========================================================================
// One GPU thread per particle: apply gravity, integrate position, bounce off
// the 4 walls, age, and recycle the dead. params is passed BY VALUE (a copy per
// thread) -- use params.gravity (dot), never params_ (that member is invisible here).
// ===========================================================================
__global__ void update_kernel(Particle *particles, int n, SimParams params, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread id = particle index
    if (i >= n)                                    // grid usually rounds up past n:
        return;                                    // extra threads must do nothing

    particles[i].vy -= params.gravity * dt; // gravity changes vertical velocity

    particles[i].x += particles[i].vx * dt; // integrate position from velocity
    particles[i].y += particles[i].vy * dt;

    // Bounce off each of the 4 walls: clamp to the wall, flip velocity, scale by
    // restitution (energy kept per bounce). Clamp value must match the wall tested.
    if (particles[i].x > params.bound)
    {
        particles[i].x = params.bound;
        particles[i].vx = -particles[i].vx * params.restitution;
    }
    if (particles[i].x < -params.bound)
    {
        particles[i].x = -params.bound;
        particles[i].vx = -particles[i].vx * params.restitution;
    }
    if (particles[i].y > params.bound)
    {
        particles[i].y = params.bound;
        particles[i].vy = -particles[i].vy * params.restitution;
    }
    if (particles[i].y < -params.bound)
    {
        particles[i].y = -params.bound;
        particles[i].vy = -particles[i].vy * params.restitution;
    }

    particles[i].life -= dt;         // age this particle
    if (particles[i].life <= 0.0f)   // dead? recycle it in place (deterministic)
    {
        spawn(particles[i], i, params.bound);
    }
}

// ===========================================================================
// Constructor  --  allocate device memory, build initial state, upload it.
// ===========================================================================
ParticleSystem::ParticleSystem(const SimParams& p) // definition of the ctor declared in the header
    : params_(p), n_(p.n)                          // member-init list: fill params_ and n_ first
{
    gpu_info(); // print which GPU we're on (from cuda_utils.h)

    // Allocate the particle array in GPU memory. (size_t) before the multiply so
    // the byte count can't overflow int as the particle count grows.
    CUDA_CHECK(cudaMalloc(&d_particles_, (size_t)n_ * sizeof(Particle)));

    // Build the starting state on the CPU, one spawn() per particle.
    host_.resize(n_);
    for (int i = 0; i < n_; ++i)
    {
        spawn(host_[i], i, params_.bound);
    }

    // Upload the initial array to the GPU (Host -> Device).
    CUDA_CHECK(cudaMemcpy(d_particles_, host_.data(),
                          (size_t)n_ * sizeof(Particle),
                          cudaMemcpyHostToDevice)); // direction flag: source is host, dest is device
}

// Destructor: every cudaMalloc needs a matching cudaFree.
ParticleSystem::~ParticleSystem()
{
    cudaFree(d_particles_);
}

// ===========================================================================
// update  --  launch the kernel, then copy the whole array back to the CPU.
// ===========================================================================
void ParticleSystem::update(float dt) // returns nothing; its output is the mutated device + host state
{
    int block = 256;                      // threads per block
    int grid = (n_ + block - 1) / block;  // round up so every particle is covered

    update_kernel<<<grid, block>>>(d_particles_, n_, params_, dt); // launch grid*block threads on the GPU
    CUDA_CHECK(cudaGetLastError());                               // check the launch itself

    // Copy results Device -> Host for rendering. THIS per-frame round trip is the
    // bottleneck L1 measures and L2 removes.
    CUDA_CHECK(cudaMemcpy(host_.data(), d_particles_,
                          (size_t)n_ * sizeof(Particle), // total bytes = count * size of one Particle
                          cudaMemcpyDeviceToHost));
}

// ===========================================================================
// to_vertices  --  pack host_ into interleaved [x,y,r,g,b] vertices.
// ===========================================================================
// const: reads our members, writes only to the out-parameter. Definition must
// repeat the 'const' from the header declaration or it's a different signature.
void ParticleSystem::to_vertices(std::vector<float> &out) const // 'const' must match the header declaration
{
    out.resize((size_t)n_ * 5); // 5 floats per particle: x, y, r, g, b
    for (int i = 0; i < n_; ++i)
    {
        const Particle &p = host_[i];
        float t = p.life; // fade factor: dim toward death
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f)
        {
            t = 1.0f;
        }
        float *v = &out[(size_t)i * 5]; // start of this particle's 5 floats
        v[0] = p.x;
        v[1] = p.y;
        v[2] = p.cr * t; // color scaled by remaining life
        v[3] = p.cg * t;
        v[4] = p.cb * t;
    }
}
