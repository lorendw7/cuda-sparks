#include "particle_system.h"
#include "cuda_utils.h" // CUDA_CHECK macro + gpu_info()  (shared include/ folder)

#include <cuda_runtime.h>
#include <curand_kernel.h> // curandState, curand_init, curand_uniform
#include <cmath>

// ===========================================================================
// Emitter table in __constant__ memory  (Level 4)
// ===========================================================================
// A small, read-only table every thread consults to know where/how to spawn a
// particle. __constant__ lives in a special cached GPU region: when a warp's 32
// threads read the same address, the hardware fetches once and broadcasts it.
// It is NOT cudaMalloc'd -- it is a fixed global symbol that already exists on
// the device. The host fills it once with cudaMemcpyToSymbol (Step 5).
// ===========================================================================
#define MAX_EMITTERS 8
__constant__ Emitter d_emitters[MAX_EMITTERS];
__constant__ int d_numEmitters;

// ===========================================================================
// emitter_spawn  --  birth a particle from a given emitter.  (given helper)
// ===========================================================================
// Used only for the one-time initial fill on the CPU (constructor), so it runs
// on the HOST -- which CANNOT read __constant__ d_emitters. That is why the
// emitter is passed in by value (em): the caller hands it a host-side Emitter.
// The randomness here is faked from the index i; at runtime, recycling uses the
// real per-particle RNG (respawn_rng), which reads d_emitters on the device.
// ===========================================================================
__host__ __device__ inline void emitter_spawn(Particle &p, int i, const Emitter &em)
{
    float f1 = i * 0.6180340f;
    f1 -= floorf(f1);
    float f2 = i * 0.7548777f;
    f2 -= floorf(f2);

    float angle = em.angle + (f1 - 0.5f) * em.spread; // 0 .. 2*pi
    float speed = em.baseSpeed * (0.8f + 0.4f * f2);
    p.x = em.x;
    p.y = em.y;
    p.vx = cosf(angle) * speed;
    p.vy = sinf(angle) * speed;
    p.r = 0.01f;
    p.cr = em.r;
    p.cg = em.g;
    p.cb = em.b;
    p.life = em.lifetime;
}

// ===========================================================================
// respawn_rng  --  recycle a particle from emitter e using its real RNG stream.
// ===========================================================================
// Same job as emitter_spawn, but runs on the DEVICE, so it reads the emitter
// straight from __constant__ d_emitters[e] and pulls fresh numbers from THIS
// particle's curandState (st) via curand_uniform, so the spray looks natural
// rather than mechanical. st is already a pointer, so it is passed straight
// through; e is the emitter index the caller picked (i % d_numEmitters).
// ===========================================================================
__device__ inline void respawn_rng(Particle &p, curandState *st, int e)
{
    float u1 = curand_uniform(st);
    float u2 = curand_uniform(st);
    float angle = d_emitters[e].angle + (u1 - 0.5f) * d_emitters[e].spread;
    float speed = d_emitters[e].baseSpeed * (0.8f + 0.4f * u2) + 0.4f;
    p.x = d_emitters[e].x;
    p.y = d_emitters[e].y;
    p.vx = cosf(angle) * speed;
    p.vy = sinf(angle) * speed;

    p.r = 0.01f;
    p.cr = d_emitters[e].r;
    p.cg = d_emitters[e].g;
    p.cb = d_emitters[e].b;

    p.life = d_emitters[e].lifetime;
}

// ===========================================================================
// init_rng_kernel  --  seed one independent RNG stream per particle. (runs once)
// ===========================================================================
// Launched once at startup, one thread per particle. Passing the particle index
// i as curand_init's "sequence" argument gives every particle a different,
// independent random stream from the same base seed.
// ===========================================================================
__global__ void init_rng_kernel(curandState *states, int n, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    curand_init(seed, i, 0, &states[i]);
}

// ===========================================================================
// nbody_force_tiled  --  O(n^2) gravity, but read through __shared__ memory.
// ===========================================================================
// Same math as the naive version: thread i sums the pull of every particle j,
// magnitude 1/dist^2 along the unit direction (dx,dy)/dist, i.e. (dx,dy)/dist^3.
// Still N^2 force evaluations -- tiling does NOT change the work, it changes WHERE
// the position data is read from.
//
// The win: instead of every thread re-reading particles[j] from slow global
// memory, the block cooperatively loads a 256-wide "tile" of positions into fast
// on-chip __shared__ memory ONCE, then all 256 threads reuse it. We march the tile
// across all particles in steps of blockDim.x (the outer t loop), so each global
// position is read from DRAM ~256x fewer times.
//
// Two __syncthreads() per tile are mandatory:
//   * after the load  -- so no thread reads tile[] before it is fully filled;
//   * after the inner k loop -- so no thread overwrites tile[] for the next step
//     while a slower thread is still reading the current one.
//
// It writes ONLY the velocity (vx/vy), never the position: positions are read by
// every other thread this pass, so moving them here would be a data race. The
// position integration runs afterwards in update_kernel (a separate launch).
// ===========================================================================
__global__ void nbody_force_tiled(Particle *particles, int n, float strength,
                                  float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float xi = 0.0f, yi = 0.0f;
    if (i < n)
    {
        xi = particles[i].x;
        yi = particles[i].y;
    }
    float ax = 0.0f, ay = 0.0f;

    __shared__ float2 tile[256];

    for (int t = 0; t < n; t += blockDim.x)
    {
        int j = t + tid;

        tile[tid] = (j < n) ? make_float2(particles[j].x, particles[j].y)
                            : make_float2(0.0f, 0.0f);
        __syncthreads();

        int valid = min((int)blockDim.x, n - t);
        for (int k = 0; k < valid; k++)
        {
            float dx = tile[k].x - xi, dy = tile[k].y - yi;
            float dist2 = dx * dx + dy * dy + 0.01f;
            float inv = rsqrtf(dist2);
            float inv3 = inv * inv * inv;
            ax += dx * inv3;
            ay += dy * inv3;
        }

        __syncthreads();
    }

    if (i < n)
    {
        particles[i].vx += ax * strength * dt;
        particles[i].vy += ay * strength * dt;
    }
}

// ===========================================================================
// update_kernel  --  advance one particle per thread.  (runs every frame)
// ===========================================================================
// Per-particle physics: gravity, integrate position, bounce off the 4 walls,
// age, and recycle dead particles through the real RNG. Phase 3 effect forces
// (e.g. the gravity well) are added inside this body.
//
// Note: use params.gravity (a dot), NOT params_.gravity -- params is the COPY
// passed in as an argument, not the class member.
// ===========================================================================
__global__ void update_kernel(Particle *particles, int n, SimParams params, float dt, curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    particles[i].vy -= params.gravity * dt;

    particles[i].x += particles[i].vx * dt;
    particles[i].y += particles[i].vy * dt;

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

    particles[i].life -= dt;

    // Recycle a dead particle through its own RNG stream (pass the address so
    // the RNG state can advance).
    if (particles[i].life <= 0.0f)
    {
        respawn_rng(particles[i], &states[i], i % d_numEmitters);
    }
}

// ===========================================================================
// GPU memory management  --  allocate / copy / free the device arrays.
// ===========================================================================
// The CPU and GPU have separate memory. The constructor allocates the device
// arrays and uploads the initial state; update() launches the kernel each frame
// and copies results back for rendering; the destructor frees the device memory.
// ===========================================================================

ParticleSystem::ParticleSystem(const SimParams &p)
    : params_(p), n_(p.n)
{
    gpu_info(); // (given) print which GPU we are on

    // Build the emitter table on the HOST (the CPU cannot read __constant__
    // memory). Each row is { x, y, angle, spread, baseSpeed, r, g, b, lifetime }.
    // Three independent fountains: positions pushed to the sides and aimed apart
    // (left fans up-left, right fans up-right) so they do not overlap into a blob.
    std::vector<Emitter> emitters;
    emitters.push_back({-0.8f, -0.6f, 2.1f, 0.8f, 0.9f, 1.0f, 0.3f, 0.2f, 6.0f});   // left,   red,    aims up-left
    emitters.push_back({0.0f, -0.6f, 1.5708f, 0.8f, 0.9f, 0.2f, 1.0f, 0.9f, 6.0f}); // middle, cyan,   aims straight up
    emitters.push_back({0.8f, -0.6f, 1.1f, 0.8f, 0.9f, 0.7f, 0.3f, 1.0f, 6.0f});    // right,  purple, aims up-right
    int numEmitters = (int)emitters.size();

    // Upload the table into __constant__ memory by SYMBOL name (not a pointer).
    CUDA_CHECK(cudaMemcpyToSymbol(d_emitters, emitters.data(), numEmitters * sizeof(Emitter)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_numEmitters, &numEmitters, sizeof(int)));

    // Allocate the particle array in GPU memory.
    CUDA_CHECK(cudaMalloc(&d_particles_, (size_t)n_ * sizeof(Particle)));

    // Build the starting state on the CPU (given).
    host_.resize(n_);
    for (int i = 0; i < n_; ++i)
        emitter_spawn(host_[i], i, emitters[i % numEmitters]);

    // Copy the initial array up to the GPU (Host -> Device).
    CUDA_CHECK(cudaMemcpy(d_particles_, host_.data(), (size_t)n_ * sizeof(Particle), cudaMemcpyHostToDevice));

    // Allocate one curandState per particle and seed them once.
    CUDA_CHECK(cudaMalloc(&d_rng_, (size_t)n_ * sizeof(curandState)));
    int block = 256, grid = (n_ + block - 1) / block;
    init_rng_kernel<<<grid, block>>>((curandState *)d_rng_, n_, 1234ULL);
    CUDA_CHECK(cudaGetLastError());
}

ParticleSystem::~ParticleSystem()
{
    // Every cudaMalloc needs a matching cudaFree, or the GPU leaks memory.
    cudaFree(d_particles_);
    cudaFree(d_rng_);
}

void ParticleSystem::update(float dt)
{
    // Launch shape: 256 threads per block, enough blocks to cover all particles.
    int block = 256;
    int grid = (n_ + block - 1) / block;

    // Phase 1 -- N-body (shared-memory tiled): read every position through a
    // 256-wide shared tile, accumulate gravity, write velocities only.
    nbody_force_tiled<<<grid, block>>>(d_particles_, n_, 0.00001f, dt);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2 -- integrate: all velocities are now final, so move the positions,
    // bounce off walls, age, and recycle. The launch boundary above is a global
    // sync point, so no thread moves a position another thread is still reading.
    update_kernel<<<grid, block>>>(d_particles_, n_, params_, dt, (curandState *)d_rng_);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to the CPU for rendering. (This per-frame copy is the
    // slow part Phase 4 removes with CUDA-OpenGL interop.)
    CUDA_CHECK(cudaMemcpy(host_.data(), d_particles_, (size_t)n_ * sizeof(Particle), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// to_vertices  --  pack particles into [x,y,r,g,b] for the renderer.  (given)
// ---------------------------------------------------------------------------
// Level 4: the color now comes from each particle's OWN cr/cg/cb (copied from
// its emitter at spawn), not a global lifetime gradient. We still multiply by a
// life-based fade `t` so a particle dims to black as it ages instead of popping
// out of existence the instant it is recycled.
// ---------------------------------------------------------------------------
void ParticleSystem::to_vertices(std::vector<float> &out) const
{
    out.resize((size_t)n_ * 5);
    for (int i = 0; i < n_; ++i)
    {
        const Particle &p = host_[i];
        float t = p.life; // fade factor: full color while alive, dim near death
        if (t < 0.0f)
            t = 0.0f;
        if (t > 1.0f)
            t = 1.0f;
        float *v = &out[(size_t)i * 5];
        v[0] = p.x;
        v[1] = p.y;
        float hot = t * t;
        v[2] = (p.cr + hot) * t; // each emitter's own color, faded by remaining life
        v[3] = (p.cg + hot) * t;
        v[4] = (p.cb + hot) * t;
    }
}
