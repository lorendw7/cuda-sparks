#include "particle_system.h"
#include "cuda_utils.h" // CUDA_CHECK macro + gpu_info()

#include <cuda_runtime.h>
#include <cmath> // floorf

#include <glad/gl.h>
#include <cuda_gl_interop.h>

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
// update_kernel  --  advance one particle AND write its vertex.  (every frame)
// ===========================================================================
// One GPU thread per particle: apply gravity, integrate position, bounce off
// the 4 walls, age, recycle the dead -- then (L2 interop) pack the result as
// [x,y,r,g,b] straight into the OpenGL VBO via the mapped pointer `vbo`. No copy
// to the CPU: physics and vertex output both land in device memory this frame.
// params is passed BY VALUE (a copy per thread) -- use params.gravity (dot),
// never params_ (that member is invisible here).
// ===========================================================================
__global__ void update_kernel(Particle *particles, float *vbo, int n, SimParams params, float dt)
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

    particles[i].life -= dt;       // age this particle
    if (particles[i].life <= 0.0f) // dead? recycle it in place (deterministic)
    {
        spawn(particles[i], i, params.bound);
    }

    // --- Write this particle's vertex straight into the VBO (must be AFTER the
    // recycle above, so a just-reborn particle emits its NEW position/color, not
    // the stale dead one). fade dims the color toward death; clamp to [0,1]
    // because a freshly spawned particle's life can exceed 1.
    float fade = particles[i].life;
    if (fade < 0.0f)
    {
        fade = 0.0f;
    }
    if (fade > 1.0f)
    {
        fade = 1.0f;
    }

    float *v = &vbo[i * 5];          // this particle's 5 floats inside the VBO
    v[0] = particles[i].x;           // attribute 0 (pos.x) -- shader reads offset 0
    v[1] = particles[i].y;           // attribute 0 (pos.y)
    v[2] = particles[i].cr * fade;   // attribute 1 (color.r) -- shader reads offset 8
    v[3] = particles[i].cg * fade;   // attribute 1 (color.g)
    v[4] = particles[i].cb * fade;   // attribute 1 (color.b)
}

// ===========================================================================
// Constructor  --  allocate device memory, build initial state, upload it.
// ===========================================================================
ParticleSystem::ParticleSystem(const SimParams &p) // definition of the ctor declared in the header
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

    cudaFree(d_particles_); // matching free for the ctor's cudaMalloc
}

// ===========================================================================
// update  --  map the VBO, run the kernel straight into it, unmap.  (per frame)
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

    // 3) Launch: physics into d_particles_, vertices into d_vbo, in one kernel.
    int block = 256;
    int grid = (n_ + block - 1) / block;
    update_kernel<<<grid, block>>>(d_particles_, d_vbo, n_, params_, dt);
    CUDA_CHECK(cudaGetLastError());

    // 4) Return the VBO: CUDA -> GL. Waits for the kernel to finish, so the
    //    buffer is safe for draw() the moment this returns. draw() must come AFTER.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource_, 0));
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
        if (t < 0.0f)
            t = 0.0f;
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
