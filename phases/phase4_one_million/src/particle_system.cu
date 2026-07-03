#include "particle_system.h"
#include "cuda_utils.h" // CUDA_CHECK macro + gpu_info()

#include <cuda_runtime.h>
#include <cmath> // floorf

#include <glad/gl.h>
#include <cuda_gl_interop.h>

// ===========================================================================
// spawn  --  write particle i's initial/reborn state into the SoA arrays.
// ===========================================================================
// Device-only now (L3): both callers -- init_kernel (initial fill) and
// update_kernel (recycle a dead particle) -- run on the GPU, so there is no
// host path left. Writes p.field[i] (SoA) instead of p.field (AoS). No curand
// yet -- randomness is faked from the index i.
// ===========================================================================
__device__ inline void spawn(ParticleSoA p, int i, float bound)
{
    // Three pseudo-random values in [0,1): index * irrational, keep the fraction.
    float f1 = i * 0.6180340f; // golden ratio
    f1 -= floorf(f1);          // fractional part -> [0,1)
    float f2 = i * 0.7548777f;
    f2 -= floorf(f2);
    float f3 = i * 0.9541230f;
    f3 -= floorf(f3);

    p.x[i] = (f1 * 2.0f - 1.0f) * bound; // map [0,1)->[-bound,bound): scatter across x
    p.y[i] = (f2 * 2.0f - 1.0f) * bound; // scatter across y
    p.vx[i] = 0.0f;                      // start at rest; gravity takes over
    p.vy[i] = 0.0f;

    p.cr[i] = 0.3f + 0.7f * f1; // per-particle color from index, so the field isn't flat
    p.cg[i] = 0.3f + 0.7f * f2;
    p.cb[i] = 1.0f;

    p.life[i] = 1.0f + f3 * 3.0f; // staggered lifetime 1..4s -> not all die on one frame
}

// ===========================================================================
// init_kernel  --  build the whole initial population on the GPU.  (runs ONCE)
// ===========================================================================
// One thread per particle calls spawn(). Because spawn is a pure function of the
// index i, we can fill the initial state directly in device memory -- no CPU
// arrays, no host->device upload (that's why L3 has no host_ mirror anymore).
// ===========================================================================
__global__ void init_kernel(ParticleSoA p, int n, float bound)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    spawn(p, i, bound);
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
__global__ void update_kernel(ParticleSoA p, float *vbo, int n, SimParams params, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread id = particle index
    if (i >= n)                                    // grid usually rounds up past n:
        return;                                    // extra threads must do nothing

    p.vy[i] -= params.gravity * dt; // gravity changes vertical velocity

    p.x[i] += p.vx[i] * dt; // integrate position from velocity
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

    p.life[i] -= dt;       // age this particle
    if (p.life[i] <= 0.0f) // dead? recycle it in place (deterministic)
    {
        spawn(p, i, params.bound);
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

    float *v = &vbo[i * 5]; // this particle's 5 floats inside the VBO
    v[0] = p.x[i];          // attribute 0 (pos.x) -- shader reads offset 0
    v[1] = p.y[i];          // attribute 0 (pos.y)
    v[2] = p.cr[i] * fade;  // attribute 1 (color.r) -- shoffset 8
    v[3] = p.cg[i] * fade;  // attribute 1 (color.g)
    v[4] = p.cb[i] * fade;  // attribute 1 (color.b)
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

    int block = 256, grid = (n_ + block - 1) / block;
    init_kernel<<<grid, block>>>(d_, n_, params_.bound);
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

    // 3) Launch: physics into the SoA arrays (d_), vertices into d_vbo, one kernel.
    int block = 256;
    int grid = (n_ + block - 1) / block;
    update_kernel<<<grid, block>>>(d_, d_vbo, n_, params_, dt);
    CUDA_CHECK(cudaGetLastError());

    // 4) Return the VBO: CUDA -> GL. Waits for the kernel to finish, so the
    //    buffer is safe for draw() the moment this returns. draw() must come AFTER.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource_, 0));
}

