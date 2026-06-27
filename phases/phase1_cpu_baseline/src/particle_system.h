#pragma once
#include "particles.h"      // Particle, SimParams  (in the shared include/ folder)

#include <vector>
#include <random>
#include <cmath>

// ===========================================================================
// ParticleSystem  --  the CPU simulation.   *** THIS IS YOUR FILE ***
// ===========================================================================
// The framework gives you:
//   - storage for N particles
//   - reset()      : fill every particle with random starting values
//   - respawn(i)   : recycle a dead particle back at the emitter
//   - to_vertices(): convert particles -> data the renderer can draw
//
// YOU write:
//   - update(dt)   : advance every particle by one time step  (the TODO below)
//
// In Phase 2 this exact loop becomes a CUDA kernel - so write it as if each
// particle is independent. That habit is what makes the GPU port easy.
// ===========================================================================
class ParticleSystem {
public:
    explicit ParticleSystem(const SimParams& p) : params_(p), rng_(1234u) {
        particles_.resize(p.n);
        reset();
    }

    int size() const { return params_.n; }

    // ---- framework: randomize all particles (called once at startup) ----
    void reset() {
        for (int i = 0; i < params_.n; ++i) respawn(i);
    }

    // ---- framework: send a dead particle back to the emitter (origin) ----
    // The emitter shoots particles upward in a fan, with a random lifetime.
    void respawn(int i) {
        std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
        float angle = uni(rng_) * 3.14159f;            // any direction
        float speed = 0.3f + 0.5f * (0.5f * uni(rng_) + 0.5f);
        particles_[i].x    = 0.0f;                     // emitter at screen center
        particles_[i].y    = -0.6f;
        particles_[i].vx   = std::cos(angle) * speed;
        particles_[i].vy   = std::abs(std::sin(angle)) * speed + 0.4f;  // bias upward
        particles_[i].r    = 0.01f;
        particles_[i].life = 0.5f + 0.5f * (0.5f * uni(rng_) + 0.5f);   // 0.5 .. 1.0
    }

    // ======================================================================
    //  -- TODO --  Advance the whole system by one time step `dt`.
    // ======================================================================
    // For EACH particle i (a simple loop from 0 to params_.n):
    //
    //   1. Gravity:   pull velocity down ->  vy -= params_.gravity * dt
    //   2. Move:      x += vx * dt ;  y += vy * dt
    //   3. Age:       life -= dt        (particles slowly die)
    //   4. Recycle:   if (life <= 0) respawn(i);
    //
    // Tips:
    //   - Access a field with  particles_[i].vy , particles_[i].x , etc.
    //   - Keep each iteration independent of the others - no particle should
    //     read another particle's data. (That is what makes Phase 2's GPU
    //     port a copy-paste of this logic.)
    // ----------------------------------------------------------------------
    void update(float dt) {
        // Write your loop here.
        for (int i = 0; i < params_.n; ++i) {
            particles_[i].vy -= params_.gravity * dt;

            particles_[i].x += particles_[i].vx * dt;
            particles_[i].y += particles_[i].vy * dt;

            if (particles_[i].x > params_.bound) {
                particles_[i].x = params_.bound;
                particles_[i].vx = 
                -particles_[i].vx * params_.restitution;
            }

            if (particles_[i].x < -params_.bound) {
                particles_[i].x = -params_.bound;
                particles_[i].vx = -particles_[i].vx * params_.restitution;
            }

            if (particles_[i].y > params_.bound)
            {
                particles_[i].y = params_.bound;
                particles_[i].vy = -particles_[i].vy * params_.restitution;
            }

            if (particles_[i].y < -params_.bound)   
            {
                particles_[i].y = -params_.bound;
                particles_[i].vy = -particles_[i].vy * params_.restitution;
            }
            
            


            particles_[i].life -= dt;

            if (particles_[i].life <= 0)
            {
                respawn(i);
            }
            
        }
    }
    // ======================================================================

    // ---- framework: pack particles into interleaved [x,y,r,g,b] vertices ----
    // Color fades from hot (white-yellow) when young to red as life runs out.
    void to_vertices(std::vector<float>& out) const {
        out.resize((size_t)params_.n * 5);
        for (int i = 0; i < params_.n; ++i) {
            const Particle& p = particles_[i];
            float t = p.life;                    // 1.0 = young, 0.0 = dead
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            float* v = &out[(size_t)i * 5];
            v[0] = p.x;
            v[1] = p.y;
            v[2] = 1.0f;                          // red   channel (always strong)
            v[3] = 0.4f + 0.6f * t;               // green: bright when young
            v[4] = 0.2f * t;                      // blue : a touch when young
        }
    }

private:
    SimParams              params_;
    std::vector<Particle>  particles_;
    std::mt19937           rng_;
};
