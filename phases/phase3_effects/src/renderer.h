#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// Renderer  --  draws particles as colored OpenGL points.  (Framework code.)
// ---------------------------------------------------------------------------
// You do not need to edit this file. It is here so you can see how the GPU
// draws things, but the CUDA learning happens in particle_system.h.
//
// Vertex format expected by upload():  5 floats per particle, interleaved:
//     [ x, y, r, g, b,  x, y, r, g, b,  ... ]
//   x, y  : world position, range [-1, 1]   (clip space, drawn directly)
//   r,g,b : color, range [0, 1]
// ---------------------------------------------------------------------------
class Renderer {
public:
    // Create shaders and GPU buffers. Call once after the GL context exists.
    bool init(int max_particles);

    // Copy a fresh frame of vertex data to the GPU. `count` = number of particles.
    void upload(const std::vector<float>& interleaved, int count);

    // Draw the most recently uploaded particles. Call once per frame.
    void draw();

    // Release GPU resources.
    void shutdown();

private:
    unsigned int vao_     = 0;
    unsigned int vbo_     = 0;
    unsigned int program_ = 0;
    int          count_   = 0;
};
