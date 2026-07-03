#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// Renderer -- draws particles as OpenGL points. Hand-written from scratch in
// Phase 4. This class OWNS three GPU resources (a VAO, a VBO, and a shader
// program), so it follows the RAII pattern: acquire in init(), release in the
// destructor, and forbid copying (see below).
//
// Vertex format expected by upload(): 5 floats per particle, interleaved:
//     [ x, y, r, g, b,  x, y, r, g, b,  ... ]
//   x,y   : clip-space position, range [-1, 1]  (drawn directly)
//   r,g,b : color, range [0, 1]
// ---------------------------------------------------------------------------
class Renderer
{
private:
    unsigned int vao_     = 0;  // vertex-array object id (the attribute layout)
    unsigned int vbo_     = 0;  // vertex-buffer object id (the vertex bytes on GPU)
    unsigned int program_ = 0;  // linked shader program id
    int          count_   = 0;  // particle count from the most recent upload()

public:
    // Constructor does NOTHING with GL: no GL context exists yet at construction
    // time, so the compiler-generated empty default is exactly right.
    Renderer() = default;

    // Destructor releases the three GPU resources (RAII: cleanup is automatic).
    ~Renderer();

    // Non-copyable. This object owns unique GPU resource ids; copying would make
    // two Renderers hold the same ids and double-free them on destruction.
    // Deleting these turns any copy attempt into a COMPILE error instead.
    Renderer(const Renderer&)            = delete;  // copy constructor (Renderer b = a;)
    Renderer& operator=(const Renderer&) = delete;  // copy assignment  (b = a;)

    // Second-phase init: build the shader program + VAO/VBO. Must be called
    // AFTER the GL context is current (glfwMakeContextCurrent + gladLoadGL).
    // max_particles = how many particles the VBO is pre-sized to hold.
    bool init(int max_particles);

    // Copy one fresh frame of [x,y,r,g,b] vertices to the GPU. count = particles.
    void upload(const std::vector<float>& interleaved, int count);

    // Draw the most recently uploaded particles. Call once per frame.
    void draw();

    unsigned int vbo() const {return vbo_;}
};
