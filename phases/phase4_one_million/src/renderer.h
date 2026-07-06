#pragma once

// ---------------------------------------------------------------------------
// Renderer -- draws particles as OpenGL points. Hand-written from scratch in
// Phase 4. This class OWNS three GPU resources (a VAO, a VBO, and a shader
// program), so it follows the RAII pattern: acquire in init(), release in the
// destructor, and forbid copying (see below).
//
// ---------------------------------------------------------------------------
class Renderer
{
private:
    unsigned int vao_     = 0;  // vertex-array object id (the attribute layout)
    unsigned int vbo_     = 0;  // vertex-buffer object id (the vertex bytes on GPU)
    unsigned int program_ = 0;  // linked shader program id
    int          count_   = 0;  // particle count to draw (set in init(); = max_particles)

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

    // Draw count_ particles from the VBO (the CUDA kernel writes it directly via
    // interop -- there is no CPU upload step). Call once per frame.
    void draw();

    unsigned int vbo() const {return vbo_;}
};
