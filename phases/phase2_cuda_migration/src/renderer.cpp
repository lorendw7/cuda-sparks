#include "renderer.h"

#include <glad/gl.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// The two shaders below are small programs that run ON the GPU during drawing:
//   - the VERTEX shader runs once per particle (positions the point),
//   - the FRAGMENT shader runs once per pixel of the point (colors it).
// They are written in GLSL and compiled at runtime by the driver.
// ---------------------------------------------------------------------------

// Vertex shader: takes one particle's position + color and places the point.
static const char *kVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;     // particle position (matches attrib 0)
layout (location = 1) in vec3 aColor;   // particle color    (matches attrib 1)
out vec3 vColor;                         // passed on to the fragment shader
void main() {
    gl_Position  = vec4(aPos, 0.0, 1.0); // position is already in [-1,1] clip space
    gl_PointSize = 12.0;                 // size of each particle in pixels
    vColor       = aColor;
}
)";

// Fragment shader: runs per pixel of the point sprite, makes it a soft round dot.
static const char *kFragmentShader = R"(
#version 330 core
in  vec3 vColor;                  // interpolated color from the vertex shader
out vec4 FragColor;               // final pixel color (rgba)
void main() {
    vec2  d    = gl_PointCoord - vec2(0.5);  // offset from the point's center
    float dist = length(d);
    if (dist > 0.5) discard;      // outside the unit circle -> draw nothing (round dot)

    float glow = 1.0 - dist * 2.0;          // 1.0 at center, 0.0 at the edge
    FragColor  = vec4(vColor * glow, glow); // fade color AND alpha toward the edge
}
)";

// Compile one shader (vertex or fragment) from GLSL source and report errors.
static unsigned int compile(unsigned int type, const char *src)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);   // hand the source to the driver
    glCompileShader(shader);                     // compile it on the GPU driver
    int ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);  // fetch the error text
        std::fprintf(stderr, "[renderer] shader compile error:\n%s\n", log);
    }
    return shader;
}

// Set up the shader program and the GPU buffers. Called once at startup.
bool Renderer::init(int max_particles)
{
    // --- shader program: compile both shaders, link them into one program ---
    unsigned int vs = compile(GL_VERTEX_SHADER, kVertexShader);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, kFragmentShader);
    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);          // combine vs + fs into a runnable program
    glDeleteShader(vs);               // the linked program keeps its own copy,
    glDeleteShader(fs);               // so the individual shaders can be freed

    // --- vertex buffer (VBO) + vertex array (VAO) ---
    // VAO  = remembers HOW to read the vertex data (the attribute layout below).
    // VBO  = the actual block of vertex bytes living in GPU memory.
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);                  // make vao_ the active layout
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);      // make vbo_ the active data buffer

    // Reserve room for the largest frame we will ever draw: 5 floats/particle.
    // GL_DYNAMIC_DRAW hints that we overwrite this buffer every frame (upload()).
    glBufferData(GL_ARRAY_BUFFER,
                 (long long)max_particles * 5 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    // Describe the interleaved layout [x,y,r,g,b] so the shader reads it right.
    // stride = bytes from one particle to the next (one full [x,y,r,g,b] block).
    const int stride = 5 * sizeof(float);
    // attribute 0 = position: 2 floats, starting at byte offset 0.
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void *)0);
    glEnableVertexAttribArray(0);
    // attribute 1 = color: 3 floats, starting after the 2 position floats.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);             // unbind: we're done configuring the layout

    // Let the vertex shader set point size via gl_PointSize.
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Alpha blending: the soft (semi-transparent) glow edges of each particle
    // blend with what's behind them instead of overwriting it. The function says
    // "new pixel * its alpha + existing pixel * (1 - alpha)".
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

// Copy one fresh frame of [x,y,r,g,b] data from the CPU into the GPU buffer.
void Renderer::upload(const std::vector<float> &interleaved, int count)
{
    count_ = count;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    // glBufferSubData overwrites the front of the buffer we already allocated in
    // init() -- cheaper than reallocating with glBufferData every frame.
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    (long long)count * 5 * sizeof(float),
                    interleaved.data());
}

// Draw the most recently uploaded particles. Called once per frame.
void Renderer::draw()
{
    glUseProgram(program_);                 // use our shader program
    glBindVertexArray(vao_);                // use our vertex layout + buffer
    glDrawArrays(GL_POINTS, 0, count_);     // draw count_ points (one per particle)
    glBindVertexArray(0);
}

// Free the GPU resources we created in init().
void Renderer::shutdown()
{
    glDeleteBuffers(1, &vbo_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);
}
