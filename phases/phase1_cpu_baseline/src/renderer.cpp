#include "renderer.h"

#include <glad/gl.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Minimal shader program: pass position straight through, color the point.
// ---------------------------------------------------------------------------
static const char *kVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position  = vec4(aPos, 0.0, 1.0);
    gl_PointSize = 12.0;          // size of each particle in pixels
    vColor       = aColor;
}
)";

static const char *kFragmentShader = R"(
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main() {
    vec2  d    = gl_PointCoord - vec2(0.5);
    float dist = length(d);
    if (dist > 0.5) discard;      // make the point circular
    
    float glow = 1.0 - dist * 2.0;  // glow = 1.0 at center, 0.0 at edge
    FragColor  = vec4(vColor * glow, glow);
}
)";

static unsigned int compile(unsigned int type, const char *src)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::fprintf(stderr, "[renderer] shader compile error:\n%s\n", log);
    }
    return shader;
}

bool Renderer::init(int max_particles)
{
    // --- shader program ---
    unsigned int vs = compile(GL_VERTEX_SHADER, kVertexShader);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, kFragmentShader);
    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // --- vertex buffer / array objects ---
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    // Reserve space for the maximum number of particles. 5 floats each.
    // GL_DYNAMIC_DRAW: we overwrite this buffer every frame.
    glBufferData(GL_ARRAY_BUFFER,
                 (long long)max_particles * 5 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    const int stride = 5 * sizeof(float);
    // location 0: position (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void *)0);
    glEnableVertexAttribArray(0);
    // location 1: color (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Let the vertex shader control point size via gl_PointSize.
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Enable alpha blending so the soft (semi-transparent) glow edges of each
    // particle blend with the background instead of overwriting it.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

void Renderer::upload(const std::vector<float> &interleaved, int count)
{
    count_ = count;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    // glBufferSubData reuses the buffer we already allocated in init().
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    (long long)count * 5 * sizeof(float),
                    interleaved.data());
}

void Renderer::draw()
{
    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, count_);
    glBindVertexArray(0);
}

void Renderer::shutdown()
{
    glDeleteBuffers(1, &vbo_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);
}
