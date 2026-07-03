#include "renderer.h"

#include <glad/gl.h> // OpenGL function declarations (glGen*, glDraw*, ...)
#include <cstdio>    // fprintf, used to print shader compile errors

// ---------------------------------------------------------------------------
// Vertex shader: runs ONCE PER PARTICLE. Places the point and forwards color.
// ---------------------------------------------------------------------------
static const char *kVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;    // attribute 0: position (matches VertexAttribPointer 0)
layout (location = 1) in vec3 aColor;  // attribute 1: color    (matches VertexAttribPointer 1)
out vec3 vColor;                       // hand the color to the fragment shader
void main() {
    gl_Position  = vec4(aPos, 0.0, 1.0); // already in [-1,1] clip space; z=0, w=1
    gl_PointSize = 2.0;                 // point diameter in pixels (needs GL_PROGRAM_POINT_SIZE)
    vColor       = aColor;               // pass color through unchanged
}
)";

// ---------------------------------------------------------------------------
// Fragment shader: runs ONCE PER PIXEL of the point. Carves a soft round dot.
// ---------------------------------------------------------------------------
static const char *kFragmentShader = R"(
#version 330 core
in  vec3 vColor;                            // interpolated color from the vertex shader
out vec4 FragColor;                         // final pixel color (r,g,b,a)
void main() {
    vec2  d    = gl_PointCoord - vec2(0.5); // offset of this pixel from the point center
    float dist = length(d);                 // how far this pixel is from the center
    if (dist > 0.5) discard;                // outside the circle: draw nothing (round dot)
    float glow = 1.0 - dist * 2.0;          // 1.0 at center, 0.0 at the edge
    FragColor  = vec4(vColor * glow, glow); // fade both color and alpha toward the edge
}
)";

// Compile one shader from source; print the GPU log on failure. Returns its id.
static unsigned int compile(unsigned int type, const char *src)
{
    unsigned int shader = glCreateShader(type); // create an empty shader object
    glShaderSource(shader, 1, &src, nullptr);   // give it the source (1 string, null-terminated)
    glCompileShader(shader);                    // compile it on the driver
    int ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok); // did compilation succeed? (1/0 into ok)
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log); // fetch the error text
        fprintf(stderr, "[renderer] shader compile error:\n%s\n", log);
    }
    return shader; // return the id (even on failure, for the caller to handle)
}

// Build the shader program + VAO/VBO. Call once, after the GL context is current.
bool Renderer::init(int max_particles)
{
    // (a) compile both shaders and link them into one program (stored in program_)
    unsigned int vs = compile(GL_VERTEX_SHADER, kVertexShader);
    unsigned int fs = compile(
        GL_FRAGMENT_SHADER, kFragmentShader);
    program_ = glCreateProgram(); // create an empty program object
    glAttachShader(program_, vs); // attach the vertex shader
    glAttachShader(program_, fs); // attach the fragment shader
    glLinkProgram(program_);      // link them into a runnable pipeline
    glDeleteShader(vs);           // program keeps its own copy now,
    glDeleteShader(fs);           // so the source shaders can be freed

    // (b) create and bind the VAO (layout) and VBO (vertex bytes)
    glGenVertexArrays(1, &vao_);         // ask GL for 1 VAO, write its id into vao_
    glGenBuffers(1, &vbo_);              // ask GL for 1 VBO, write its id into vbo_
    glBindVertexArray(vao_);             // from now on, attribute config records into vao_
    glBindBuffer(GL_ARRAY_BUFFER, vbo_); // make vbo_ the active vertex buffer

    // (c) pre-allocate GPU memory big enough for max_particles, upload no data yet
    glBufferData(GL_ARRAY_BUFFER, (long long)max_particles * 5 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW); // nullptr = reserve only; DYNAMIC = rewritten often

    // (d) describe the interleaved [x,y,r,g,b] layout so the shader reads it right
    const int stride = 5 * sizeof(float);                               // bytes from one particle to the next
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void *)0); // attr 0: 2 floats at offset 0
    glEnableVertexAttribArray(0);                                       // turn attribute 0 on
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *)(2 * sizeof(float))); // attr 1: 3 floats at offset 8
    glEnableVertexAttribArray(1);                                       // turn attribute 1 on

    // (e) unbind the VAO, then enable point-size + alpha blending
    glBindVertexArray(0);                              // done configuring; unbind to avoid accidental edits
    glEnable(GL_PROGRAM_POINT_SIZE);                   // let the shader's gl_PointSize take effect
    glEnable(GL_BLEND);                                // enable alpha blending (soft glowing edges)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // new*alpha + old*(1-alpha): standard transparency
    return true;
}

// Copy one fresh frame of [x,y,r,g,b] vertices into the pre-allocated VBO.
void Renderer::upload(const std::vector<float> &interleaved, int count)
{
    count_ = count;                      // remember this frame's particle count for draw()
    glBindBuffer(GL_ARRAY_BUFFER, vbo_); // select our VBO
    glBufferSubData(GL_ARRAY_BUFFER, 0,  // overwrite the front of the existing buffer (offset 0)...
                    (long long)count * 5 * sizeof(float), // ...this many bytes...
                    interleaved.data()); // ...from the vector's raw array. No realloc = cheap per frame.
}

// Draw the most recently uploaded particles as points. Call once per frame.
void Renderer::draw()
{
    glUseProgram(program_);             // activate our shader program
    glBindVertexArray(vao_);            // restore the vertex layout (recorded in the VAO)
    glDrawArrays(GL_POINTS, 0, count_); // draw count_ points, one per particle, starting at index 0
    glBindVertexArray(0);               // unbind to avoid accidental edits afterwards
}

// Destructor (RAII): release the three GPU resources init() acquired. Automatic.
Renderer::~Renderer()
{
    glDeleteBuffers(1, &vbo_);      // free the vertex buffer
    glDeleteVertexArrays(1, &vao_); // free the vertex-array object
    glDeleteProgram(program_);      // free the shader program
}
