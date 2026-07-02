// Headers: glad must be included before glfw
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio> // for fprintf on errors

static const float kTraingle[] = {
    -0.5f, -0.5f,
    0.5f, -0.5f,
    0.0f, 0.5f};

static const char *kVerSrc = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char *kFragSrc = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.6, 0.1, 1.0);
}
)";

// Compile one shader from source, return the compiled shader object ID.
// type is GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
static unsigned int compileShader(unsigned int type, const char *src)
{
    // Ask OpenGL for an empty shader object and get its ID.
    unsigned int id = glCreateShader(type);
    // Hand the source string to the shader object.
    glShaderSource(id, 1, &src, nullptr);
    // Compile it.
    glCompileShader(id);
    // Query whether compilation succeeded.
    int ok = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    // On failure, fetch and print the GPU's compile log.
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(id, 512, nullptr, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log);
    }
    return id;
}

// Build a shader program from vertex + fragment source. Returns the program ID.
static unsigned int makeProgram(const char *vertSrc, const char *fragSrc)
{
    // C-1: compile both shaders using the helper you just wrote.
    //      unsigned int vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    //      unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    // C-2: create an empty program object.
    //      unsigned int prog = glCreateProgram();
    unsigned int prog = glCreateProgram();
    // C-3: attach both compiled shaders to the program.
    //      glAttachShader(prog, vs);
    //      glAttachShader(prog, fs);
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    // C-4: link them into one program.
    //      glLinkProgram(prog);
    glLinkProgram(prog);
    // C-5: check link status (same pattern as compile, but glGetProgramiv / GL_LINK_STATUS).
    //      int ok = 0;
    //      glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    //      if (!ok) { char log[512]; glGetProgramInfoLog(prog, 512, nullptr, log);
    //                 fprintf(stderr, "program link failed:\n%s\n", log); }
    int ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        fprintf(stderr, "program link failed:\n%s\n", log);
    }

    // C-6: the individual shaders are now baked into the program; delete them.
    //      glDeleteShader(vs);
    //      glDeleteShader(fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    // C-7: return prog;
    return prog;
}

int main()
{
    // 1) Initialize GLFW. On failure (returns 0), report and return 1.
    //    call: glfwInit()
    if (!glfwInit())
    {
        fprintf(stderr, "glfwInit failed.\n");
        return 1;
    }
    // 2) Tell GLFW we want an OpenGL 3.3 core profile (same as Phase 3).
    //    call glfwWindowHint 3 times:
    //      GLFW_CONTEXT_VERSION_MAJOR -> 3
    //      GLFW_CONTEXT_VERSION_MINOR -> 3
    //      GLFW_OPENGL_PROFILE        -> GLFW_OPENGL_CORE_PROFILE
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 3) Create a 1280x1280 window titled "cuda-sparks | Phase 4".
    //    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    //    On failure (window == nullptr) call glfwTerminate() then return 1.
    GLFWwindow *window = glfwCreateWindow(1280, 1280, "cuda-sparks | Phase 4",
                                          nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 1;
    }

    // 4) Make this window's context current.
    //    glfwMakeContextCurrent(window);
    //    then glfwSwapInterval(0);  // disable vsync so FPS reflects real speed
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    // 5) Now that a context exists, load GL functions with glad. Return 1 on failure.
    //    if (!gladLoadGL(glfwGetProcAddress)) { ... }
    if (!gladLoadGL(glfwGetProcAddress))
    {
        fprintf(stderr, "gladLoadGL failed.\n");
        return 1;
    }

    // ---- D. Upload the triangle and describe its layout ----

    // D-0: build the shader program once, keep its ID for the draw loop.
    unsigned int prog = makeProgram(kVerSrc, kFragSrc);

    // D-1: create a VAO and a VBO (each is just an unsigned int ID).
    //      unsigned int vao, vbo;
    //      glGenVertexArrays(1, &vao);   // 1 = "give me one", writes the ID into &vao
    //      glGenBuffers(1, &vbo);
    unsigned int vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    // D-2: bind the VAO first. From now on, vertex-attribute state is recorded into it.
    //      glBindVertexArray(vao);
    glBindVertexArray(vao);

    // D-3: bind the VBO as the current GL_ARRAY_BUFFER, then upload the data.
    //      glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //      glBufferData(GL_ARRAY_BUFFER, sizeof(kTraingle), kTraingle, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kTraingle), kTraingle, GL_STATIC_DRAW);
    // D-4: describe attribute 0 (matches `layout(location = 0)` in the vertex shader).
    //      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    //      glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // D-5: (optional tidy) unbind so nothing accidental modifies them later.
    //      glBindBuffer(GL_ARRAY_BUFFER, 0);
    //      glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 6) Render loop: keep running until the window is asked to close.
    //    while (!glfwWindowShouldClose(window)) {
    //        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);  // set clear color = dark blue
    //        glClear(GL_COLOR_BUFFER_BIT);             // clear back buffer with it
    //        glfwSwapBuffers(window);                  // present to screen
    //        glfwPollEvents();                         // handle close/keyboard events
    //    }

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw: activate our shaders, restore the vertex layout, issue the draw.
        glUseProgram(prog);                 // use our compiled vertex+fragment shaders
        glBindVertexArray(vao);             // restore attribute 0's layout (recorded in Step D)
        glDrawArrays(GL_TRIANGLES, 0, 3);   // read 3 vertices as one triangle, starting at index 0

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Release GL objects before the context is torn down.
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    // 7) Cleanup: glfwDestroyWindow(window); glfwTerminate(); return 0;
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
