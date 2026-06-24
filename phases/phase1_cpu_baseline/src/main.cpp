// ===========================================================================
// Phase 1 — CPU Baseline.   (Framework code — you do not need to edit this.)
// ===========================================================================
// Opens a window, runs the CPU particle simulation, draws it with OpenGL, and
// shows particle count + FPS + CPU update time in the window title. That title
// bar is your *baseline benchmark* — remember the numbers; Phase 2 (CUDA) will
// beat them.
// ===========================================================================
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "particle_system.h"
#include "renderer.h"

#include <cstdio>
#include <chrono>
#include <vector>

static constexpr int kNumParticles = 10000;   // Phase 1 target
static constexpr int kWidth  = 1000;
static constexpr int kHeight = 1000;

int main() {
    // ---- 1. create a window with an OpenGL 3.3 core context ----
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight,
                                          "cuda-sparks | Phase 1 (CPU)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);   // 0 = no vsync, so FPS reflects real speed

    // ---- 2. load OpenGL function pointers ----
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL\n");
        return 1;
    }

    // ---- 3. set up the simulation and the renderer ----
    SimParams params{};
    params.dt          = 0.016f;   // not used directly; we measure real dt below
    params.gravity     = 0.5f;
    params.damping     = 1.0f;
    params.restitution = 0.8f;
    params.bound       = 1.0f;
    params.n           = kNumParticles;

    ParticleSystem sim(params);
    Renderer renderer;
    renderer.init(kNumParticles);

    std::vector<float> vertices;

    // ---- 4. main loop ----
    using clock = std::chrono::high_resolution_clock;
    auto  last_time   = clock::now();
    auto  fps_time    = clock::now();
    int   frames      = 0;
    double update_ms  = 0.0;

    while (!glfwWindowShouldClose(window)) {
        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        if (dt > 0.05f) dt = 0.05f;   // clamp after a stall so nothing explodes

        // --- simulate (this is the part you wrote) ---
        auto t0 = clock::now();
        sim.update(dt);
        auto t1 = clock::now();
        update_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- draw ---
        sim.to_vertices(vertices);
        renderer.upload(vertices, sim.size());

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.draw();
        glfwSwapBuffers(window);
        glfwPollEvents();

        // --- FPS counter in the title bar (updates ~2x/sec) ---
        ++frames;
        if (std::chrono::duration<float>(now - fps_time).count() >= 0.5f) {
            double fps = frames / std::chrono::duration<double>(now - fps_time).count();
            char title[160];
            std::snprintf(title, sizeof(title),
                          "cuda-sparks | Phase 1 (CPU) | %d particles | %.0f FPS | update %.2f ms",
                          sim.size(), fps, update_ms);
            glfwSetWindowTitle(window, title);
            frames   = 0;
            fps_time = now;
        }
    }

    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
