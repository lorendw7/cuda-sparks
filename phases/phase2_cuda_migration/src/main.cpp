// ===========================================================================
// Phase 2 - CUDA Migration.   (Framework code - you do not need to edit this.)
// ===========================================================================
// This file is IDENTICAL in shape to Phase 1's main.cpp. That is the lesson:
// the application does not care whether the physics runs on the CPU or the GPU.
// All the CUDA lives inside ParticleSystem (particle_system.cu).
//
// The title bar shows particle count + FPS + update time. Compare these numbers
// against your Phase 1 baseline -- especially after you bump kNumParticles up.
// ===========================================================================
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "particle_system.h"
#include "renderer.h"

#include <cstdio>
#include <chrono>
#include <vector>

static constexpr int kNumParticles = 10000;   // start here; try 100000, 1000000
static constexpr int kWidth  = 1280;   // keep square so [-1,1] world coords aren't distorted
static constexpr int kHeight = 1280;

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight,
                                          "cuda-sparks | Phase 2 (CUDA)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);   // no vsync, so FPS reflects real speed

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL\n");
        return 1;
    }

    SimParams params{};
    params.dt          = 0.016f;
    params.gravity     = 0.5f;
    params.damping     = 1.0f;
    params.restitution = 0.8f;
    params.bound       = 1.0f;
    params.n           = kNumParticles;

    ParticleSystem sim(params);    // <-- allocates GPU memory, prints GPU info
    Renderer renderer;
    renderer.init(kNumParticles);

    std::vector<float> vertices;

    using clock = std::chrono::high_resolution_clock;
    auto  last_time  = clock::now();
    auto  fps_time   = clock::now();
    int   frames     = 0;
    double update_ms = 0.0;

    while (!glfwWindowShouldClose(window)) {
        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        if (dt > 0.05f) dt = 0.05f;

        // --- simulate on the GPU (kernel launch + copy back happen inside) ---
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

        ++frames;
        if (std::chrono::duration<float>(now - fps_time).count() >= 0.5f) {
            double fps = frames / std::chrono::duration<double>(now - fps_time).count();
            char title[160];
            std::snprintf(title, sizeof(title),
                          "cuda-sparks | Phase 2 (CUDA) | %d particles | %.0f FPS | update %.2f ms",
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
