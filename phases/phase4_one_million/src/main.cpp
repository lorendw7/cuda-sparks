// Phase 4 (L2-L5) -- window + GL context, then run the 1M-particle sim each frame.
// CUDA-GL interop: the kernel writes vertices straight into the VBO (no CPU round
// trip). Number keys 1/2/3/4 switch effect presets (fireworks / fire / galaxy / Jia).
// SPARKS_MAX_FRAMES caps the loop so Nsight application-replay can profile it.
// glad must be included before glfw.
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio>            // fprintf / printf
#include "renderer.h"        // our hand-written rendering layer
#include "particle_system.h" // the 1M-particle GPU simulation
#include <chrono>            // high_resolution_clock for the per-frame timing
#include <cstdlib>           // getenv, atol

int main()
{
    // 1) Start GLFW. Returns 0 on failure.
    if (!glfwInit())
    {
        fprintf(stderr, "glfwInit failed.\n");
        return 1;
    }

    // 2) Ask for an OpenGL 3.3 core-profile context (same as Phase 3).
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 3) Create a 1280x1280 window. nullptr window means creation failed.
    GLFWwindow *window = glfwCreateWindow(1280, 1280, "cuda-sparks | Phase 4",
                                          nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 1;
    }

    // 4) Make the window's GL context current, disable vsync (uncapped FPS).
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // 5) Load GL function pointers with glad -- only possible once a context exists.
    if (!gladLoadGL(glfwGetProcAddress))
    {
        fprintf(stderr, "gladLoadGL failed.\n");
        return 1;
    }

    // Inner scope: the Renderer must be destroyed (its ~Renderer runs glDelete*)
    // while the GL context is STILL alive. This "}" ends before glfwTerminate(),
    // so the destructor fires at the right time. (RAII + GL ordering.)
    {
        // Simulation config. {} zero-inits every field first. NOTE: gravity /
        // nbodyStrength / swirl set here are OVERWRITTEN by set_preset(0) in the
        // ctor -- presets own those three effect knobs now. Only n / restitution /
        // bound below feed the running sim; params.dt is unused (update() runs on
        // the local dt in the render loop, not params.dt).
        SimParams params{};
        // Particle count knob: env var SPARKS_PARTICLES overrides the default so you
        // can A/B different counts without recompiling. Default 30k -- sparse enough that
        // the dots stay separate and crisp instead of an over-dense blur. 1M is the
        // perf-lesson stress figure, set via the env var.
        const char *nEnv = std::getenv("SPARKS_PARTICLES");
        params.n = nEnv ? std::atol(nEnv) : 30000;
        if (params.n < 1)
        {
            params.n = 1;
        }

        params.gravity = 0.0f;       // (overridden by preset)
        params.nbodyStrength = 0.0f; // (overridden by preset)
        params.swirl = 1.0f;         // (overridden by preset)
        params.restitution = 0.8f;   // keep 80% of speed on each wall bounce
        params.bound = 1.0f;         // world edge [-1,1], matches clip space
        params.dt = 0.016f;          // placeholder, unused this level

        // Construct the system: cudaMallocs the 8 SoA field arrays and fills the
        // initial state on the GPU (init_kernel) -- no host upload.
        ParticleSystem sim(params);

        // Renderer's VBO must be sized to hold every particle (sim.size() = params.n).
        Renderer renderer;
        renderer.init(sim.size());
        sim.register_vbo(renderer.vbo());

        auto lastTime = std::chrono::high_resolution_clock::now(); // previous-frame timestamp, for real dt

        double accUpdate = 0.0;                                      // sim update ms, summed over the current second
        int frames = 0;                                              // frames counted this second
        auto lastReport = std::chrono::high_resolution_clock::now(); // last time we printed

        // Profiling escape hatch: if SPARKS_MAX_FRAMES is set, quit after that many
        // frames so Nsight application-replay can relaunch a process that terminates.
        const char *maxEnv = std::getenv("SPARKS_MAX_FRAMES");
        long maxFrames = maxEnv ? std::atol(maxEnv) : 0; // 0 = run forever (normal use)
        long frameCount = 0;

        // Render loop: run until the user closes the window.
        while (!glfwWindowShouldClose(window))
        {
            // Real frame-time step: dt = seconds since the previous frame, so the sim
            // advances by wall-clock time and runs at the SAME speed at any FPS. (A
            // fixed dt at uncapped FPS ran the physics ~50x too fast -> chaotic churn.)
            // t0 also opens the sim-timing bracket that t1 (below) closes.
            auto t0 = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(t0 - lastTime).count(); // seconds since last frame
            lastTime = t0;                                                  // remember for next frame
            if (dt > 0.05f)                                                 // clamp a hitch (window drag / breakpoint) so particles don't teleport
            {
                dt = 0.05f;
            }

            sim.update(dt); // map VBO -> kernel writes physics + vertices -> unmap
            auto t1 = std::chrono::high_resolution_clock::now();

            // Accumulate the update time in milliseconds (duration<double, milli>).
            accUpdate += std::chrono::duration<double, std::milli>(t1 - t0).count();
            frames++;

            glClearColor(0.02f, 0.02f, 0.05f, 1.0f); // dark blue background
            glClear(GL_COLOR_BUFFER_BIT);            // clear the back buffer

            renderer.draw(); // draw every particle as a point (count = sim.size())

            glfwSwapBuffers(window); // present the frame
            glfwPollEvents();        // handle close/keyboard events

            // Number keys switch the effect preset live. glfwGetKey reads the
            // current key state (already refreshed by glfwPollEvents above); a press
            // re-uploads that preset's emitters + physics. Existing particles adopt
            // the new look only as they recycle, so it fades in over ~1 lifetime.
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
            {
                sim.set_preset(0); // fireworks
            }

            if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
            {
                sim.set_preset(1); // fire
            }

            if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
            {
                sim.set_preset(2); // galaxy
            }

            if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
            {
                sim.set_preset(3); // Jia
            }

            if (maxFrames > 0 && ++frameCount >= maxFrames)
                glfwSetWindowShouldClose(window, 1); // hit the cap -> exit the loop cleanly

            // Once per second, print the average update time + the FPS, then reset.
            if (std::chrono::duration<double>(t1 - lastReport).count() >= 1.0)
            {
                printf("update %.2f ms | %d FPS\n", // \n flushes one clean line/sec
                       accUpdate / frames, frames);
                accUpdate = 0.0;
                frames = 0;
                lastReport = t1;
            }
        }
    } // <- renderer destroyed here, GL context still alive

    // 6) Tear down the context AFTER the renderer is gone.
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
