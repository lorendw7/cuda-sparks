// Phase 4 (L2-L6 + Presentation P1) -- window + GL context, then run the 1M-particle
// sim each frame. CUDA-GL interop: the kernel writes vertices straight into the VBO (no
// CPU round trip). Hotkeys switch effect presets: J = Jia, 1 = fireworks (boot default),
// 2 = fire, 3 = galaxy, 4 = rain, 5 = smoke, 6 = curl-noise, 7 = strange attractor.
// P1 auto-play: Space toggles hands-off auto-cycling (advances a preset every ~10 s);
// any number key drops back to manual. SPARKS_MAX_FRAMES caps the loop for Nsight replay.
// glad must be included before glfw.
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio>            // fprintf / printf
#include "renderer.h"        // our hand-written rendering layer
#include "particle_system.h" // the 1M-particle GPU simulation
#include <chrono>            // high_resolution_clock for the per-frame timing
#include <cstdlib>           // getenv, atol

// P2a: keep the particle viewport a SQUARE so NDC [-1,1]^2 never stretches (dots stay
// round on any aspect ratio). GLFW calls this whenever the framebuffer resizes -- window
// drag, or the F11 fullscreen toggle. width/height are the new size in real pixels.
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    int side = width < height ? width : height; // shorter edge -> the largest square that fits
    glViewport(0, 0, side, side);               // map NDC onto that square, anchored bottom-left;
                                                // the leftover strip (right, on landscape) is for the HUD
}

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

    // P2a: register the callback (GLFW invokes it on every future resize), then set the
    // viewport ONCE for the current size -- the callback only fires on CHANGES, so the very
    // first frame needs a manual call. glfwGetFramebufferSize returns REAL pixels (not the
    // logical window size), which is what glViewport needs on a HiDPI / scaled 2.5K display.
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    int fbw, fbh;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    framebuffer_size_callback(window, fbw, fbh);

    // Inner scope: the Renderer must be destroyed (its ~Renderer runs glDelete*)
    // while the GL context is STILL alive. This "}" ends before glfwTerminate(),
    // so the destructor fires at the right time. (RAII + GL ordering.)
    {
        // Simulation config. {} zero-inits every field first. NOTE: gravity /
        // nbodyStrength / swirl set here are OVERWRITTEN by set_preset(1) in the
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
        params.numShells = 16; // L6-2: firework shells (groups). Particles per shell = n / 16.

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
        struct KeyBind
        {
            int key;
            int preset;
        };

        const KeyBind kBinds[] = {
            {GLFW_KEY_J, 0},
            {GLFW_KEY_1, 1},
            {GLFW_KEY_2, 2},
            {GLFW_KEY_3, 3},
            {GLFW_KEY_4, 4},
            {GLFW_KEY_5, 5},
            {GLFW_KEY_6, 6},
            {GLFW_KEY_7, 7}};
        const int nBinds = sizeof(kBinds) / sizeof(kBinds[0]); // count from the table itself -> prevState size + loop bound track it; adding a preset = one kBinds row
        // Previous-frame state of each bind's key (parallel to kBinds), for the
        // RELEASE->PRESS edge detection in the loop below. Sized nBinds so it always
        // matches the table; any slots left unlisted would zero-init to GLFW_RELEASE (0).
        int prevState[nBinds] = {
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
            GLFW_RELEASE,
        };

        // P1 auto-play state (Presentation track). autoPlay off = manual control (Space
        // toggles it). presetTimer accumulates real dt in auto mode; when it passes
        // presetInterval the sim advances to the next preset. currentPreset is the app's
        // single notion of the active preset, so the auto-cycler, the number keys, and
        // (later) the ImGui menu all agree on one value.
        bool autoPlay = false;
        float presetTimer = 0.0f;
        int currentPreset = 1; // must match the preset the ctor boots (set_preset(1) =
                               // fireworks); otherwise the app's notion disagrees with the
                               // screen and the first auto-cycle would step to fireworks again.
        const float presetInterval = 10.0f; // seconds each preset holds in auto mode (~8-12)

        int prevSpace = GLFW_RELEASE; // previous-frame Space state, for the RELEASE->PRESS
                                      // edge detection below (same idea as prevState[] for
                                      // the number keys)
        // P2a fullscreen state. fullscreen = current mode. savedX/Y/W/H remembers the
        // windowed rectangle so F11 can restore the exact same window on the way back.
        // prevF11 = edge-detect state (same debounce pattern as the number keys / Space).
        bool fullscreen = false;
        int savedX, savedY, savedW, savedH;
        int prevF11 = GLFW_RELEASE;

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

            if (dt > 0.05f) // clamp a hitch (window drag / breakpoint) so particles don't teleport
            {
                dt = 0.05f;
            }

            // P1: in auto-play, accumulate real frame time and, once a full interval has
            // elapsed, advance to the next preset (wrapping with % nBinds) and reset the
            // timer. Uses dt (wall-clock) rather than a frame count, so the interval stays
            // ~10 s at any FPS. nBinds doubles as the preset count here (one hotkey per
            // preset); P3 will replace it with a real preset_count() when the menu needs names.
            if (autoPlay)
            {
                presetTimer += dt;
                if (presetTimer >= presetInterval)
                {
                    currentPreset = (currentPreset + 1) % nBinds;
                    sim.set_preset(currentPreset);
                    presetTimer = 0.0f;
                }
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

            // Hotkeys switch the effect preset live. Table-driven (kBinds) so the
            // logic is written once and looped over every key; adding a preset is one
            // table row. EDGE-DETECTED (debounce): glfwGetKey returns the current level
            // (PRESS while held), which at thousands of FPS would re-fire set_preset --
            // and its cudaMemcpyToSymbol -- every frame a key is down. Comparing this
            // frame's state to the last (prevState[k]) fires only on the RELEASE->PRESS
            // transition, so one physical press = one switch. A switch re-uploads that
            // preset's emitters + physics; existing particles adopt the new look only as
            // they recycle, so it fades in over ~1 lifetime. GLFW_KEY_* is the PHYSICAL
            // key, independent of Shift / Caps Lock -- so J is naturally case-insensitive.
            for (int k = 0; k < nBinds; ++k)
            {
                int now = glfwGetKey(window, kBinds[k].key);
                if (now == GLFW_PRESS && prevState[k] == GLFW_RELEASE)
                {
                    sim.set_preset(kBinds[k].preset);
                    currentPreset = kBinds[k].preset; // P1: keep the app's notion in sync with the manual pick
                    autoPlay = false;                 // P1: a manual pick drops back to manual so auto never yanks it away
                }
                prevState[k] = now;
            }

            // P1: Space toggles auto <-> manual, edge-detected like the number keys so one
            // physical press = one toggle (not thousands while the key is held). Reset the
            // timer either way so entering auto starts a fresh interval.
            int nowSpace = glfwGetKey(window, GLFW_KEY_SPACE);
            if (nowSpace == GLFW_PRESS && prevSpace == GLFW_RELEASE)
            {
                autoPlay = !autoPlay;
                presetTimer = 0.0f;
            }
            // Remember this frame's Space state so the edge test fires once per physical
            // press, not every frame the key is held (the Space counterpart of the number
            // keys' prevState[k] = now).
            prevSpace = nowSpace;
            // P2a: F11 toggles windowed <-> fullscreen (edge-detected, one press = one toggle).
            int nowF11 = glfwGetKey(window, GLFW_KEY_F11);
            if (nowF11 == GLFW_PRESS && prevF11 == GLFW_RELEASE)
            {
                fullscreen = !fullscreen;
                if (fullscreen)
                {
                    // Save the windowed rectangle BEFORE leaving windowed mode, so we can
                    // restore this exact position + size later.
                    glfwGetWindowPos(window, &savedX, &savedY);
                    glfwGetWindowSize(window, &savedW, &savedH);
                    GLFWmonitor *mon = glfwGetPrimaryMonitor();        // the main display
                    const GLFWvidmode *mode = glfwGetVideoMode(mon);   // its native resolution + refresh (auto-detected)
                    // Passing a monitor = exclusive fullscreen at the monitor's native mode.
                    glfwSetWindowMonitor(window, mon, 0, 0,
                                         mode->width, mode->height, mode->refreshRate);
                }
                else
                {
                    // Passing nullptr = back to a windowed window, at the saved rectangle.
                    glfwSetWindowMonitor(window, nullptr,
                                         savedX, savedY, savedW, savedH, 0);
                }
                glfwSwapInterval(0); // re-assert vsync-off: a monitor switch can reset it (would re-cap FPS at 60)
                // (the framebuffer-size callback fires on this switch and re-squares the viewport for us)
            }
            prevF11 = nowF11; // remember for next frame's edge test

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
