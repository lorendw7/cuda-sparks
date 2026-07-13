// Phase 4 (L2-L6 + Presentation P1-P3) -- window + GL context, then run the 1M-particle
// sim each frame. CUDA-GL interop: the kernel writes vertices straight into the VBO (no
// CPU round trip). Hotkeys switch effect presets: J = Jia, 1 = fireworks (boot default),
// 2 = fire, 3 = galaxy, 4 = rain, 5 = smoke, 6 = curl-noise, 7 = strange attractor.
// Presentation shell: Space toggles hands-off auto-cycling (advances a preset every ~10 s,
// any number key drops back to manual); F11 toggles fullscreen (square sim + a two-panel
// telemetry console over the leftover strip; windowed shows one control bar instead);
// H hides the Dear ImGui menu. SPARKS_MAX_FRAMES caps the loop for Nsight replay.
// glad must be included before glfw.
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio>            // fprintf / printf
#include "renderer.h"        // our hand-written rendering layer
#include "particle_system.h" // the 1M-particle GPU simulation
#include <chrono>            // high_resolution_clock for the per-frame timing
#include <cstdlib>           // getenv, atol
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "hud_log.h"
#include "audio.h" // Phase 4 audio track: playback device + synthesized tone

// P2a: keep the particle viewport a SQUARE so NDC [-1,1]^2 never stretches (dots stay
// round on any aspect ratio). GLFW calls this whenever the framebuffer resizes -- window
// drag, or the F11 fullscreen toggle. width/height are the new size in real pixels.
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
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
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // create the window HIDDEN -- position it below, THEN show
                                              // it, so it never flashes at GLFW's default spot first

    // P2a+: size and place the window on the primary monitor's WORK AREA (the desktop minus
    // the taskbar), so it always fits and centers on any screen. Query the monitor BEFORE
    // creating the window -- monitor functions don't need a window to exist.
    GLFWmonitor *mon = glfwGetPrimaryMonitor();
    int mx, my, mw, mh;
    glfwGetMonitorWorkarea(mon, &mx, &my, &mw, &mh); // work area: origin (mx,my) + size (mw,mh)
    const int margin = 80;                           // breathing room so the window never touches the edges
    int side = 1280;                                 // desired square edge (matches P2a's square viewport)
    int maxSide = (mw < mh ? mw : mh) - margin;      // largest square the work area can hold
    if (side > maxSide)                              // only shrink when the screen is too small to fit 1280
    {
        side = maxSide;
    }

    // 3) Create the window at the computed square size. nullptr window means creation failed.
    GLFWwindow *window = glfwCreateWindow(side, side, "cuda-sparks | Phase 4", nullptr, nullptr);

    if (window == nullptr) // validate the handle BEFORE using it -- the calls below would crash on NULL
    {
        glfwTerminate();
        return 1;
    }

    // Window is known-good: center it on the work area, then reveal it (it was created hidden).
    glfwSetWindowPos(window, mx + (mw - side) / 2, // horizontal center (side <= mw -> never negative)
                     my + (mh - side) / 2);        // vertical center
    glfwShowWindow(window);                        // now show it -> no flash at GLFW's default position

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

        // P3: bring up Dear ImGui on our live GLFW + OpenGL3 context. Init here (inside the
        // GL-context scope) so shutdown can also run before the context is destroyed.
        IMGUI_CHECKVERSION();                       // header vs linked-lib version guard
        ImGui::CreateContext();                     // the global UI state lives here
        ImGui_ImplGlfw_InitForOpenGL(window, true); // input backend; true = install + chain GLFW callbacks
        ImGui_ImplOpenGL3_Init("#version 330");     // render backend; GLSL version matches our GL 3.3

        // P3: load a crisp 22px TTF instead of scaling the tiny built-in bitmap font.
        // SPARKS_ASSET_DIR (a CMake -D macro holding the assets ABSOLUTE path) is
        // string-literal-concatenated with the filename, so the font loads regardless of
        // the working directory. The first font added becomes the default.
        ImGuiIO &io = ImGui::GetIO();
        io.Fonts->AddFontFromFileTTF(SPARKS_ASSET_DIR "/fonts/Roboto-Medium.ttf", 22.0f);
        io.FontGlobalScale = 1.0f; // 1.0 -- the font is rasterized at the right size already

        // P3: soften the default look -- rounded corners + roomier padding/spacing.
        ImGuiStyle &style = ImGui::GetStyle();
        style.FrameRounding = 4.0f;         // button/checkbox corner radius
        style.WindowRounding = 6.0f;        // panel corner radius
        style.FramePadding = ImVec2(10, 6); // space inside buttons
        style.ItemSpacing = ImVec2(8, 6);   // space between widgets

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

        // P3-2: button labels for the preset picker. Indexed by preset id (0..nBinds-1),
        // same order as the presets[] table in particle_system.cu.
        const char *presetNames[] = {
            "Jia", "Fireworks", "Fire", "Galaxy",
            "Rain", "Smoke", "Curl-Noise", "Attractor"};

        // P1 auto-play state (Presentation track). autoPlay off = manual control (Space
        // toggles it). presetTimer accumulates real dt in auto mode; when it passes
        // presetInterval the sim advances to the next preset. currentPreset is the app's
        // single notion of the active preset, so the auto-cycler, the number keys, and
        // (later) the ImGui menu all agree on one value.
        bool autoPlay = false;
        bool audioMuted = false;
        float audioVolume = 0.7f;
        float presetTimer = 0.0f;
        int currentPreset = 1;        // must match the preset the ctor boots (set_preset(1) =
                                      // fireworks); otherwise the app's notion disagrees with the
                                      // screen and the first auto-cycle would step to fireworks again.
        float presetInterval = 10.0f; // seconds each preset holds in auto mode (~8-12)
        HudLog hud;
        float logTimer = 0.0f;
        int logSeq = 0;

        int prevSpace = GLFW_RELEASE; // previous-frame Space state, for the RELEASE->PRESS
                                      // edge detection below (same idea as prevState[] for
                                      // the number keys)
        // P2a fullscreen state. fullscreen = current mode. savedX/Y/W/H remembers the
        // windowed rectangle so F11 can restore the exact same window on the way back.
        // prevF11 = edge-detect state (same debounce pattern as the number keys / Space).
        bool fullscreen = false;
        int savedX, savedY, savedW, savedH;
        int prevF11 = GLFW_RELEASE;

        // P3-2: H toggles the whole menu on/off so you can watch the effect unobstructed
        // (the panel is an opaque overlay). showUI = visible; prevH = edge-detect state.
        bool showUI = true;
        int prevH = GLFW_RELEASE;

        // P3-4: throttled readout snapshot. io.Framerate updates every frame, so at
        // thousands of FPS the digits flicker unreadably; we sample it into uiFps/uiMs
        // twice a second and display the held copy. lastUiTime = last sample time.
        float uiFps = 0.0f;
        float uiMs = 0.0f;
        auto lastUiTime = std::chrono::high_resolution_clock::now();

        // P3-5: ONE shared "switch preset" path for the keyboard, the auto-cycler, and the
        // menu, so they can never disagree about currentPreset / autoPlay. A lambda (not a
        // free function) because it needs main's locals -- [&] captures them by reference so
        // it can write them. manual=true = a human picked it -> leave auto-play (and reset
        // the timer); manual=false = the auto-cycler advancing, which stays in auto (and
        // resets the timer itself at the call site, since we don't reset it here).
        auto selectPreset = [&](int i, bool manual)
        {
            sim.set_preset(i); // upload the preset's emitters + physics
            audio_play_chime();
            currentPreset = i; // keep the app's single notion of the active preset in sync
            if (manual)
            {
                autoPlay = false;   // manual pick drops back to manual
                presetTimer = 0.0f; // and starts a fresh interval
            }
        };

        // Audio track: open + start the playback device BEFORE the render loop, so the
        // synthesized tone plays for the loop's whole lifetime. Independent of the GL context,
        // but bracketed around the loop here (with audio_shutdown after it) for clarity.
        // NON-FATAL on failure -- unlike glfwInit above, a missing/held audio device just
        // means "run silently", so we warn and carry on instead of returning.
        if (!audio_init())
        {
            fprintf(stderr, "audio init failed -- running without sound.\n");
        }

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

            // P2b: throttle -- emit ONE log line per interval, NOT per frame. At 3000 FPS a
            // per-frame push would fill the 128-line buffer in a single instant. The interval
            // is pure PACING (scroll speed), not performance: bigger = calmer, smaller = busier.
            logTimer += dt;      // accumulate real frame time
            if (logTimer >= 0.3) // ...once a full interval has passed (~1.6 lines/sec here)
            {
                logTimer = 0.0f;                               // reset -> wait another full interval before the next line
                const ImU32 green = IM_COL32(0, 255, 65, 255); // classic terminal green
                int blocks = (sim.size() + 255) / 256;         // ceil(n/256): grid size for a 256-thread block

                // Rotate through 4 telemetry templates -- one per emit. logSeq (bumped just
                // AFTER the switch) walks 0->1->2->3->0..., so the stream reads varied, not
                // one repeated line. Each case needs its own { } -- case labels share one
                // scope, so a local declared bare in one case clashes with the next.
                switch (logSeq % 4)
                {
                case 0: // GPU: kernel launch config + this frame's cost
                {
                    hud.push(green, "[GPU] KERNEL update<<<%d,256>>>  n=%d  %.2fms", blocks, sim.size(), uiMs);
                    break;
                }
                case 1: // MEM: vertex bytes streamed to the VBO this frame (5 floats x 4 bytes each)
                {
                    float mem = sim.size() * 20 / 1.0e6; // total bytes / 1e6 -> MB (1.0e6 forces a float divide)
                    hud.push(green, "[VBO] %d verts %.2f MB/frame", sim.size(), mem);
                    break;
                }
                case 2: // SIM: active preset, frame rate, running frame index
                {
                    hud.push(green, "[SIM] preset Name: %s  fps: %.0f frame number: %ld", presetNames[currentPreset], uiFps, frameCount);
                    break;
                }
                case 3: // SIM: physics step size + static config knobs
                {
                    hud.push(green, "[SIM] dt=%.4fs  shells=%d  bound=%.1f",
                             dt, params.numShells, params.bound);
                    break;
                }
                default:
                    break;
                }
                logSeq++; // advance so the NEXT emit picks the next template (wraps via % 4)
            }

            // P3-4: refresh the readout snapshot only every 0.5 s (throttle) so the digits
            // are legible. Same timer pattern as the once-per-second printf below.
            if (std::chrono::duration<double>(t0 - lastUiTime).count() > 0.5)
            {
                uiFps = io.Framerate;
                uiMs = 1000.0f / io.Framerate;
                lastUiTime = t0;
            }

            // P3-2: H toggles menu visibility (edge-detected). NewFrame/Render still run
            // every frame below -- only the panel's Begin/End is gated on showUI -- so a
            // hidden frame just declares no windows (an empty UI layer), not a missing frame.
            int nowH = glfwGetKey(window, GLFW_KEY_H);
            if (nowH == GLFW_PRESS && prevH == GLFW_RELEASE)
            {
                showUI = !showUI;
            }
            prevH = nowH;

            // P3: open a new ImGui frame. Everything declared between NewFrame() and
            // Render() (below) is THIS frame's UI -- immediate mode rebuilds it every frame.
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // P2b/P3 layout geometry, recomputed each frame (the framebuffer_size_callback's
            // `side` is a dead local by now). io.DisplaySize == the framebuffer size in px,
            // kept live by the ImGui backend, so these stay correct across an F11 toggle.
            const float W = io.DisplaySize.x; // framebuffer width  (px)
            const float H = io.DisplaySize.y; // framebuffer height (px)
            const float sq = W < H ? W : H;   // square sim edge = shorter side (same formula as the callback)
            const bool landscape = (W >= H);  // wide screen? true -> HUD strip on the right, false -> below

            // ── UI content, extracted into 3 lambdas so BOTH layouts (windowed top-bar and
            //    the fullscreen side panels) call ONE copy of each widget group -- no dup.
            //    [&] captures main's state by reference, same mechanism as selectPreset. ──

            // Content 1: preset picker. horizontal=true -> one row (top bar); false -> stacked
            // (the narrow fullscreen strip, where 8 buttons in a row would overflow and clip).
            auto drawPresetPicker = [&](bool horizontal)
            {
                ImGui::Text("Preset");
                for (int i = 0; i < nBinds; ++i)
                {
                    bool active = (i == currentPreset);
                    if (active)
                    {
                        // Highlight the active button: gold background + black text. Two Push =
                        // two Pop below (the color stack must stay balanced or the color leaks).
                        ImGui::PushStyleColor(ImGuiCol_Button,
                                              ImVec4(1.0f, 0.84f, 0.0f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f)); // black text
                    }

                    if (horizontal && i > 0) // horizontal -> SameLine packs buttons into a row;
                    {                        // else (stacked) each button drops to its own line
                        ImGui::SameLine();
                    }

                    if (ImGui::Button(presetNames[i])) // true only on the frame it's clicked
                    {
                        selectPreset(i, true);
                    }

                    if (active)
                        ImGui::PopStyleColor(2); // pop the 2 colors pushed above
                }
            };

            // Content 2: auto-play controls (checkbox + interval slider), bound to the P1 state.
            auto drawAutoPlay = [&]
            {
                if (ImGui::Checkbox("Auto-play", &autoPlay))
                {
                    presetTimer = 0.0f;
                }
                // The slider edits `presetInterval` in place (it needed the `const` dropped);
                // the auto-cycler reads it live, so dragging changes the cycle speed at once.
                ImGui::SetNextItemWidth(200.0f);
                ImGui::SliderFloat("Interval (s)", &presetInterval, 3.0f, 60.0f, "%.1f");
            };

            // Content 2b: audio controls -- mute checkbox + volume slider. Each pushes its new
            // value into the audio engine's atomics (audio_set_muted / audio_set_volume) ONLY on
            // the frame ImGui reports a change (its return value), so we don't spam the setters.
            auto drawAudio = [&]
            {
                if (ImGui::Checkbox("Mute", &audioMuted))
                {
                    audio_set_muted(audioMuted);
                }

                ImGui::SetNextItemWidth(200.0f);
                if (ImGui::SliderFloat("Volume", &audioVolume, 0.0f, 1.0f, "%.2f"))
                {
                    audio_set_volume(audioVolume);
                }
            };

            // Content 3: live readouts (FPS / particle count / active preset name).
            auto drawReadouts = [&]
            {
                ImGui::Text("FPS: %.0f (%.2f ms)", uiFps, uiMs);
                ImGui::Text("Particles: %d", sim.size());
                ImGui::Text("Preset %s", presetNames[currentPreset]);
            };

            // P3/P2b layout branch -- one shared UI state rendered three ways:
            //   hidden (H)         -> no windows this frame
            //   windowed (default) -> the full-width control bar pinned below
            //   fullscreen (F11)   -> a two-panel console over the leftover strip
            // All three reuse the drawXxx() lambdas above, so each widget exists in one copy.
            // The next two lines pin the WINDOWED bar (pos (0,0), width = full display,
            // height 0 = auto-fit); ImGuiCond_Always + NoMove/NoResize keep it from drifting.
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 0), ImGuiCond_Always);
            if (!showUI)
            {
                // H hid the menu -> declare no windows this frame (NewFrame/Render still run).
            }
            else if (!fullscreen)
            {
                ImGui::Begin("cuda-sparks", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
                drawPresetPicker(true);
                ImGui::Separator();
                drawAutoPlay();
                ImGui::Separator();
                drawAudio();
                ImGui::Separator();
                drawReadouts();
                ImGui::Separator();
                hud.draw();
                ImGui::End();
            }
            else
            {
                // FULLSCREEN: the square sim sits on one edge (P2a); the leftover strip becomes
                // a two-panel hacker console. Strip rect derived from the square edge `sq`:
                // landscape -> right band (full height); portrait -> bottom band (full width).
                const float stripX = landscape ? sq : 0.0f;
                const float stripY = landscape ? 0.0f : sq;
                const float stripW = landscape ? (W - sq) : sq;
                const float stripH = landscape ? sq : (H - sq);
                const float halfH = stripH * 0.5f;

                // Console skin: near-black bg + terminal-green text, shared by both panels.
                // 2 pushes here -> 2 pops after the second End() (the color stack must balance).
                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.02f, 0.95f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.25f, 1.0f));

                // Top half: PERF MONITOR -- the fast-moving readouts.
                ImGui::SetNextWindowPos(ImVec2(stripX, stripY), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(stripW, halfH), ImGuiCond_Always);
                ImGui::Begin("PERF MONITOR", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse);
                drawReadouts();
                ImGui::End();

                // Bottom half: PARTICLE INFO -- controls fold in here + the scrolling log.
                ImGui::SetNextWindowPos(ImVec2(stripX, stripY + halfH), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(stripW, halfH), ImGuiCond_Always);
                ImGui::Begin("PARTICLE INFO", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
                drawPresetPicker(false);
                ImGui::Separator();
                drawAutoPlay();
                ImGui::Separator();
                drawAudio();
                ImGui::Separator();
                hud.draw();
                ImGui::End();

                ImGui::PopStyleColor(2);
            }

            // P1: in auto-play, accumulate real frame time and, once a full interval has
            // elapsed, advance to the next preset (wrapping with % nBinds) and reset the
            // timer. Uses dt (wall-clock) rather than a frame count, so the interval stays
            // ~10 s at any FPS. nBinds doubles as the preset count (one hotkey per preset).
            if (autoPlay)
            {
                presetTimer += dt;
                if (presetTimer >= presetInterval)
                {
                    selectPreset((currentPreset + 1) % nBinds, false); // manual=false: stay in auto
                    presetTimer = 0.0f;                                // REQUIRED: selectPreset only resets the timer for a
                                                                       // manual pick, so the auto path must clear it here --
                                                                       // else presetTimer stays >= interval and it re-switches
                                                                       // every frame (runaway).
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

            // P3: turn this frame's UI into draw data and paint it with OpenGL -- AFTER the
            // particles (menu on top), BEFORE SwapBuffers. ImGui uses its own full-window
            // projection, so it ignores P2a's square glViewport and can cover the whole window.
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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
            // P3-5: stand down while ImGui wants the keyboard (a widget focused / text
            // input), so a keystroke meant for the UI doesn't ALSO fire a preset hotkey.
            if (!io.WantCaptureKeyboard)
            {
                for (int k = 0; k < nBinds; ++k)
                {
                    int now = glfwGetKey(window, kBinds[k].key);
                    if (now == GLFW_PRESS && prevState[k] == GLFW_RELEASE)
                    {
                        selectPreset(kBinds[k].preset, true);
                    }
                    prevState[k] = now;
                }
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
                // Exclusive fullscreen bypasses the OS compositor -> the system mouse cursor
                // vanishes. Have ImGui draw a software cursor into the frame instead (fullscreen
                // only; windowed keeps the normal OS cursor).
                ImGui::GetIO().MouseDrawCursor = fullscreen;
                if (fullscreen)
                {
                    // Save the windowed rectangle BEFORE leaving windowed mode, so we can
                    // restore this exact position + size later.
                    glfwGetWindowPos(window, &savedX, &savedY);
                    glfwGetWindowSize(window, &savedW, &savedH);
                    GLFWmonitor *mon = glfwGetPrimaryMonitor();      // the main display
                    const GLFWvidmode *mode = glfwGetVideoMode(mon); // its native resolution + refresh (auto-detected)
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

        // Audio track: stop + release the device after the loop (mirror of audio_init above).
        audio_shutdown();

        // P3: tear down ImGui ONCE, after the loop, while the GL context is still alive
        // (reverse order of init; the same GL-context-ordering rule as ~Renderer below).
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    } // <- renderer destroyed here, GL context still alive

    // 6) Tear down the context AFTER the renderer is gone.
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
