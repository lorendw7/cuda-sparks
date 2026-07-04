# Phase 4 — Presentation & UX track

> An **application-shell** layer on top of the finished 1M simulation: an on-screen menu,
> a real fullscreen mode, and a hands-off auto-play loop. It is **not** a CUDA lesson —
> everything here is C++ / GLFW / OpenGL / Dear ImGui plumbing, **no kernel changes**.
> Code lives in the same [`src/`](src/) as the sim (no forked folder). See the CUDA levels
> in **[README.md](README.md)**; the sound layer in **[AUDIO.md](AUDIO.md)**.

**Ordering.** Do this track **after L6** (make the presets genuinely good-looking first — a
menu around ugly presets is polish on the wrong thing) and **before the audio track** (which
plugs directly into the menu + auto-play). L7 (FP16) is an optional stretch and does not
block this track.

Build the three pieces cheapest-first; each is usable on its own.

---

## P1 — Auto-play + manual toggle *(cheapest — no new dependency, do this first)*

A hands-off mode that cycles the presets on a timer, plus a clean way to switch back to
manual control.

- Add a `bool autoPlay` and a `float presetTimer`. In auto mode, accumulate the real
  frame `dt` (the loop already computes it); when the timer passes an interval
  (~8–12 s), call `set_preset((current + 1) % numPresets)` and reset the timer.
- Track `currentPreset` in the app so both the auto-cycler and (later) the menu agree on
  which preset is active.
- **Space** toggles auto ↔ manual. The number keys **1–4** still work and, when pressed,
  should also drop you back to manual — a manual pick shouldn't get yanked away two
  seconds later by the auto-cycler.
- The transition is already soft: a preset change fades in over ~1 lifetime as particles
  recycle, so auto-play never hard-cuts.

*Deliverable:* the sim runs itself as an ambient display, and you can grab manual control
with the keyboard at any time.

---

## P2 — Fullscreen + square viewport + telemetry strip *(pulls in the deferred window backlog)*

Turn the fixed 1280×1280 window into a proper fullscreen presentation: a square simulation
area sized to the screen, with the leftover space used as a hacker-style telemetry HUD.

### Fullscreen toggle

- `glfwGetPrimaryMonitor` + `glfwGetVideoMode` for the native resolution; toggle
  windowed ↔ fullscreen at runtime with **F11** via `glfwSetWindowMonitor` (pass the
  monitor for fullscreen; `nullptr` + the saved window rectangle to restore).
- Add the viewport handling that the square-window shortcut skipped:
  `glfwGetFramebufferSize` + `glViewport`, plus a `glfwSetFramebufferSizeCallback` so a
  resolution change / fullscreen toggle recomputes the layout.

### Square sim area — side = the shorter screen edge

- Take `side = min(width, height)` and give the particles a square
  `glViewport(0, 0, side, side)` anchored to one edge (left on a landscape monitor). This
  keeps NDC `[-1,1]²` **undistorted** — the dots stay round, with no stretch and no
  letterbox bars. On 1920×1080 → a 1080×1080 sim square on the left.
- `glViewport` only clips the particle `GL_POINTS` draw. Dear ImGui (P3) uses its own
  full-window orthographic projection, so the HUD panels can sit over the leftover strip
  independently of the sim viewport.
- Portrait monitors are the mirror case (`side = width`, sim square on top, HUD strip
  below); derive the split from `min(width, height)` so it works either way.

### The leftover strip = a hacker-style telemetry HUD

The remaining band (`x` from `side` to `width`, full height on a landscape monitor) is
split in half:

```text
┌────────────────────────┬──────────────────┐
│                        │  PERF MONITOR     │  top half
│                        │  FPS / frame ms   │
│     square sim area    │  update µs        │
│     side = min(W,H)    │  count / draws    │
│     dots stay round    │  ▁▂▃▅▇ sparkline   │
│                        ├──────────────────┤
│                        │  PARTICLE INFO    │  bottom half
│                        │  preset / emitters│
│                        │  gravity/swirl/…  │
│                        │  live, streaming  │
└────────────────────────┴──────────────────┘
        side × side            (W − side) strip
```

- **Top half — performance monitor:** FPS + frame time (ms), the `update_kernel` time
  (already measured as `accUpdate / frames`), particle count, draw count, and — optionally
  — GPU memory via `cudaMemGetInfo`. Make it feel alive with a rolling `ImGui::PlotLines`
  sparkline of the FPS / frame-time history.
- **Bottom half — live particle info:** current preset name, emitter count, and the live
  physics knobs (`gravity` / `swirl` / `nbodyStrength` / `damping`), refreshed every frame
  so the numbers stream.
- **Aesthetic:** monospace font, green-on-near-black, tight rows, fast-updating counters —
  a *readout console*, not a settings dialog. Push an ImGui style (`ImGuiCol_Text` green,
  dark `WindowBg`, no rounding) for these two panels.

### Cheap vs "hacker-mode" telemetry

Everything above is **host-known** — free to display, no GPU read-back. The genuinely
GPU-sourced stats ("alive particle count / average speed / recycles-per-second") live on
the **device**, so showing them needs a tiny per-frame **read-back**: a device atomic
counter or a small reduction, then a few-`float` D2H copy. That is a few **bytes**/frame —
nothing like L1's whole-array round trip — but it *is* a real GPU→CPU dependency. Treat the
GPU-sourced stats as an optional **hacker-mode** upgrade layered on top of the cheap panel,
not a requirement.

*Deliverable:* F11 gives a clean fullscreen demo — round dots in a square, live telemetry
console beside it — on any monitor aspect ratio.

---

## P3 — Dear ImGui menu *(the main new piece)*

An immediate-mode GUI for browsing presets and driving auto-play with the mouse, replacing
"remember which number key does what".

### Dependency & wiring

- Pull **Dear ImGui** via CMake `FetchContent` (same mechanism as GLFW / glad). ImGui
  isn't CMake-native, so you compile its sources into the target yourself: the core
  `imgui*.cpp` plus the two backends `imgui_impl_glfw.cpp` and `imgui_impl_opengl3.cpp`.
  One-time CMake wiring; after that it's immediate-mode UI. *(This is the one place in
  Phase 4 you integrate a mature library rather than hand-writing — everything else stays
  from scratch.)*
- **Frame integration:** `ImGui_ImplOpenGL3_NewFrame` + `ImGui_ImplGlfw_NewFrame` +
  `ImGui::NewFrame` at the top of the loop; build the panels; `ImGui::Render` +
  `ImGui_ImplOpenGL3_RenderDrawData` **after** `renderer.draw()` (menu on top), before
  `glfwSwapBuffers`. ImGui installs its own GLFW input callbacks — read
  `io.WantCaptureKeyboard` so a click in a widget doesn't also fire a preset hotkey.

### Panel contents (start minimal, grow)

- **Preset picker** — a button/combo per preset (fireworks / fire / galaxy / Jia), the
  active one highlighted. Replaces having to remember the number keys.
- **Auto-play controls** — a checkbox for auto ↔ manual and a slider for the cycle
  interval, wired to the **same state as P1**.
- **Live readouts** — current FPS and particle count.
- *(Stretch)* **live physics sliders** — `gravity` / `swirl` / `nbodyStrength` / count — so
  you can tune a look without recompiling. Ties into the `SPARKS_PARTICLES` count knob.

### Default interface vs fullscreen interface = two real layouts

Not just show/hide — the two modes present differently, over one shared state:

- **Windowed (default):** one compact **control panel** — preset picker + auto-play
  checkbox/interval + the physics sliders. A settings dialog you drive with the mouse.
- **Fullscreen:** the square sim on one edge (P2) with the **two-panel telemetry strip**
  (perf monitor on top, live particle info below) filling the leftover band — the
  hacker-console readout. The mouse-driven controls can fold into the strip or be summoned
  with a key (**Tab** / **H**) so a clean fullscreen demo is mostly readout.
- Both share the same underlying state (current preset, auto-play flag, params); only the
  *presentation* differs. Reuse **one** "draw ImGui" function that branches on the
  fullscreen flag rather than duplicating widget code.

*Deliverable:* a mouse-driven menu in windowed mode and a telemetry console in fullscreen,
both reading and writing the same sim state, so keyboard, menu, and auto-play never
disagree.

---

## Hand-off to the audio track

Once this shell exists, **[AUDIO.md](AUDIO.md)** hooks straight onto it:

- the menu gains a **volume slider** and a **mute** toggle;
- **auto-play** can advance presets **on the beat** instead of on a fixed timer;
- the same scalar the perf HUD already samples from the sim (kinetic energy, particles near
  the well) becomes the **audio-reactive** drive signal.

---

## Progress

- [ ] P1 Auto-play + manual toggle (`autoPlay` / `presetTimer`, Space toggles, number keys
      drop back to manual)
- [ ] P2 Fullscreen (F11) + square viewport (`side = min(W,H)`) + hacker-style telemetry
      strip (perf monitor / live particle info); optional GPU-read-back "hacker-mode" stats
- [ ] P3 Dear ImGui menu (FetchContent + backends; preset picker + auto-play controls +
      readouts; optional live physics sliders); windowed control-panel vs fullscreen
      telemetry-console layouts sharing one state
