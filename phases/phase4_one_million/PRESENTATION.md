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
- **(Enhancement) Randomized order.** Instead of the fixed `(current + 1) % numPresets`
  sequential walk, pick the next preset at random so the show never repeats the same loop.
  Draw a random index (`std::mt19937` / `std::uniform_int_distribution`, or `std::rand`),
  and **re-draw if it equals `current`** so you never "advance" to the preset already on
  screen (a self-transition would look like a stall). Optional: keep a short history to
  avoid recently-shown presets, or a shuffled-deck order (shuffle all presets, play through,
  reshuffle) so every preset shows once per round before any repeats. Make sequential vs
  random a small flag/menu toggle later in P3.

*Deliverable:* the sim runs itself as an ambient display, and you can grab manual control
with the keyboard at any time.

---

## P2 — Fullscreen + square viewport + telemetry strip *(pulls in the deferred window backlog)*

Turn the fixed 1280×1280 window into a proper fullscreen presentation: a square simulation
area sized to the screen, with the leftover space used as a hacker-style telemetry HUD.

> **Split by dependency.** The **telemetry HUD** needs to draw text / plots — i.e. Dear
> ImGui, which only arrives in **P3**. So P2 is built in two halves:
> **P2a** = fullscreen toggle + square viewport (pure GLFW/OpenGL, no new dependency —
> **done**), and **P2b** = the telemetry HUD, deferred to land **with P3's ImGui setup**.
> P2a already leaves the correct leftover strip empty and waiting for P2b.

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

### Refactor: one shared "switch preset" path

By P3 there are **three** things that change the active preset — the number-key handler,
the auto-play cycler, and now a menu click — and each must do the *same* bookkeeping:
call `sim.set_preset(i)`, update `currentPreset`, and (for a manual pick) set
`autoPlay = false`. P1 duplicates that logic across the keyboard loop and the cycler; a
menu button would be a third copy. **Extract it into one small helper** — a
`selectPreset(int i, bool manual)` lambda/function that keyboard, menu, and auto-play all
call — so the three input sources can never disagree about `currentPreset` / `autoPlay`
(e.g. forgetting to drop out of auto on one path). Do this refactor *when adding the menu*,
not before: with only two callers in P1 the duplication is cheap and premature extraction
adds indirection for no gain; the third caller (the menu) is what makes one shared path
clearly worth it.

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

- [x] P1 Auto-play + manual toggle — `autoPlay` / `presetTimer` / `currentPreset` state;
      Space edge-toggles auto ↔ manual (advances a preset every `presetInterval` ≈ 10 s via
      `(currentPreset + 1) % nBinds`); the number keys still switch presets and drop back to
      manual (sync `currentPreset`, clear `autoPlay`). `currentPreset` boots at 1 to match the
      ctor's `set_preset(1)`. *(Enhancement — randomized order — still open.)*
- [x] **P2a** Fullscreen (F11) + square viewport (`side = min(W,H)`) — `framebuffer_size_callback`
      re-squares the viewport on every resize/toggle; F11 edge-toggles windowed ↔ exclusive
      fullscreen via `glfwSetWindowMonitor` (saving/restoring the windowed rect), native
      resolution auto-detected from `glfwGetVideoMode`. Round dots on any aspect ratio; the
      leftover strip is left empty for the HUD.
- [x] **P2b** Hacker-style telemetry strip — in fullscreen (F11) the leftover band beside the
      square sim splits into two pinned panels: **PERF MONITOR** (FPS / frame-ms / particle
      count) on top, **PARTICLE INFO** (preset picker + auto-play + a green-on-black scrolling
      `HudLog` ring buffer) below. Near-black bg + terminal-green text via a shared
      `PushStyleColor` pair; strip rect derived from the square edge `sq` (right band on
      landscape, bottom band on portrait). *(Optional GPU-read-back "hacker-mode" stats + a
      `PlotLines` FPS sparkline still open.)*
- [x] P3 Dear ImGui menu — built in sub-steps (core + the windowed-vs-fullscreen layout split
      now done; live physics sliders remain the one optional stretch):
  - [x] **P3-1** ImGui wired up — FetchContent pulls Dear ImGui; a small static `imgui`
        library compiles its core + the GLFW/OpenGL3 backends; init/NewFrame/Render/Shutdown
        integrated into the loop (shutdown inside the GL-context scope). A test panel renders
        over the particles, confirming the pipeline.
  - [x] **P3-2** Preset picker panel — a full-width top bar with a button per preset
        (`presetNames[]`), the active one gold-highlighted; a click is a manual pick (same
        trio as the number keys: `set_preset` + sync `currentPreset` + clear `autoPlay`).
        Pinned top via `SetNextWindowPos/Size(ImGuiCond_Always)` + `NoMove/NoResize`, buttons
        laid out with `SameLine`. Crisp text from a bundled **Roboto** TTF (loaded via a
        CMake `SPARKS_ASSET_DIR` macro; Cousine kept for the P2b HUD). **H** toggles the whole
        menu so the effect can be viewed unobstructed.
  - [x] **P3-3** Auto-play controls — an `ImGui::Checkbox` bound to `&autoPlay` (stays in
        sync with the Space toggle automatically, since immediate mode re-reads the live
        value) and an `ImGui::SliderFloat` editing `presetInterval` in place (dropped its
        `const`); resets `presetTimer` on toggle, like Space. Plus a light global style pass
        (rounded corners, padding).
  - [x] **P3-4** Readouts — `ImGui::Text` lines for FPS + frame-ms, particle count, and the
        active preset name. FPS/ms use `io.Framerate` **throttled** to a 0.5 s snapshot
        (`uiFps`/`uiMs`) so the digits are legible at thousands of FPS; count/preset read
        live (they only change on action). Groundwork for the P2b telemetry HUD.
  - [x] **P3-5** Shared preset path + input hygiene — a `selectPreset(i, manual)` lambda
        (`[&]`-capturing main's state) now used by the keyboard, the auto-cycler
        (`manual=false`), and the menu (`manual=true`), so the `set_preset`+`currentPreset`
        +`autoPlay` trio lives in one place. Number-key hotkeys gated on
        `!io.WantCaptureKeyboard` so a keystroke for the UI doesn't also switch presets.
  - [x] **P3-6** Windowed/fullscreen layout split — the per-frame UI branches on the
        `fullscreen` flag over one shared state: windowed = a top control bar; fullscreen =
        the P2b two-panel console. The three widget groups (preset picker / auto-play /
        readouts) are extracted into `[&]` lambdas so both layouts call one copy each; the
        picker takes a `horizontal` flag (a row in the bar, stacked in the narrow strip).
        **H** hides all of it. *(Still optional: live physics sliders.)*
