# Phase 4 — Audio track

> **Procedurally generated, zero-license** sound layered onto the finished simulation. The
> goal is an **audio-reactive** demo: a scalar sampled from the live GPU sim each frame
> drives the sound, tying the particles to what you hear. Code lives in the same
> [`src/`](src/) as the sim. See the CUDA levels in **[README.md](README.md)** and the UI
> shell in **[PRESENTATION.md](PRESENTATION.md)**.

**Ordering.** Do this track **last** — after the [Presentation & UX track](PRESENTATION.md)
— so there is (a) a menu to mount volume / mute controls on, (b) an auto-play loop to sync
to, and (c) rich sim state to sample for the reactive layer. This is the 4th skill track of
the project (after C++ / CUDA / OpenGL); it was "Phase 5" in the roadmap and is folded here
as a track on the same app.

---

## Zero-license rule *(decided up front)*

**All audio is generated in code or from self-authored tools — no downloaded sound files
in the repo.** This keeps the public repository asset-free and licence-clean.

- **Event SFX** (whoosh / chime): synthesize in C++, **or** author with an sfxr-family tool
  (bfxr / jsfxr) whose exported output is your own work (no licence).
- **Reactive layer:** fully code-synthesized — generate sine / noise buffers in C++ and
  modulate volume / pitch from the sim scalar; miniaudio just plays the generated buffer.
- Only fall back to **CC0** files (e.g. Kenney / Pixabay) if a sound is genuinely
  impractical to synthesize, and add a `CREDITS` note if you do.

---

## Engine: miniaudio

Use **[miniaudio](https://miniaud.io/)** — a single-header, zero-dependency C library that
drops into the CMake build the same way GLFW / glad do (`FetchContent`, or just vendor the
one header). It gives you a playback device + callback; you fill the callback with samples,
whether pre-generated or synthesized on the fly.

---

## Tiers — build in order

### T1 — Event sounds

One-shot SFX triggered by discrete events, to prove the audio path end to end.

- A **preset-switch chime** (fired from the same code path as P1/P3 preset changes) and an
  **emitter / launch whoosh**.
- Generate the buffers once at startup (a short enveloped sine/blip for the chime, filtered
  noise for the whoosh) and play them on demand.
- Wire a **mute** toggle and a **volume** slider into the Presentation menu.

*Deliverable:* switching presets makes a sound; the menu controls it.

### T2 — Ambient per-preset beds

A continuous background layer whose character matches the current look.

- One looping bed per preset — e.g. fireworks *crackle*, fire *roar*, galaxy *drone* —
  synthesized (filtered noise + slow LFOs), not sampled.
- **Cross-fade** between beds when the preset changes (manual, or auto-play's timer), so the
  soundscape follows the visuals without hard cuts.

*Deliverable:* each preset has its own evolving ambient sound; changing preset cross-fades.

### T3 — Audio-reactive *(the goal)*

Close the loop: let the **simulation drive the sound**.

- Each frame, extract a **scalar** from the sim — e.g. total kinetic energy, or the count of
  particles near the gravity well / attractor. This is the *same kind* of GPU read-back the
  Presentation HUD's "hacker-mode" stats use (a small reduction / atomic + a few-`float`
  D2H), so the two tracks can **share one telemetry read-back** rather than doing it twice.
- Map that scalar onto audio parameters: **volume**, **pitch**, filter cutoff, or the
  density of synthesized grains — so a bright energetic burst *sounds* bright and energetic.
- Optionally feed the scalar back to **auto-play** so preset advance / visual pulses land on
  the beat, making sight and sound feel like one system.

*Deliverable:* the sound changes moment-to-moment with what the particles are doing —
the GPU sim and the audio are one instrument.

---

## Ties to the other tracks

- **Presentation menu** hosts the mute / volume controls (T1) and, if you add them, per-bed
  levels (T2).
- **Auto-play** (P1) can drive T2 cross-fades on preset change and, in T3, advance on the
  beat.
- **The reactive scalar (T3) is the same GPU read-back** as the HUD's optional hacker-mode
  telemetry — implement it once, use it for both the numbers on screen and the sound.

---

## Per-preset sound design *(T2 planning — record intentions here as they're decided)*

The T2 bed for each preset should be **derived from that preset's look**, not chosen at
random. Read the visual on four axes and map each to a sound axis:

| Read the visual | Maps to sound |
| --- | --- |
| **Motion** — sparse bursts vs. continuous flow | one-shot hits (chime/crackle) vs. a **sustained drone/pad** |
| **Energy** — violent vs. gentle | loud/aggressive vs. soft, low level |
| **Colour temperature** — cold/hard vs. warm/soft | bright, buzzy (saw, open filter) vs. **warm** (sine/triangle, low-pass) |
| **Signature motion** — swirl, fall, explosion… | a matching modulation (rotating pan, downward glide, noise burst) |

### 0 — Jia *(decided)*

Two slow diagonal jets (pink + gold) braided by a gentle counter-clockwise swirl; no
gravity, long life, low speed → **continuous, warm, meditative**, not percussive.

**Bed: a warm, slowly-rotating two-voice drone.**

- **Two sine oscillators** — one per jet (pink / gold); a soft interval (e.g. root + fifth).
- **Detune them a few Hz** so they **beat** ("waah…waah…") — the acoustic image of the two
  ribbons winding around each other.
- **Slow pan LFO** (~5–10 s period) drifts the whole sound between L and R — the audible
  form of the swirl vortex.
- **Low-pass filter** to keep it warm (matches pink/gold), no bright harmonics; optionally a
  very slow amplitude LFO for a breathing swell that suits the long particle life.

Building blocks to synthesize later: oscillator, detune → beating, LFO, low-pass filter.

### 1–7 — *(to sketch as T2 is built)*

fireworks → sparse crackle · fire/smoke → low roar · galaxy (n-body) → airy drone ·
rain → continuous white-noise hiss · Lorenz → drifting, slowly-swelling tone.

---

## Progress

- [ ] T1 Event SFX — synthesized chime (preset switch) + whoosh (launch); mute + volume in
      the menu
- [ ] T2 Ambient beds — one synthesized loop per preset, cross-fade on preset change
- [ ] T3 Audio-reactive — per-frame sim scalar (kinetic energy / particles near well) drives
      volume / pitch; shares the HUD's GPU read-back; optional beat-synced auto-play
