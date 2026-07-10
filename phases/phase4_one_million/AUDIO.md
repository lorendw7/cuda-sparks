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

### 0 — Jia *(decided — option D: music-box over a drone)*

Two slow diagonal jets (pink + gold) braided by a gentle counter-clockwise swirl; no
gravity, long life, low speed → **continuous, warm, meditative**, not percussive. Jia is
named for a person, so it gets the most-considered, most-memorable treatment.

**Bed: a quiet warm drone with a sparse pentatonic music-box melody drifting over it.**

*Two layers:*

**Layer 1 — the drone (the base, = a minimal version of the "warm beating drone"):**

- **Two sine oscillators** — one per jet (pink / gold); a soft interval (e.g. root + fifth),
  **detuned a few Hz** so they **beat** ("waah…waah…") = the two ribbons winding around
  each other. Kept **quiet** — it's a bed under the melody, not the star.
- **Slow pan LFO** (~5–10 s) drifts it between L and R = the audible form of the swirl.
- **Low-pass filter** for warmth (matches pink/gold, no bright harmonics).

**Layer 2 — the music box (what makes it option D):**

- A **sparse melody of enveloped-sine "ding" bells** from a **pentatonic scale** (5-note,
  no semitones → every note consonant, never sour; ideal for algorithmic/random note pick).
- Notes are **triggered slowly** (one every ~1–3 s), the pitch chosen randomly from the
  scale; tie the trigger to the **swirl phase** so the melody breathes with the rotation.
- Each bell is one sine through a fast-attack / slow-decay **envelope (ADSR-ish)** so it
  "dings" and rings out — the enveloping is exactly what turns a flat tone into a bell/pluck.

Building blocks to synthesize later: oscillator, detune → beating, LFO, low-pass filter,
**envelope (attack/decay)**, and a **tiny note sequencer** (a timer that fires a bell and
picks a pentatonic pitch). Layers 1 (T2 bed) + 2 (event-triggered) share the same output.

### 1–7 — *(optimized designs; detail during T2)*

Same visual→sound method as Jia. Rule of thumb: **episodic looks (discrete events) get
percussive / burst sounds; continuous looks get a sustained bed + slow modulation.**

- **1 Fireworks** *(episodic → event-driven, ties to T1)* — silence in the dark gap, then per
  shell: a low **"thump"** on launch (short enveloped low sine) + a **crackle** on burst
  (band-passed noise burst, fast decay, a few random pops). Reactive scalar = live-shell count.
- **2 Fire** *(continuous)* — a **low roar**: brown/low-passed noise + slow random amplitude
  **flicker** + a very low **sub rumble**; occasional crackle pops. Warm.
- **3 Galaxy** *(continuous)* — an **airy, cold drone** (deliberate contrast to Jia's warmth):
  detuned high sines forming a thin pad + a deep **sub** for cosmic depth + slow swirl-panning.
- **4 Rain** *(continuous)* — a **hiss**: band-passed white/pink noise tuned to a rain band +
  slight stereo spread + occasional low resonant **"drip"**; the wind-slant nudges the pan.
- **5 Smoke** *(continuous)* — a **soft breath**: low-passed noise, quieter/duller than fire
  (no crackle), very slow swell, filter cutoff **rising over time** = smoke lifting.
- **6 Curl-Noise** *(continuous)* — a **morphing pad**: a detuned drone with a slow filter /
  resonance sweep driven by **LFOs at incommensurate periods**, so it **never repeats** —
  the audible echo of the field's organic, non-repeating swirls.
- **7 Lorenz attractor** *(chaotic → the natural T3 pick)* — a **drifting tone** whose **pitch**
  tracks the attractor's jumps between its two lobes (sample-and-hold-ish glide) + a slow
  swell. Best driven directly by the live sim scalar → sonified chaos.

---

## Progress

- [x] **T0 Foundation** — miniaudio playback device + a continuous 440 Hz test sine
      (phase accumulation) in `audio.{h,cpp}`; `audio_init` / `audio_shutdown` bracket main's
      render loop (init is non-fatal — runs silently if no device). The "audio Hello World":
      proves the OS audio path device → thread → callback → samples → speakers.
- [ ] T1 Event SFX — synthesized chime (preset switch) + whoosh (launch); mute + volume in
      the menu
- [ ] T2 Ambient beds — one synthesized loop per preset, cross-fade on preset change
- [ ] T3 Audio-reactive — per-frame sim scalar (kinetic energy / particles near well) drives
      volume / pitch; shares the HUD's GPU read-back; optional beat-synced auto-play
