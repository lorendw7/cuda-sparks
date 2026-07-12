#pragma once // include guard: expand this header at most once per translation unit

// Phase 4 Audio track -- public interface. Implementation (device, callback, chime synthesis)
// lives in audio.cpp; only these entry points are exposed. Call audio_init() once before the
// render loop, audio_shutdown() once after; audio_play_chime() on each preset switch.
bool audio_init();       // open + start the playback device; returns true on success
void audio_shutdown();   // stop + release the device (pairs with audio_init)
void audio_play_chime(); // T1: fire a one-shot chime (call from the preset-switch path)