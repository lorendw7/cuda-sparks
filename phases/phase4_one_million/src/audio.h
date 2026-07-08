#pragma once // include guard: expand this header at most once per translation unit

// Phase 4 Audio track -- public interface. Implementation (the device, callback, and sine
// synthesis) lives in audio.cpp; only these two entry points are exposed. Call audio_init()
// once before the render loop and audio_shutdown() once after it.
bool audio_init();     // open + start the playback device; returns true on success
void audio_shutdown(); // stop + release the device (pairs with audio_init)
