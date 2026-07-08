// Phase 4 -- Audio track. Procedurally-synthesized sound for the sim (see AUDIO.md).
// Foundation step (T0): open a playback device and feed it a continuous 440 Hz sine tone,
// to prove the whole audio path end to end (device -> audio thread -> callback -> samples
// -> speakers) before layering on real SFX / beds / the reactive scalar.
//
// miniaudio is a single-header library: this ONE .cpp defines MINIAUDIO_IMPLEMENTATION so
// the whole implementation is compiled here (every OTHER file must include miniaudio.h
// WITHOUT the macro, or the symbols get duplicated at link time).
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <math.h> // sinf() -- synthesize the sine waveform

// Phase accumulator: the sine wave's CURRENT angle (radians). Each callback keeps advancing
// it from where the last one left off, so the tone is continuous across callbacks. MUST be
// file-scope static (lives for the whole program) -- a local would reset to 0 every callback
// and produce clicks/glitches instead of a smooth tone.
static float g_phase = 0.0f;

// The playback device handle. Opened by audio_init, read by data_callback, freed by
// audio_shutdown -- all three need it, so it's file-scope. static = internal linkage
// (private to this .cpp), so it never collides with symbols in other files.
static ma_device g_device;

// The audio thread calls this repeatedly to PULL data: each call hands us an empty output
// buffer we must fill with samples. The signature (types + order) is fixed by miniaudio --
// it must match exactly or the assignment to config.dataCallback below won't type-check.
// Runs on a SEPARATE thread from main(), so it must stay fast: no locks, no malloc, no I/O.
static void data_callback(ma_device *pDevice,
                          void *pOutput,        // [we fill this] output buffer, interleaved
                          const void *pInput,   // mic input; playback-only -> ignored
                          ma_uint32 frameCount) // how many FRAMES to fill this call (not samples)
{
    // pOutput is typeless; we requested ma_format_f32, so the buffer is really an array of
    // floats -- cast so we can write samples by index.
    float *out = (float *)pOutput;

    // Read the device's ACTUAL params (never assume our requested values were honored -- the
    // hardware may hand back a different channel count / rate, and using the wrong rate here
    // would detune the tone).
    ma_uint32 channels = pDevice->playback.channels; // interleaved stride (stereo = 2)
    float sampleRate = (float)pDevice->sampleRate;   // frames per second (e.g. 48000)

    // Per-frame phase increment dphi = 2*PI * f / sampleRate: how far the angle advances
    // between two adjacent samples. It's a constant, so compute it once outside the loop.
    const float PI2 = 6.28318530718f; // 2*PI = one full sine cycle
    float f = 440.f;                  // tone frequency; 440 Hz = concert-pitch A4
    float dphi = PI2 * f / sampleRate;

    // Outer loop: one iteration per FRAME. Inner loop: write that frame's value into every
    // channel of the interleaved buffer.
    for (ma_uint32 frame = 0; frame < frameCount; ++frame)
    {
        // This frame's sample. 0.2 is the amplitude (volume) -- keep it well below 1.0 to
        // stay quiet and avoid clipping (values outside [-1,1] get hard-limited = distortion).
        float value = 0.2f * sinf(g_phase);

        // Interleaved layout is [L0,R0, L1,R1, ...], so frame `frame` channel `ch` lives at
        // index frame*channels + ch. Writing the SAME value to every channel = a centered
        // mono tone.
        for (ma_uint32 ch = 0; ch < channels; ++ch)
        {
            out[frame * channels + ch] = value;
        }

        g_phase += dphi; // advance the angle by one sample's worth

        // Wrap: subtracting a whole turn leaves the sound identical (sin is 2*PI-periodic) but
        // keeps g_phase in [0, 2*PI) forever -- otherwise it grows without bound and sinf's
        // float precision degrades. -= (not =0) preserves the tiny overshoot, so no drift.
        if (g_phase >= PI2)
        {
            g_phase -= PI2;
        }
    }
}

// Open + start the playback device. Called once from main() BEFORE the render loop. Returns
// true on success; false lets main() carry on silently (audio failure is non-fatal).
bool audio_init()
{
    // Start from a filled-in default config for a playback device, then override the few
    // fields we care about.
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32; // 32-bit float samples -> we write floats above
    config.playback.channels = 2;           // stereo (0 would let miniaudio pick)
    config.sampleRate = 48000;              // 48 kHz (0 would use the device default)
    config.dataCallback = data_callback;    // hand miniaudio our callback (a function pointer)

    // Open the device. NULL context = default backend + default output device. Every
    // miniaudio call reports errors via a return code; MA_SUCCESS (0) means OK.
    if (ma_device_init(NULL, &config, &g_device) != MA_SUCCESS)
    {
        return false; // couldn't open (no device / exclusively held / ...)
    }

    // Start it: the audio thread spins up and begins calling data_callback -> sound starts.
    if (ma_device_start(&g_device) != MA_SUCCESS)
    {
        ma_device_uninit(&g_device); // start failed -> release the already-init'd device (no leak)
        return false;
    }

    return true;
}

// Stop + release the device. Called once from main() AFTER the render loop, mirroring
// audio_init. uninit stops the device (halts the callback) and frees it in one step -- no
// separate ma_device_stop needed.
void audio_shutdown()
{
    ma_device_uninit(&g_device);
}
