// Phase 4 -- Audio track. Procedurally-synthesized sound for the sim (see AUDIO.md).
// T1 (event SFX): the device is SILENT by default and fires a one-shot "ding" only when the
// sim triggers it (on every preset switch). The ding is pre-rendered into a buffer once at
// init; "playing" it = walking a read cursor across that buffer, mixed into the output.
//
// miniaudio is a single-header library: this ONE .cpp defines MINIAUDIO_IMPLEMENTATION so
// the whole implementation is compiled here (every OTHER file must include miniaudio.h
// WITHOUT the macro, or the symbols get duplicated at link time).
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <math.h>  // sinf() -- synthesize the sine waveform
#include <vector>  // std::vector<float> -- the pre-rendered chime buffer
#include <atomic>  // std::atomic<bool> -- lock-free main <-> audio-thread signalling
#include <cstdlib> // rand() / RAND_MAX -- white-noise samples for build_whoosh

// The playback device handle. Opened by audio_init, read by data_callback, freed by
// audio_shutdown -- all three need it, so it's file-scope. static = internal linkage
// (private to this .cpp), so it never collides with symbols in other files.
static ma_device g_device;

// ── The chime "voice": state shared between the MAIN thread (builds the sound, requests it)
// and the AUDIO thread (plays it). Whether each needs atomic depends on ONE thing: is it
// touched by a second thread? ──

// The whole "ding", pre-rendered ONCE at init on the main thread. The callback only READS it
// and never resizes it -- no concurrent writer, no allocation on the audio thread -- so a
// plain vector is safe here. Empty until build_chime fills it.
static std::vector<float> g_chimeBuf;

// The ENTIRE main->audio message: main sets it true to ask for a ding; the callback reads and
// clears it. Shared across two threads, so it MUST be atomic -- a plain bool would be a data
// race (undefined behaviour). atomic<bool> is lock-free, so reading it in the realtime
// callback is safe. Starts false = no ding pending.
static std::atomic<bool> g_chimeTrigger{false};

// Read cursor into g_chimeBuf. Touched by the audio thread ONLY, so a plain int is enough --
// single owner means no race, no atomic needed. -1 = idle; 0..size-1 = currently playing.
static int g_chimePos = -1;

// ── The whoosh "voice": like the chime, but with NUM_WHOOSH pre-rendered VARIANTS so no two
// launches sound identical (a single buffer replayed = an obvious mechanical repeat the ear
// catches). Fired on fireworks shell launches (the GPU tallies launches, the CPU reads the count
// back and calls audio_play_whoosh). WHICH variant plays is chosen per launch on the main thread. ──
static const int NUM_WHOOSH = 5;                    // how many variants to pre-render

static std::vector<float> g_whooshBufs[NUM_WHOOSH]; // the N pre-rendered whooshes (each a noise burst)
static std::atomic<bool> g_whooshTrigger{false};    // main -> audio "play it" flag
// Which variant to play NEXT: main picks it (audio_play_whoosh) BEFORE raising the trigger; the
// audio thread reads it when it consumes the trigger. Cross-thread -> must be atomic.
static std::atomic<int> g_whooshWhich{0};
static int g_whooshPos = -1;                         // audio-thread read cursor; -1 = idle
// Which variant is CURRENTLY playing: latched from g_whooshWhich the instant the trigger fires,
// then used for the whole playback (so a mid-play re-pick can't swap the buffer under us).
// Audio-thread-only -> plain int, no atomic.
static int g_whooshCur = 0;

// ── Master controls: WRITTEN by the main thread (the menu), READ by the audio thread every
// block. Cross-thread, so both are atomic (same rule as g_chimeTrigger); lock-free on desktop,
// so reading them in the realtime callback is safe. muted wins over volume. ──

// Master volume: a linear gain the callback multiplies onto every sample. 0 = silent, 1 = full.
// Should start at 0.7 (70%) so audio isn't silent before the menu ever sets it.
static std::atomic<float> g_volume{0.7f};

// Master mute: while true the callback forces the gain to 0 (voices keep advancing, just silent).
static std::atomic<bool> g_muted{false};


// Render ONE whoosh variant into `buf`. All the per-variant knobs (length, sweep range, decay,
// level) come in as params so build_whoosh can jitter them. Combines three ingredients:
// low-passed noise (the body) + a cutoff that SWEEPS high->low (movement) + a low "thump" (the
// mortar launch punch). Runs at init on the main thread (rand + allocation are fine here).
static void build_one_whoosh(std::vector<float> &buf, float sampleRate, float seconds, float fcStart, float fcEnd, float tau, float amp)
{
    int len = (int)(seconds * sampleRate);
    if (len < 1) len = 1;
    buf.assign(len, 0.0f);

    const float PI2 = 6.28318530718f;
    float y = 0.0f; // low-pass filter state (carries across samples, as before)

    for (int i = 0; i < len; ++i)
    {
        float t = (float)i / sampleRate;

        // SWEEP: the cutoff glides fcStart -> fcEnd across the whoosh, so `a` is recomputed each
        // sample (no longer constant). frac = play progress 0..1; fcNow = linear interp between them.
        float frac = t / seconds;
        float fcNow = fcStart + (fcEnd - fcStart) * frac;
        float a = 1.0f - expf(-PI2 * fcNow / sampleRate);

        // body: white noise -> one low-pass step (y is now the low-passed sample)
        float noise = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        y += a * (noise - y);
        float attack = (t < 0.01f) ? (t / 0.01f) : 1.0f; // 10 ms ramp-in -> no onset click
        float env = attack * expf(-t / tau);             // then exponential decay (tau = param)

        // THUMP: a short low (130 Hz) sine with its OWN fast decay (tau = 0.06 s) = the mortar
        // punch at launch. 130 Hz (not sub-100) so small speakers can actually reproduce it.
        // Fixed pitch (no sweep) so a plain sinf(2*pi*f*t) is safe here -- no phase accumulator
        // needed (that trap only bites a SWEPT oscillator).
        float thump = 0.35f * expf(-t / 0.06f) * sinf(PI2 * 130.0f * t);

        buf[i] = amp * env * y + thump; // mix: low-passed-noise whoosh + the thump
    }
}

// Pre-render ALL NUM_WHOOSH whoosh variants. Each gets jittered params via the pattern
// base + range * (rand()/RAND_MAX) = a random value in [base, base+range], so no two variants
// (hence no two launches) sound the same; build_one_whoosh does the actual synthesis. Runs ONCE
// from audio_init on the main thread (rand + allocation are fine here, never in the callback).
static void build_whoosh(float sampleRate)
{
    for (int k = 0; k < NUM_WHOOSH; k++)
    {
        float seconds = 0.9f + 0.6f * (rand() / (float)RAND_MAX);       // 0.9 .. 1.5 s length (>= 3*tau, safe)
        float fcStart = 1000.0f + 1400.0f * (rand() / (float)RAND_MAX); // 1000 .. 2400 sweep HIGH end (start)
        float fcEnd = 300.0f + 600.0f * (rand() / (float)RAND_MAX);     // 300 .. 900 sweep LOW end (finish)
        float tau = 0.18f + 0.12f * (rand() / (float)RAND_MAX);         // 0.18 .. 0.30 s decay
        float amp = 0.55f;                                             // fixed level (not jittered)

        build_one_whoosh(g_whooshBufs[k], sampleRate, seconds, fcStart, fcEnd, tau, amp);
    }
}

// Pre-render one chime "ding" into g_chimeBuf. Runs ONCE from audio_init, on the MAIN thread --
// the only place allocation is allowed (the realtime callback must never malloc). Rendered at
// the device's ACTUAL sample rate so pitch and length come out right.
static void build_chime(float sampleRate)
{
    // Size the voice: a short ding (~0.4 s). length in samples = seconds * samples-per-second.
    const float seconds = 0.4f;
    int len = (int)(seconds * sampleRate);

    if (len < 1)
    {
        len = 1; // guard a bogus sample rate so the buffer is never empty
    }

    g_chimeBuf.assign(len, 0.0f); // allocate + zero (silent) before the loop fills it

    // Fill each sample = amp * envelope(t) * sine(t). The envelope shapes the VOLUME over time;
    // it's what turns a flat "beep" into a bell "ding" that strikes and then rings out.
    for (int i = 0; i < len; ++i)
    {
        float t = (float)i / sampleRate;  // this sample's time, in seconds
        float f = 880.f;                  // pitch: 880 Hz = A5 (bright, pleasant)
        float amp = 0.3f;                 // stay well under 1.0 so mixing never clips
        const float PI2 = 6.28318530718f; // 2*PI -> one full sine cycle
        const float tau = 0.12f;          // decay time constant (s): bigger = rings longer

        // Envelope = fast ATTACK * exponential DECAY:
        //   attack: ramp 0->1 over the first 5 ms so the onset doesn't click (a hard start
        //           snaps the speaker); flat 1.0 afterwards.
        //   decay:  expf(-t/tau) falls from 1 toward 0 -- the "ring out" that makes it a ding.
        float attack = (t < 0.005f) ? (t / 0.005f) : 1.0f;
        float env = attack * expf(-t / tau);

        g_chimeBuf[i] = amp * env * sinf(PI2 * f * t);
    }
}

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

    // Consume a pending trigger ONCE per block: exchange() reads the old value AND stores
    // false in a single atomic step, so a ding requested since the last block rewinds the
    // voice to sample 0 exactly once -- never double-fired, never lost.
    if (g_chimeTrigger.exchange(false))
    {
        g_chimePos = 0; // (re)start the ding from its first sample
    }

    if (g_whooshTrigger.exchange(false))
    {
        g_whooshCur = g_whooshWhich.load();
        g_whooshPos = 0;
    }

    // Snapshot the master gain ONCE per block (cheap, and stable across the whole block). muted
    // overrides volume: muted -> 0, else the current volume. .load() = atomic read.
    float mainVolume = g_muted.load() ? 0.0f : g_volume.load();

    // Outer loop: one iteration per FRAME. Inner loop: write that frame's value into every
    // channel of the interleaved buffer.
    for (ma_uint32 frame = 0; frame < frameCount; ++frame)
    {
        // Each frame starts at silence, then every voice ADDS itself in (`+=`, never `=`) so the
        // voices MIX. Two voices today (chime, whoosh); T2/T3 just add more of the same blocks.
        float sample = 0.0f;

        // Advance the chime voice while its cursor points at a real sample. The `< size()`
        // half also keeps an empty/unfinished buffer safe (no out-of-bounds read).
        if (g_chimePos >= 0 && g_chimePos < (int)g_chimeBuf.size())
        {
            sample += g_chimeBuf[g_chimePos]; // read the sample the cursor points at...
            g_chimePos++;                     // ...then step the cursor forward one
        }
        else
        {
            g_chimePos = -1; // ran off the end (or nothing playing) -> idle
        }

        if (g_whooshPos >= 0 && g_whooshPos < (int)g_whooshBufs[g_whooshCur].size())
        {
            sample += g_whooshBufs[g_whooshCur][g_whooshPos];
            g_whooshPos++;
        }
        else
        {
            g_whooshPos = -1;
        }

        sample *= mainVolume; // apply master volume / mute to the mixed voices, before output
        // Interleaved layout is [L0,R0, L1,R1, ...], so frame `frame` channel `ch` lives at
        // index frame*channels + ch. Writing the SAME value to every channel = a centered
        // mono tone.
        for (ma_uint32 ch = 0; ch < channels; ++ch)
        {
            out[frame * channels + ch] = sample;
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

    build_chime((float)g_device.sampleRate);
    build_whoosh((float)g_device.sampleRate);

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

// T1 public API: ask for a one-shot ding. Safe to call from the MAIN thread -- it only sets
// an atomic flag; the audio thread picks it up next block (via exchange() in data_callback)
// and rewinds the voice. Never touches g_chimePos or the buffer directly, so it cannot race.
void audio_play_chime()
{
    g_chimeTrigger.store(true);
}

// T1 master controls, called from the menu (main thread). Each just writes an atomic, so they're
// cheap and safe to call every frame; the callback picks up the change on its next block.

// Menu mute toggle: the gain becomes 0 while muted (voices keep advancing, just silent).
void audio_set_muted(bool muted)
{
    g_muted.store(muted);
}

// Menu volume slider: clamp to a sane [0,1] gain, then publish it to the audio thread.
void audio_set_volume(float volume)
{
    if (volume < 0.0f)
    {
        volume = 0.0f;
    }
    else if (volume > 1.0f)
    {
        volume = 1.0f;
    }
    g_volume.store(volume);
}

// T1 public API: ask for a one-shot whoosh. Picks a RANDOM variant, then raises the trigger --
// both just atomic stores from the main thread, so it's safe and can't race. Order matters: the
// variant is stored BEFORE the trigger, so when the audio thread sees the trigger it also sees
// the fresh index (seq_cst atomics guarantee that ordering).
void audio_play_whoosh()
{
    g_whooshWhich.store(rand() % NUM_WHOOSH); // choose which of the NUM_WHOOSH variants to play
    g_whooshTrigger.store(true);              // then signal the callback to (re)start it
}