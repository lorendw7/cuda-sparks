#pragma once
#include "imgui.h" // ImGui::BeginChild/Text/..., ImU32, ImVec2
#include <cstdarg> // va_list / va_start / va_end -- the variadic plumbing push() forwards
#include <cstdio>  // vsnprintf -- the va_list-taking version of snprintf

// P2b: hacker-style scrolling log. A fixed-capacity RING BUFFER -- zero heap
// allocation, O(1) push, cache-friendly. Keeps the most recent kCap lines; once
// full, the write cursor wraps and overwrites the oldest line (the exact scheme
// behind the kernel's dmesg buffer and a flight recorder). Lives on the stack
// inside main(), so the whole 12KB comes and goes with one scope -- no new/delete.
struct HudLog
{
    static constexpr int kCap     = 128; // max lines kept (older ones get overwritten)
    static constexpr int kLineLen = 96;  // max chars per line (fixed -> stays off the heap)

    char  lines[kCap][kLineLen] = {}; // ring storage: kCap lines x kLineLen bytes, zero-inited
    ImU32 colors[kCap]          = {}; // per-line color (IM_COL32-packed RGBA), parallel to lines[]
    int   head  = 0;                  // index where the NEXT line will be written
    int   count = 0;                  // number of valid lines so far (saturates at kCap)

    // Append one line, printf-style: push(IM_COL32(0,255,65,255), "n=%d", n).
    // A member function -> the hidden `this` lets it read/write lines/head/colors/count.
    void push(ImU32 color, const char *fmt, ...)
    {
        va_list args;                                // cursor over the "..." (unnamed) args
        va_start(args, fmt);                         // anchor it just past the last NAMED param (fmt)
        vsnprintf(lines[head], kLineLen, fmt, args); // format into the current line; bounded + always NUL-terminated
        va_end(args);                                // close the cursor (must pair with va_start)
        colors[head] = color;                        // store this line's color at the same index
        head = (head + 1) % kCap;                    // advance the write cursor, wrapping back to 0 at kCap
        if (count < kCap)                            // grow while not yet full...
        {
            count++;                                 // ...then hold at kCap (further pushes only overwrite)
        }
    }

    // Draw the scrollable log region. Called once per frame (immediate mode rebuilds it).
    void draw()
    {
        // A scrollable child region: width 0 = fill the parent's remaining width, height 220px.
        // Borders = draw an outline; HorizontalScrollbar = long lines scroll instead of wrapping.
        ImGui::BeginChild("hudlog", ImVec2(0, 220),
                          ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar);

        for (int i = 0; i < count; i++) // walk the `count` valid lines, oldest -> newest
        {
            // Map logical row i (0 = oldest) to its physical slot. The oldest line sits at
            // (head - count); adding kCap keeps that non-negative before the wrap-around modulo.
            int idx = (head - count + i + kCap) % kCap;

            // Override the text color for THIS line only (ColorConvert... unpacks ImU32 -> ImVec4).
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImGui::ColorConvertU32ToFloat4(colors[idx]));

            ImGui::TextUnformatted(lines[idx]); // print raw -- a '%' in the text is NOT a format spec

            ImGui::PopStyleColor();             // pop the color pushed above (stack must stay balanced)
        }

        // Auto-scroll to the newest line ONLY when the user is already at the bottom, so
        // scrolling up to read history isn't yanked back down. Must run AFTER the loop:
        // GetScrollMaxY() only knows the full content height once every line is submitted.
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        {
            ImGui::SetScrollHereY(1.0f); // 1.0 = align to the bottom edge
        }

        ImGui::EndChild(); // pair with BeginChild
    }
};
