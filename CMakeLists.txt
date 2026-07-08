cmake_minimum_required(VERSION 3.18)

# Phase 1 is pure C++; Phase 2 onward adds CUDA. Both languages are enabled here.
project(cuda_sparks LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings (used from Phase 2 on).
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)   # auto-detect this machine's GPU
# Keep line numbers in compiled device code so Nsight (Phase 4) can map them.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")

# Make every target (ours + GLFW + glad) use the same MSVC runtime (dynamic /MD),
# otherwise the linker warns about LIBCMT conflicts.
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Put all executables in a predictable place per-phase (handled by each subdir).
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Third-party graphics dependencies (downloaded automatically on first build)
# ---------------------------------------------------------------------------
include(FetchContent)

# --- GLFW: window + input + OpenGL context ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- glad: loads modern OpenGL function pointers at runtime ---
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v2.0.6
    SOURCE_SUBDIR  cmake
)
FetchContent_MakeAvailable(glad)
# Generate a static glad library for OpenGL 3.3 Core.
glad_add_library(glad_gl_core STATIC API gl:core=3.3)

# --- Dear ImGui: immediate-mode GUI (P3 menu). Not CMake-native, so FetchContent
#     only downloads it; we compile its core + the GLFW/OpenGL3 backends ourselves. ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
)
FetchContent_MakeAvailable(imgui)   # no CMakeLists in imgui -> this just downloads it

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)
if (WIN32)
    target_link_libraries(imgui PUBLIC opengl32)   # the GL3 backend's loader uses wglGetProcAddress
endif()

# --- miniaudio: single-header audio library (playback device + callback). Like imgui it
#     ships no CMake, so FetchContent only DOWNLOADS it. Unlike imgui it has no .cpp files:
#     the whole implementation lives inside miniaudio.h and is compiled into our own
#     src/audio.cpp via `#define MINIAUDIO_IMPLEMENTATION`. So there is deliberately NO
#     add_library here -- miniaudio is just a header; phase4 includes it directly (the
#     header path ${miniaudio_SOURCE_DIR} is added to phase4's include dirs in its CMake). ---
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.21          # pin a release tag, never a branch, for reproducible builds
)
FetchContent_MakeAvailable(miniaudio)   # no CMakeLists in miniaudio -> this just downloads it

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
add_subdirectory(phases/phase1_cpu_baseline)
add_subdirectory(phases/phase2_cuda_migration)
add_subdirectory(phases/phase3_effects)
add_subdirectory(phases/phase4_one_million)