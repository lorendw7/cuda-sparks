cmake_minimum_required(VERSION 3.18)

# CUDA is only required from Phase 2 on. Phase 1 is pure C++.
# We enable CUDA here but make it optional so Phase 1 still configures on machines
# that are still setting up the toolkit.
project(cuda_sparks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
add_subdirectory(phases/phase1_cpu_baseline)
