# Phase 1 — pure C++ + OpenGL. No CUDA yet.
add_executable(phase1
    src/main.cpp
    src/renderer.cpp
)

target_link_libraries(phase1 PRIVATE glfw glad_gl_core)

# Make sure the shared include/ (particles.h) and local src/ are both visible.
target_include_directories(phase1 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
