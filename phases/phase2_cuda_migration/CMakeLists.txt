# Phase 2 - CUDA + OpenGL.
# main.cpp / renderer.cpp are compiled by the host compiler (MSVC);
# particle_system.cu is compiled by nvcc. CMake links them together.
add_executable(phase2
    src/main.cpp
    src/renderer.cpp
    src/particle_system.cu
)

target_link_libraries(phase2 PRIVATE glfw glad_gl_core)

target_include_directories(phase2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
