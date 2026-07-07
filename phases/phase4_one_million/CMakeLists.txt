# Phase 4 — one million particles.
# Source files are added as you write them. Start with main.cpp, then the
# renderer, then the CUDA particle_system. Uncomment lines as files appear.

add_executable(phase4
    src/main.cpp
    src/renderer.cpp
    src/particle_system.cu
)

target_link_libraries(phase4 PRIVATE glfw glad_gl_core imgui)

target_include_directories(phase4 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Bake the assets directory's ABSOLUTE path into the binary as a string macro, so
# runtime file loads (e.g. the ImGui font) work no matter the working directory.
target_compile_definitions(phase4 PRIVATE
    SPARKS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)
