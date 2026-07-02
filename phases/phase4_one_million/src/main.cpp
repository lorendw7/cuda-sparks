// Phase 4 -- window + GL context setup, then draw test points via the Renderer.
// glad must be included before glfw.
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio>     // fprintf on errors
#include <vector>     // std::vector for the test points
#include "renderer.h" // our hand-written rendering layer

int main()
{
    // 1) Start GLFW. Returns 0 on failure.
    if (!glfwInit())
    {
        fprintf(stderr, "glfwInit failed.\n");
        return 1;
    }

    // 2) Ask for an OpenGL 3.3 core-profile context (same as Phase 3).
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 3) Create a 1280x1280 window. nullptr window means creation failed.
    GLFWwindow *window = glfwCreateWindow(1280, 1280, "cuda-sparks | Phase 4",
                                          nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 1;
    }

    // 4) Make the window's GL context current, disable vsync (uncapped FPS).
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // 5) Load GL function pointers with glad -- only possible once a context exists.
    if (!gladLoadGL(glfwGetProcAddress))
    {
        fprintf(stderr, "gladLoadGL failed.\n");
        return 1;
    }

    // Inner scope: the Renderer must be destroyed (its ~Renderer runs glDelete*)
    // while the GL context is STILL alive. This "}" ends before glfwTerminate(),
    // so the destructor fires at the right time. (RAII + GL ordering.)
    {
        // (A) Build the renderer now that the GL context is ready.
        Renderer renderer;
        renderer.init(1000); // pre-size the VBO for up to 1000 particles (plenty for the test)

        // (B) Four hand-made test points: 5 floats each -> [x, y, r, g, b].
        std::vector<float> test = {
            -0.5f, -0.5f, 1.0f, 0.2f, 0.2f, // bottom-left, red
            0.5f, -0.5f, 0.2f, 1.0f, 0.2f,  // bottom-right, green
            0.0f, 0.5f, 0.2f, 0.4f, 1.0f,   // top, blue
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,   // center, white
        };
        int count = 4;

        // (C) Upload once -- the test data never changes, so no need to re-upload each frame.
        renderer.upload(test, count);

        // (D) Render loop: run until the user closes the window.
        while (!glfwWindowShouldClose(window))
        {
            glClearColor(0.02f, 0.02f, 0.05f, 1.0f); // dark blue background
            glClear(GL_COLOR_BUFFER_BIT);            // clear the back buffer

            renderer.draw(); // draw the 4 uploaded points

            glfwSwapBuffers(window); // present the frame
            glfwPollEvents();        // handle close/keyboard events
        }
    } // <- renderer destroyed here, GL context still alive

    // 6) Tear down the context AFTER the renderer is gone.
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
