
#include <filesystem>

#include "GLRenderer.h"
#include "Runtime.h"

namespace fs = std::filesystem;

static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                   GLsizei length, const GLchar *message, const void *userParam) {

    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION || severity == GL_DEBUG_SEVERITY_LOW ||
        severity == GL_DEBUG_SEVERITY_MEDIUM) {
        return;
    }

    std::cout << "OPENGL DEBUG CALLBACK: " << message << '\n';
}

void error_callback(int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
}

void GLRenderer::onKey(GLFWwindow *window, int key, int scancode, int action, int mods) {

    std::cout << "key: " << key << ", scancode: " << scancode << ", action: " << action
              << ", mods: " << mods << '\n';

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    Runtime::keyStates[key] = action;

    std::cout << key << '\n';
}

void GLRenderer::onCursorMove(GLFWwindow *window, double xpos, double ypos) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    Runtime::mousePosition = {xpos, ypos};

    controls->onMouseMove(xpos, ypos);
}

void GLRenderer::onScroll(GLFWwindow *window, double xoffset, double yoffset) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    controls->onMouseScroll(xoffset, yoffset);
}

void GLRenderer::onMouseButton(GLFWwindow *window, int button, int action, int mods) {

    // cout << "start button: " << button << ", action: " << action << ", mods: " << mods << endl;

    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    // cout << "end button: " << button << ", action: " << action << ", mods: " << mods << endl;

    if (action == 1) {
        Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
    } else if (action == 0) {
        uint32_t mask = ~(1 << button);
        Runtime::mouseButtons = Runtime::mouseButtons & mask;
    }

    controls->onMouseButton(button, action, mods);
}

GLRenderer::GLRenderer(std::shared_ptr<Camera> camera, std::shared_ptr<Controls> controls)
    : camera(camera), controls(controls) {

    init();

    view.framebuffer = this->createFramebuffer(128, 128);
}

void GLRenderer::init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        // Initialization failed
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, true);

    int numMonitors;
    GLFWmonitor **monitors = glfwGetMonitors(&numMonitors);

    std::cout << "<create windows>\n";
    if (numMonitors > 1 && false) {
        const GLFWvidmode *modeLeft = glfwGetVideoMode(monitors[0]);
        const GLFWvidmode *modeRight = glfwGetVideoMode(monitors[1]);

        window = glfwCreateWindow(modeRight->width, modeRight->height - 300, "Simple example",
                                  nullptr, nullptr);

        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        int xpos;
        int ypos;
        glfwGetMonitorPos(monitors[1], &xpos, &ypos);

        glfwSetWindowPos(window, xpos, ypos);
    } else {
        const GLFWvidmode *mode = glfwGetVideoMode(monitors[0]);

        window = glfwCreateWindow(mode->width - 200, mode->height - 200, "Simple example", nullptr,
                                  nullptr);

        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetWindowPos(window, 100, 100);
    }

    glfwSetWindowUserPointer(window, this);

    std::cout << "<set input callbacks>\n";
    glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        GLRenderer &renderer = *(GLRenderer *)glfwGetWindowUserPointer(window);
        renderer.onKey(window, key, scancode, action, mods);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos) {
        GLRenderer &renderer = *(GLRenderer *)glfwGetWindowUserPointer(window);
        renderer.onCursorMove(window, xpos, ypos);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
        GLRenderer &renderer = *(GLRenderer *)glfwGetWindowUserPointer(window);
        renderer.onMouseButton(window, button, action, mods);
    });
    glfwSetScrollCallback(window, [](GLFWwindow *window, double xoffset, double yoffset) {
        GLRenderer &renderer = *(GLRenderer *)glfwGetWindowUserPointer(window);
        renderer.onScroll(window, xoffset, yoffset);
    });

    static GLRenderer *ref = this;
    glfwSetDropCallback(window, [](GLFWwindow *, int count, const char **paths) {
        std::vector<std::string> files;
        for (int i = 0; i < count; i++) {
            std::string file = paths[i];
            files.push_back(file);
        }

        for (auto &listener : ref->fileDropListeners) {
            listener(files);
        }
    });

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
    }

    std::cout << "<glewInit done> "
              << "(" << now() << ")\n";

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
    glDebugMessageCallback(debugCallback, NULL);

    { // SETUP IMGUI
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 450");
        ImGui::StyleColorsDark();
    }
}

std::shared_ptr<Texture> GLRenderer::createTexture(int width, int height, GLuint colorType) {

    auto texture = Texture::create(width, height, colorType, this);

    return texture;
}

std::shared_ptr<Framebuffer> GLRenderer::createFramebuffer(int width, int height) {

    auto framebuffer = Framebuffer::create(this);

    GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "framebuffer incomplete\n";
    }

    framebuffer->setSize(width, height);

    return framebuffer;
}

void GLRenderer::loop(std::function<void(void)> update, std::function<void(void)> render) {

    int fpsCounter = 0;
    double start = now();
    double tPrevious = start;
    double tPreviousFPSMeasure = start;

    std::vector<float> frameTimes(1000, 0);

    while (!glfwWindowShouldClose(window)) {

        // TIMING
        double timeSinceLastFrame;
        {
            double tCurrent = now();
            timeSinceLastFrame = tCurrent - tPrevious;
            tPrevious = tCurrent;

            double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

            if (timeSinceLastFPSMeasure >= 1.0) {
                this->fps = double(fpsCounter) / timeSinceLastFPSMeasure;

                tPreviousFPSMeasure = tCurrent;
                fpsCounter = 0;
            }
            frameTimes[frameCount % frameTimes.size()] = timeSinceLastFrame;
        }

        // WINDOW
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        camera->setSize(width, height);
        this->width = width;
        this->height = height;

        EventQueue::instance->process();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, this->width, this->height);

        glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
        glViewport(0, 0, this->width, this->height);

        {
            controls->update();
            camera->setWorld(controls->worldMatrix());
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        { // UPDATE & RENDER
            camera->update();
            update();
            camera->update();

            render();
        }

        // IMGUI
        auto windowSize_perf = ImVec2(490, 260);

        { // RENDER IMGUI PERFORMANCE WINDOW

            std::stringstream ssFPS;
            ssFPS << this->fps;
            std::string strFps = ssFPS.str();

            ImGui::SetNextWindowPos(ImVec2(10, 10));
            ImGui::SetNextWindowSize(windowSize_perf);

            ImGui::Begin("Performance");
            ImGui::Text((rightPad("FPS:", 30) + strFps).c_str());

            static float history = 2.0f;
            static ScrollingBuffer sFrames;
            static ScrollingBuffer s60fps;
            static ScrollingBuffer s120fps;
            float t = now();

            sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);

            // sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);
            s60fps.AddPoint(t, 1000.0f / 60.0f);
            s120fps.AddPoint(t, 1000.0f / 120.0f);
            static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
            ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
            ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

            if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))) {

                auto x = &sFrames.Data[0].x;
                auto y = &sFrames.Data[0].y;
                ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity,
                                   sFrames.Offset, 2 * sizeof(float));

                ImPlot::PlotLine("16.6ms (60 FPS)", &s60fps.Data[0].x, &s60fps.Data[0].y,
                                 s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));
                ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y,
                                 s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

                ImPlot::EndPlot();
            }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        auto source = view.framebuffer;
        glBlitNamedFramebuffer(source->handle, 0, 0, 0, source->width, source->height, 0, 0,
                               0 + source->width, 0 + source->height, GL_COLOR_BUFFER_BIT,
                               GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, this->width, this->height);

        glfwSwapBuffers(window);
        glfwPollEvents();

        fpsCounter++;
        frameCount++;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

std::shared_ptr<Framebuffer> Framebuffer::create(GLRenderer *renderer) {

    auto fbo = std::make_shared<Framebuffer>();
    fbo->renderer = renderer;

    glCreateFramebuffers(1, &fbo->handle);

    { // COLOR ATTACHMENT 0

        auto texture = renderer->createTexture(fbo->width, fbo->height, GL_RGBA8);
        fbo->colorAttachments.push_back(texture);

        glNamedFramebufferTexture(fbo->handle, GL_COLOR_ATTACHMENT0, texture->handle, 0);
    }

    { // DEPTH ATTACHMENT

        auto texture = renderer->createTexture(fbo->width, fbo->height, GL_DEPTH_COMPONENT32F);
        fbo->depth = texture;

        glNamedFramebufferTexture(fbo->handle, GL_DEPTH_ATTACHMENT, texture->handle, 0);
    }

    fbo->setSize(128, 128);

    return fbo;
}

std::shared_ptr<Texture> Texture::create(int width, int height, GLuint colorType,
                                         GLRenderer *renderer) {

    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    auto texture = std::make_shared<Texture>();
    texture->renderer = renderer;
    texture->handle = handle;
    texture->colorType = colorType;

    texture->setSize(width, height);

    return texture;
}

void Texture::setSize(int width, int height) {

    bool needsResize = this->width != width || this->height != height;

    if (needsResize) {

        glDeleteTextures(1, &this->handle);
        glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

        glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTextureStorage2D(this->handle, 1, this->colorType, width, height);

        this->width = width;
        this->height = height;
    }
}