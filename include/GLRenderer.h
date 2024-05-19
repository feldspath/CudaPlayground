
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "implot_internal.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Camera.h"
#include "Controls.h"
#include "unsuck.hpp"

struct GLRenderer;

// ScrollingBuffer from ImPlot implot_demo.cpp.
// MIT License
// url: https://github.com/epezent/implot
struct ScrollingBuffer {
    int MaxSize;
    int Offset;
    ImVector<ImVec2> Data;
    ScrollingBuffer() {
        MaxSize = 2000;
        Offset = 0;
        Data.reserve(MaxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize)
            Data.push_back(ImVec2(x, y));
        else {
            Data[Offset] = ImVec2(x, y);
            Offset = (Offset + 1) % MaxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            Offset = 0;
        }
    }
};

struct GLBuffer {

    GLuint handle = -1;
    int64_t size = 0;
};

struct Texture {

    GLRenderer *renderer = nullptr;
    GLuint handle = -1;
    GLuint colorType = -1;
    int width = 0;
    int height = 0;

    static std::shared_ptr<Texture> create(int width, int height, GLuint colorType,
                                           GLRenderer *renderer);

    void setSize(int width, int height);
};

struct Framebuffer {

    std::vector<std::shared_ptr<Texture>> colorAttachments;
    std::shared_ptr<Texture> depth;
    GLuint handle = -1;
    GLRenderer *renderer = nullptr;

    int width = 0;
    int height = 0;

    Framebuffer() {}

    static std::shared_ptr<Framebuffer> create(GLRenderer *renderer);

    void setSize(int width, int height) {

        bool needsResize = this->width != width || this->height != height;

        if (needsResize) {

            // COLOR
            for (int i = 0; i < this->colorAttachments.size(); i++) {
                auto &attachment = this->colorAttachments[i];
                attachment->setSize(width, height);
                glNamedFramebufferTexture(this->handle, GL_COLOR_ATTACHMENT0 + i,
                                          attachment->handle, 0);
            }

            { // DEPTH
                this->depth->setSize(width, height);
                glNamedFramebufferTexture(this->handle, GL_DEPTH_ATTACHMENT, this->depth->handle,
                                          0);
            }

            this->width = width;
            this->height = height;
        }
    }
};

struct View {
    glm::dmat4 view;
    glm::dmat4 proj;
    std::shared_ptr<Framebuffer> framebuffer = nullptr;
};

struct GLRenderer {

    GLFWwindow *window = nullptr;
    double fps = 0.0;
    int64_t frameCount = 0;

    std::shared_ptr<Camera> camera;
    std::shared_ptr<Controls> controls;

    View view;

    struct WindowData {};

    std::vector<std::function<void(std::vector<std::string>)>> fileDropListeners;

    int width = 0;
    int height = 0;
    std::string selectedMethod = "";

    GLRenderer(std::shared_ptr<Camera> camera, std::shared_ptr<Controls> controls);

    void init();

    std::shared_ptr<Texture> createTexture(int width, int height, GLuint colorType);

    std::shared_ptr<Framebuffer> createFramebuffer(int width, int height);

    inline GLBuffer createBuffer(int64_t size) {
        GLuint handle;
        glCreateBuffers(1, &handle);
        glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);

        GLBuffer buffer;
        buffer.handle = handle;
        buffer.size = size;

        return buffer;
    }

    inline GLBuffer createSparseBuffer(int64_t size) {
        GLuint handle;
        glCreateBuffers(1, &handle);
        glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_SPARSE_STORAGE_BIT_ARB);

        GLBuffer buffer;
        buffer.handle = handle;
        buffer.size = size;

        return buffer;
    }

    inline GLBuffer createUniformBuffer(int64_t size) {
        GLuint handle;
        glCreateBuffers(1, &handle);
        glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT);

        GLBuffer buffer;
        buffer.handle = handle;
        buffer.size = size;

        return buffer;
    }

    inline std::shared_ptr<Buffer> readBuffer(GLBuffer glBuffer, uint32_t offset, uint32_t size) {

        auto target = std::make_shared<Buffer>(size);

        glGetNamedBufferSubData(glBuffer.handle, offset, size, target->data);

        return target;
    }

    void loop(std::function<void(void)> update, std::function<void(void)> render);

    void onFileDrop(std::function<void(std::vector<std::string>)> callback) {
        fileDropListeners.push_back(callback);
    }

    void onMouseButton(GLFWwindow *window, int button, int action, int mods);
    void onCursorMove(GLFWwindow *window, double xpos, double ypos);
    void onScroll(GLFWwindow *window, double xoffset, double yoffset);
    void onKey(GLFWwindow *window, int key, int scancode, int action, int mods);
};