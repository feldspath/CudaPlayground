
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

#include "Controls.h"
#include "unsuck.hpp"

using namespace std;
using glm::dmat4;
using glm::dvec3;
using glm::dvec4;

#define BIND_MEMBER_FN(fn) std::bind(&fn, this, std::placeholders::_1)

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

    static shared_ptr<Texture> create(int width, int height, GLuint colorType,
                                      GLRenderer *renderer);

    void setSize(int width, int height);
};

struct Framebuffer {

    vector<shared_ptr<Texture>> colorAttachments;
    shared_ptr<Texture> depth;
    GLuint handle = -1;
    GLRenderer *renderer = nullptr;

    int width = 0;
    int height = 0;

    Framebuffer() {}

    static shared_ptr<Framebuffer> create(GLRenderer *renderer);

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
    dmat4 view;
    dmat4 proj;
    shared_ptr<Framebuffer> framebuffer = nullptr;
};

struct Camera {
    virtual void setSize(int width, int height) = 0;
    virtual void update(){};
    virtual glm::dmat4 viewMatrix() = 0;
    virtual glm::dmat4 projMatrix() = 0;

    virtual void setWorld(dmat4 world) = 0;
};

struct Camera2D : public Camera {

    glm::dvec3 position;
    glm::dmat4 proj, view, world;
    double rotation;

    double minX = 0.0;
    double maxX = 100.0;
    double minY = 0.0;
    double maxY = 100.0;
    int width = 128;
    int height = 128;
    double aspect = 1.0;

    Camera2D() {}

    void setSize(int width, int height) override {
        this->width = width;
        this->height = height;
        this->aspect = double(width) / double(height);

        maxY = (maxX - minX) / aspect + minY;
    }

    void update() override {
        view = glm::inverse(world);

        proj = glm::ortho(minX, maxX, minY, maxY);
    }

    void setWorld(dmat4 world) override {
        this->world = world;
        position = world * dvec4(0.0, 0.0, 0.0, 1.0);
    }

    glm::dmat4 viewMatrix() override { return view; }
    glm::dmat4 projMatrix() override { return proj; }
};

struct Camera3D : public Camera {

    glm::dvec3 position;
    glm::dmat4 rotation;

    glm::dmat4 world;
    glm::dmat4 view;
    glm::dmat4 proj;

    double aspect = 1.0;
    double fovy = 60.0;
    double near = 0.1;
    double far = 2'000'000.0;
    int width = 128;
    int height = 128;

    Camera3D() {}

    void setSize(int width, int height) override {
        this->width = width;
        this->height = height;
        this->aspect = double(width) / double(height);
    }

    void update() override {
        view = glm::inverse(world);

        double pi = glm::pi<double>();
        proj = glm::perspective(pi * fovy / 180.0, aspect, near, far);
    }

    void setWorld(dmat4 world) override {
        this->world = world;
        position = world * dvec4(0.0, 0.0, 0.0, 1.0);
    }

    glm::dmat4 viewMatrix() override { return view; }
    glm::dmat4 projMatrix() override { return proj; }
};

struct GLRenderer {

    GLFWwindow *window = nullptr;
    double fps = 0.0;
    int64_t frameCount = 0;

    shared_ptr<Camera> camera;
    shared_ptr<Controls> controls;

    View view;

    struct WindowData {};

    vector<function<void(vector<string>)>> fileDropListeners;

    int width = 0;
    int height = 0;
    string selectedMethod = "";

    GLRenderer(shared_ptr<Camera> camera, shared_ptr<Controls> controls);

    void init();

    shared_ptr<Texture> createTexture(int width, int height, GLuint colorType);

    shared_ptr<Framebuffer> createFramebuffer(int width, int height);

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

    inline shared_ptr<Buffer> readBuffer(GLBuffer glBuffer, uint32_t offset, uint32_t size) {

        auto target = make_shared<Buffer>(size);

        glGetNamedBufferSubData(glBuffer.handle, offset, size, target->data);

        return target;
    }

    void loop(function<void(void)> update, function<void(void)> render);

    void onFileDrop(function<void(vector<string>)> callback) {
        fileDropListeners.push_back(callback);
    }

    void onMouseButton(GLFWwindow *window, int button, int action, int mods);
    void onCursorMove(GLFWwindow *window, double xpos, double ypos);
    void onScroll(GLFWwindow *window, double xoffset, double yoffset);
    void onKey(GLFWwindow *window, int key, int scancode, int action, int mods);
};