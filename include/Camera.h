#pragma once

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

struct Camera {
    virtual void setSize(int width, int height) = 0;
    virtual void update(){};
    virtual glm::dmat4 viewMatrix() = 0;
    virtual glm::dmat4 projMatrix() = 0;

    virtual void setWorld(glm::dmat4 world) = 0;
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

    void setWorld(glm::dmat4 world) override {
        this->world = world;
        position = world * glm::dvec4(0.0, 0.0, 0.0, 1.0);
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

    void setWorld(glm::dmat4 world) override {
        this->world = world;
        position = world * glm::dvec4(0.0, 0.0, 0.0, 1.0);
    }

    glm::dmat4 viewMatrix() override { return view; }
    glm::dmat4 projMatrix() override { return proj; }
};