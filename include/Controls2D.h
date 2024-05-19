
#pragma once

#include <iostream>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Runtime.h"

#include "Controls.h"

using namespace std;

struct Controls2D : public Controls {
    double zoom;

    glm::dvec2 pos = {0.0, 0.0};
    glm::dvec2 left = {1.0, 0.0};
    glm::dvec2 up = {0.0, 1.0};

    bool isLeftDown = false;
    bool isRightDown = false;

    glm::dvec2 mousePos;

    glm::dmat4 world;

    Controls2D() {}

    glm::dvec3 getPosition() { return glm::dvec3(pos, 0.0); }

    void translate(glm::dvec2 translation) { pos += translation; }

    void onMouseButton(int button, int action, int mods) {
        // cout << "button: " << button << ", action: " << action << ", mods: " << mods << endl;

        if (button == 0 && action == 1) {
            isLeftDown = true;
        } else if (action == 0) {
            isLeftDown = false;
        }

        if (button == 1 && action == 1) {
            isRightDown = true;
        } else if (action == 0) {
            isRightDown = false;
        }
    }

    void onMouseMove(double xpos, double ypos) override {

        bool selectActive = Runtime::keyStates[342] > 0;
        if (selectActive) {
            return;
        }

        glm::dvec2 newMousePos = {xpos, ypos};
        glm::dvec2 diff = newMousePos - mousePos;

        if (isRightDown) {
            auto ux = -diff.x / 10.0;
            auto uy = diff.y / 10.0;
            translate(glm::dvec2(ux, uy));
        }

        mousePos = newMousePos;
    }

    void onMouseScroll(double xoffset, double yoffset) override {
        // cout << xoffset << ", " << yoffset << endl;

        // +1: zoom in
        // -1: zoom out

        // if (yoffset < 0.0) {
        //     radius = radius * 1.1;
        // } else {
        //     radius = radius / 1.1;
        // }

        // cout << radius << endl;
    }

    void update() override {
        auto w = glm::dmat4(1.0);
        world = glm::translate(w, getPosition());
    }

    glm::dmat4 worldMatrix() const override { return world; };
};