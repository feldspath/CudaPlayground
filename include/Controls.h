#pragma once

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

struct Controls {
public:
    virtual void update() {}
    virtual void onMouseScroll(double xoffset, double yoffset) {}
    virtual void onMouseMove(double xpos, double ypos) {}
    virtual void onMouseButton(int button, int action, int mods) {}
    virtual glm::dmat4 worldMatrix() const = 0;
};