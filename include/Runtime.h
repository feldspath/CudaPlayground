
#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "glm/common.hpp"

struct Runtime {

    struct GuiItem {
        uint32_t type = 0;
        float min = 0.0;
        float max = 1.0;
        float oldValue = 0.5;
        float value = 0.5;
        std::string label = "";
    };

    inline static std::vector<int> keyStates = std::vector<int>(65536, 0);
    inline static glm::dvec2 mousePosition = {0.0, 0.0};
    inline static int mouseButtons = 0;

    inline static int modeId = -1;

    Runtime() {}

    static Runtime *getInstance() {
        static Runtime *instance = new Runtime();

        return instance;
    }
};