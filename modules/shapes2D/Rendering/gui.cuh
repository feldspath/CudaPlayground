#pragma once

#include "common/utils.cuh"
#include "HostDeviceInterface.h"

#include "World/map.cuh"
#include "framebuffer.cuh"
#include "sprite.cuh"
#include "text.cuh"

struct GUIBox {
    uint32_t x, y, width, height;
    Sprite sprite;
    GUIBox(uint32_t x, uint32_t y, uint32_t width, uint32_t height, Sprite sprite)
        : x(x), y(y), width(width), height(height), sprite(sprite) {}
};

class GUI {
    GUIBox moneyDisplay;
    GUIBox populationDisplay;
    GUIBox timeDisplay;

    TextRenderer &textRenderer;
    Framebuffer &framebuffer;
    SpriteSheet &sprites;

    mat4 viewProj;

public:
    GUI(Framebuffer &framebuffer, TextRenderer &textRenderer, SpriteSheet &sprites, mat4 viewProj);
    void render(Map map, Entity* entities, uint32_t numEntities);

private:
    void renderDisplay(GUIBox box, char *displayString);
    void renderInfoPanel(Map map, Entity* entities, uint32_t numEntities);
};