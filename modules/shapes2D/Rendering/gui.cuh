#pragma once

#include "common/utils.cuh"

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
    SpriteSheet &sprites;

    mat4 viewProj;

public:
    GUI(uint32_t width, uint32_t height, TextRenderer &textRenderer, SpriteSheet &sprites,
        mat4 viewProj);
    void render(Framebuffer framebuffer, Map *map, Entities *entities);

private:
    void renderDisplay(GUIBox box, char *displayString, Framebuffer framebuffer);
    void renderInfoPanel(Framebuffer framebuffer, Map *map, Entities *entities);
};