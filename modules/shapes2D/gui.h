#pragma once

#include "./../common/utils.cuh"
#include "framebuffer.h"
#include "sprite.h"
#include "text.cuh"

struct GUIBox {
    uint32_t x, y, width, height;
    Sprite sprite;
    GUIBox(uint32_t x, uint32_t y, uint32_t width, uint32_t height, Sprite sprite)
        : x(x), y(y), width(width), height(height), sprite(sprite) {}
};

#define MAX_STRING_LENGTH 16

static void itos(unsigned int value, char *string) {
    char chars[MAX_STRING_LENGTH];
    int i;
    for (i = 0; i < 10; ++i) {
        if (i > 0 && value == 0) {
            break;
        }
        chars[i] = value % 10 + '0';
        value = value / 10;
    }

    for (int j = 0; j < i; j++) {
        string[i - 1 - j] = chars[j];
    }
    string[i] = 0;
}

class GUI {
    GUIBox moneyDisplay;
    GUIBox populationDisplay;

    TextRenderer &textRenderer;
    SpriteSheet &sprites;

public:
    GUI(uint32_t width, uint32_t height, TextRenderer &textRenderer, SpriteSheet &sprites)
        : textRenderer(textRenderer), sprites(sprites),
          moneyDisplay(width - 400, height - 100, 100, 30, sprites.moneyDisplay),
          populationDisplay(width - 400, height - 200, 100, 30, sprites.populationDisplay) {}

    void render(Framebuffer framebuffer, GameState *gameState) {
        unsigned int money = gameState->playerMoney;
        renderDisplay(moneyDisplay, money, framebuffer);

        unsigned int population = gameState->population;
        renderDisplay(populationDisplay, population, framebuffer);
    }

    void renderDisplay(GUIBox box, unsigned int value, Framebuffer framebuffer) {
        auto grid = cg::this_grid();

        box.sprite.draw(framebuffer, box.x, box.y, 3.0f);

        grid.sync();

        char displayString[MAX_STRING_LENGTH + 1];
        itos(value, displayString);

        textRenderer.drawText(displayString, box.x + 150.0, box.y + 30.0, 30, framebuffer);
    }
};