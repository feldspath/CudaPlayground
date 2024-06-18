#pragma once

#include "./../common/utils.cuh"
#include "framebuffer.h"
#include "sprite.h"
#include "text.cuh"

struct GUIBox {
    uint32_t x, y, width, height;
    GUIBox(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
        : x(x), y(y), width(width), height(height) {}
};

class GUI {
    GUIBox moneyDisplay;
    // GUIBox populationDisplay;

    TextRenderer &textRenderer;
    SpriteSheet &sprites;

public:
    GUI(uint32_t width, uint32_t height, TextRenderer &textRenderer, SpriteSheet &sprites)
        : textRenderer(textRenderer), sprites(sprites),
          moneyDisplay(width - 400, height - 100, 100, 30) {}

    void render(Framebuffer framebuffer, GameState *gamestate) {
        auto grid = cg::this_grid();

        sprites.moneyDisplay.draw(framebuffer, moneyDisplay.x, moneyDisplay.y, 3.0f);

        grid.sync();

        int money = gamestate->playerMoney;
        char moneyString[17];
        char chars[16];
        int i;
        for (i = 0; i < 10; ++i) {
            if (i > 0 && money == 0) {
                break;
            }
            chars[i] = money % 10 + '0';
            money = money / 10;
        }

        for (int j = 0; j < i; j++) {
            moneyString[i - 1 - j] = chars[j];
        }
        moneyString[i] = 0;

        textRenderer.drawText(moneyString, moneyDisplay.x + 150.0, moneyDisplay.y + 30.0, 30,
                              framebuffer);
    }
};