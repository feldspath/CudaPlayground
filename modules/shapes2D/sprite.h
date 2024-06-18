#pragma once
#include "framebuffer.h"

struct Sprite {
    // in pixels
    uint32_t x, y, width, height;
    int2 textureSize;
    uint32_t *data;

    void draw(Framebuffer framebuffer, float drawx, float drawy, float scale) {
        processRange(int(ceil(width * scale) * ceil(height * scale)), [&](int index) {
            int l_x = index % int(ceil(width * scale));
            int l_y = index / int(ceil(width * scale));

            float u = float(l_x) / ceil(width * scale);
            float v = 1.0f - float(l_y) / ceil(height * scale);

            int sx = x + width * u;
            int sy = y + height * v;
            int sourceTexel = sx + sy * textureSize.x;

            uint32_t color = data[sourceTexel];
            uint8_t *rgba = (uint8_t *)&color;

            int t_x = l_x + drawx;
            int t_y = l_y + drawy;
            int targetPixelIndex = t_x + t_y * framebuffer.width;

            // blend with current framebuffer value
            uint64_t current = framebuffer.data[targetPixelIndex];
            uint32_t currentColor = current & 0xffffffff;
            uint8_t *currentRGBA = (uint8_t *)&currentColor;

            float w = float(rgba[3]) / 255.0f;
            rgba[0] = (1.0f - w) * float(currentRGBA[0]) + w * rgba[0];
            rgba[1] = (1.0f - w) * float(currentRGBA[1]) + w * rgba[1];
            rgba[2] = (1.0f - w) * float(currentRGBA[2]) + w * rgba[2];

            framebuffer.data[targetPixelIndex] = color;
        });
    }
};

struct SpriteSheet {
    uint32_t *data;

    Sprite moneyDisplay;

    SpriteSheet(uint32_t *data) : data(data) {
        moneyDisplay.x = 0;
        moneyDisplay.y = 0;
        moneyDisplay.width = 128;
        moneyDisplay.height = 32;
        moneyDisplay.data = data;
        moneyDisplay.textureSize = int2{512, 512};
    }
};