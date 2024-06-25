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

            int t_x = l_x + drawx;
            int t_y = l_y + drawy;

            if (t_x < 0 || t_x >= framebuffer.width || t_y < 0 || t_y >= framebuffer.height) {
                return;
            }

            float u = float(l_x) / ceil(width * scale);
            float v = 1.0f - float(l_y) / ceil(height * scale);

            uint32_t color = sample(u, v);
            uint8_t *rgba = (uint8_t *)&color;

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

    uint32_t sample(float u, float v) {
        int sx = x + width * u;
        int sy = y + height * v;
        int sourceTexel = sx + sy * textureSize.x;

        return data[sourceTexel];
    }

    float3 sampleFloat(float u, float v) { return float3color(sample(u, v)); }
};

struct SpriteSheet {
    uint32_t *data;

    Sprite moneyDisplay;
    Sprite populationDisplay;
    Sprite timeDisplay;
    Sprite grass;
    Sprite house;

    SpriteSheet(uint32_t *data) : data(data) {
        moneyDisplay.x = 0;
        moneyDisplay.y = 0;
        moneyDisplay.width = 128;
        moneyDisplay.height = 32;
        moneyDisplay.data = data;
        moneyDisplay.textureSize = int2{512, 512};

        populationDisplay.x = 0;
        populationDisplay.y = 32;
        populationDisplay.width = 128;
        populationDisplay.height = 32;
        populationDisplay.data = data;
        populationDisplay.textureSize = int2{512, 512};

        timeDisplay.x = 0;
        timeDisplay.y = 64;
        timeDisplay.width = 128;
        timeDisplay.height = 32;
        timeDisplay.data = data;
        timeDisplay.textureSize = int2{512, 512};

        grass.x = 128;
        grass.y = 0;
        grass.width = 8;
        grass.height = 8;
        grass.data = data;
        grass.textureSize = int2{512, 512};

        house.x = 128;
        house.y = 8;
        house.width = 8;
        house.height = 8;
        house.data = data;
        house.textureSize = int2{512, 512};
    }
};