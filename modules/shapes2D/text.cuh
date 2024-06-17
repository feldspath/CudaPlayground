#pragma once

#include "./../common/utils.cuh"
#include "framebuffer.h"

class TextRenderer {
    uint32_t *font_image;

public:
    TextRenderer(uint32_t *font) : font_image(font) {}

    void drawText(const char *text, float x, float y, float fontsize, Framebuffer framebuffer) {

        auto grid = cg::this_grid();

        uint32_t *image = font_image;

        float tilesize = 16;
        int NUM_CHARS = 95;

        int numchars = strlen(text);

        // one char after the other, utilizing 10k threads for each char haha
        for (int i = 0; i < numchars; i++) {

            int charcode = text[i];
            int tilepx = (charcode - 32) * tilesize;

            processRange(ceil(fontsize) * ceil(fontsize), [&](int index) {
                int l_x = index % int(ceil(fontsize));
                int l_y = index / int(ceil(fontsize));

                float u = float(l_x) / fontsize;
                float v = 1.0f - float(l_y) / fontsize;

                int sx = tilepx + tilesize * u;
                int sy = tilesize * v;
                int sourceTexel = sx + sy * NUM_CHARS * tilesize;

                uint32_t color = image[sourceTexel];
                uint8_t *rgba = (uint8_t *)&color;

                int t_x = l_x + x + i * fontsize;
                int t_y = l_y + y;
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

            grid.sync();
        }
    }
};
