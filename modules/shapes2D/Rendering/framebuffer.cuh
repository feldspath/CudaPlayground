#pragma once

#include "./../common/utils.cuh"

struct Framebuffer {
    unsigned int width;
    unsigned int height;
    uint64_t *data;

    Framebuffer(unsigned int width, unsigned int height, uint64_t *data)
        : width(width), height(height), data(data) {}

    void clear(uint64_t color) {
        processRange(0, width * height, [&](int pixelIndex) {
            data[pixelIndex] = (uint64_t(Infinity) << 32ull) | color;
        });
    }

    void blend(int pixelIndex, uint32_t rgba) {
        uint64_t currentValue = data[pixelIndex];
        uint32_t currentColor = currentValue & 0xffffffff;
        uint8_t *pixelRGBA = (uint8_t *)&currentColor;

        uint8_t *blendedRGBA = (uint8_t *)&rgba;

        float w = float(blendedRGBA[3]) / 255.0f;

        pixelRGBA[0] = (1.0f - w) * float(pixelRGBA[0]) + w * blendedRGBA[0];
        pixelRGBA[1] = (1.0f - w) * float(pixelRGBA[1]) + w * blendedRGBA[1];
        pixelRGBA[2] = (1.0f - w) * float(pixelRGBA[2]) + w * blendedRGBA[2];

        data[pixelIndex] = currentColor;
    }
};
