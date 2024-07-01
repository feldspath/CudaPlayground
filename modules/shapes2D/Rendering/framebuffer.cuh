#pragma once

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
};
