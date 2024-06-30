#pragma once
#include "framebuffer.cuh"

struct Sprite {
    // in pixels
    uint32_t x, y, width, height;
    int2 textureSize;
    uint32_t *data;

    void draw(Framebuffer framebuffer, float drawx, float drawy, float scale);
    uint32_t sample(float u, float v);
    float3 sampleFloat(float u, float v) { return float3color(sample(u, v)); }
};

struct SpriteSheet {
    uint32_t *data;

    Sprite moneyDisplay;
    Sprite populationDisplay;
    Sprite timeDisplay;
    Sprite grass;
    Sprite house;

    SpriteSheet(uint32_t *data);
};