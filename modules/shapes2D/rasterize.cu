#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "helper_math.h"
#include "matrix_math.h"

uint32_t rgb8color(float3 color) {
    uint32_t r = color.x * 255.0f;
    uint32_t g = color.y * 255.0f;
    uint32_t b = color.z * 255.0f;
    uint32_t rgb8color = r | (g << 8) | (b << 16);
    return rgb8color;
}

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator *allocator;
uint64_t nanotime_start;

constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

// https://coolors.co/palette/8cb369-f4e285-f4a259-5b8e7d-bc4b51
float3 GRASS_COLOR = float3{140.0 / 255, 179.0 / 255, 105.0 / 255};
float3 HOUSE_COLOR = float3{91.0 / 255, 142.0 / 255, 125.0 / 255};
float3 FACTORY_COLOR = float3{188.0 / 255, 75.0 / 255, 81.0 / 255};
float3 ROAD_COLOR = float3{244.0 / 255, 226.0 / 255, 133.0 / 255};
float3 UNKOWN_COLOR = float3{1.0, 0.0, 1.0};

float3 colorFromId(uint32_t id) {
    switch (id) {
    case GRASS:
        return GRASS_COLOR;
    case ROAD:
        return ROAD_COLOR;
    case HOUSE:
        return HOUSE_COLOR;
    case FACTORY:
        return FACTORY_COLOR;
    default:
        return UNKOWN_COLOR;
    }
}

// rasterizes the grid
// - Each thread computes the color of a pixel.
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeGrid(Grid2D *grid2D, uint64_t *framebuffer) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    for (int offset = 0; offset < uniforms.width * uniforms.height; offset += grid.num_threads()) {
        int pixelId = offset + grid.thread_rank();
        if (pixelId >= uniforms.width * uniforms.height) {
            continue;
        }

        int pixelX = pixelId % int(uniforms.width);
        int pixelY = pixelId / int(uniforms.width);

        float2 pFrag = make_float2(pixelX, pixelY);

        float3 pos_W =
            unproject(pFrag, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
        int sh_cellIndex = grid2D->cellAtPosition(float2{pos_W.x, pos_W.y});
        if (sh_cellIndex == -1) {
            continue;
        }

        Cell cell = grid2D->getCell(sh_cellIndex);
        float2 diff = float2{pos_W.x - cell.center.x, pos_W.y - cell.center.y};

        if (abs(diff.x) > CELL_RADIUS || abs(diff.y) > CELL_RADIUS) {
            continue;
        }

        float3 color;
        if (uniforms.renderMode == RENDERMODE_DEFAULT) {
            color = colorFromId(grid2D->getTileId(sh_cellIndex));
        } else if (uniforms.renderMode == RENDERMODE_NETWORK) {
            TileId tileId = grid2D->getTileId(sh_cellIndex);
            if (tileId == GRASS || tileId == UNKNOWN) {
                color = {0.0f, 0.0f, 0.0f};
            } else {
                int colorId;
                if (tileId == HOUSE) {
                    colorId = *(grid2D->houseTileData(sh_cellIndex));
                } else if (tileId == ROAD) {
                    colorId = grid2D->roadNetworkRepr(sh_cellIndex);
                } else {
                    colorId = sh_cellIndex;
                }

                float r = (float)(colorId % 3) / 3.0;
                float g = (float)(colorId % 11) / 11.0;
                float b = (float)(colorId % 37) / 37.0;
                color = float3{r, g, b};

                if (colorId == -1) {
                    color = float3{1.0f, 0.0f, 1.0f};
                }
            }
        }

        float3 pixelColor = color;

        float depth = 0.0f;
        uint64_t udepth = *((uint32_t *)&depth);
        uint64_t pixel = (udepth << 32ull) | rgb8color(pixelColor);

        atomicMin(&framebuffer[pixelId], pixel);
    }
}

extern "C" __global__ void kernel(const Uniforms _uniforms, unsigned int *buffer,
                                  cudaSurfaceObject_t gl_colorbuffer, uint32_t numRows,
                                  uint32_t numCols, char *cells) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    // allocate framebuffer memory
    int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
    uint64_t *framebuffer = allocator->alloc<uint64_t *>(framebufferSize);

    // clear framebuffer
    processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex) {
        framebuffer[pixelIndex] = (uint64_t(Infinity) << 32ull) | uint64_t(BACKGROUND_COLOR);
    });

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        rasterizeGrid(grid2D, framebuffer);
    }

    grid.sync();

    uint32_t &maxNanos = *allocator->alloc<uint32_t *>(4);

    // transfer framebuffer to opengl texture
    processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex) {
        int x = pixelIndex % int(uniforms.width);
        int y = pixelIndex / int(uniforms.width);

        uint64_t encoded = framebuffer[pixelIndex];
        uint32_t color = encoded & 0xffffffffull;

        surf2Dwrite(color, gl_colorbuffer, x * 4, y);
    });
}
