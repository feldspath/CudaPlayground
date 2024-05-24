#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "helper_math.h"
#include "matrix_math.h"
#include "network.h"

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

// rasterizes voxels
// - each block grabs a voxel
// - all threads of that block process different fragments of the voxel
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeGrid(Grid2D *grid2D, Network *network, uint64_t *framebuffer) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uint32_t &processedCells = *allocator->alloc<uint32_t *>(4);
    if (grid.thread_rank() == 0) {
        processedCells = 0;
    }
    grid.sync();

    {
        __shared__ int sh_cellIndex;

        block.sync();

        // safety mechanism: each block draws at most <loop_max> voxels
        int loop_max = 10'000;
        for (int loop_i = 0; loop_i < loop_max; loop_i++) {

            // grab the index of the next unprocessed voxel
            block.sync();
            if (block.thread_rank() == 0) {
                sh_cellIndex = atomicAdd(&processedCells, 1);
            }
            block.sync();

            if (sh_cellIndex >= grid2D->count)
                break;

            Cell cell = grid2D->getCell(sh_cellIndex);
            float3 diff = float3{CELL_RADIUS, CELL_RADIUS, 0.0f};
            float3 center = float3{cell.center.x, cell.center.y, 0.0f};
            float4 cornerLow_W = make_float4(center - diff, 1.0f);
            float4 cornerHigh_W = make_float4(center + diff, 1.0f);

            float4 cornerLow_S = uniforms.proj * uniforms.view * cornerLow_W;
            float4 cornerHigh_S = uniforms.proj * uniforms.view * cornerHigh_W;

            // clamp to screen
            float min_x = max((cornerLow_S.x * 0.5f + 0.5f) * uniforms.width, 0.0f);
            float min_y = max((cornerLow_S.y * 0.5f + 0.5f) * uniforms.height, 0.0f);
            float max_x = min((cornerHigh_S.x * 0.5f + 0.5f) * uniforms.width, uniforms.width);
            float max_y = min((cornerHigh_S.y * 0.5f + 0.5f) * uniforms.height, uniforms.height);

            int size_x = ceil(max_x) - floor(min_x);
            int size_y = ceil(max_y) - floor(min_y);
            int numFragments = size_x * size_y;

            // iterate through fragments in bounding rectangle and draw if within triangle
            int numProcessedSamples = 0;
            for (int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()) {

                // safety mechanism: don't draw more than <x> pixels per thread
                if (numProcessedSamples > 5'000)
                    break;

                numProcessedSamples++;

                int fragID = fragOffset + block.thread_rank();

                int fragX = fragID % size_x;
                int fragY = fragID / size_x;

                float2 pFrag = {floor(min_x) + float(fragX), floor(min_y) + float(fragY)};

                if (pFrag.x < min_x || pFrag.x >= max_x || pFrag.y < min_y || pFrag.y >= max_y) {
                    continue;
                }

                int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
                int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
                pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

                float3 color;
                if (uniforms.renderMode == RENDERMODE_DEFAULT) {
                    color = colorFromId(grid2D->cellIndices[sh_cellIndex]);
                } else {
                    if (grid2D->cellIndices[sh_cellIndex] == GRASS) {
                        color = float3{0.0, 0.0, 0.0};
                    } else {
                        int repr = network->cellRepr(sh_cellIndex);
                        float r = (float)(repr % 3) / 3.0;
                        float g = (float)(repr % 11) / 11.0;
                        float b = (float)(repr % 37) / 37.0;
                        color = float3{r, g, b};
                    }
                }

                float3 fragColor = color;

                float depth = 0.0f;
                uint64_t udepth = *((uint32_t *)&depth);
                uint64_t pixel = (udepth << 32ull) | rgb8color(fragColor);

                atomicMin(&framebuffer[pixelID], pixel);
            }
        }
    }
}

extern "C" __global__ void kernel(const Uniforms _uniforms, unsigned int *buffer,
                                  cudaSurfaceObject_t gl_colorbuffer, uint32_t numRows,
                                  uint32_t numCols, uint32_t *gridCells, uint32_t *networkParents) {
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
        *grid2D = Grid2D(numRows, numCols, gridCells);

        Network *network = allocator->alloc<Network *>(sizeof(Network));
        network->parents = networkParents;

        rasterizeGrid(grid2D, network, framebuffer);
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
