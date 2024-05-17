#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "helper_math.h"

float4 operator*(const mat4 &a, const float4 &b) {
    return make_float4(dot(a.rows[0], b), dot(a.rows[1], b), dot(a.rows[2], b), dot(a.rows[3], b));
}

mat4 operator*(const mat4 &a, const mat4 &b) {

    mat4 result;

    result.rows[0].x = dot(a.rows[0], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[0].y = dot(a.rows[0], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[0].z = dot(a.rows[0], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[0].w = dot(a.rows[0], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[1].x = dot(a.rows[1], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[1].y = dot(a.rows[1], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[1].z = dot(a.rows[1], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[1].w = dot(a.rows[1], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[2].x = dot(a.rows[2], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[2].y = dot(a.rows[2], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[2].z = dot(a.rows[2], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[2].w = dot(a.rows[2], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[3].x = dot(a.rows[3], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[3].y = dot(a.rows[3], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[3].z = dot(a.rows[3], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[3].w = dot(a.rows[3], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    return result;
}

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

constexpr float PI = 3.1415;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

struct RasterizationSettings {
    int colorMode = COLORMODE_ID;
    mat4 world;
};

struct Grid2D {
    uint32_t *cellIndices;
    int rows;
    int cols;
    int count;
};

float3 colorFromId(uint32_t id) {
    if (id == 0) {
        return float3{0.0, 0.8, 0.2};
    }

    return float3{1.0, 0.0, 1.0};
}

// rasterizes voxels
// - each block grabs a voxel
// - all threads of that block process different fragments of the voxel
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeGrid(Grid2D *grid2D, uint64_t *framebuffer, RasterizationSettings settings) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int colorMode = settings.colorMode;

    mat4 transform = uniforms.proj * uniforms.view * settings.world;

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

            int idx = sh_cellIndex;
            int x = idx % grid2D->cols;
            int y = idx / grid2D->cols;

            float4 cornerLow_W = make_float4(x, y, 0.0f, 1.0f);
            float4 cornerHigh_W = cornerLow_W + float4{0.9f, 0.9f, 0.0f, 0.0f};

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

                float3 color = colorFromId(grid2D->cellIndices[sh_cellIndex]);
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
                                  uint32_t numCols, uint32_t *gridCells) {
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

    { // generate and draw a single voxel
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        grid2D->cellIndices = gridCells;
        grid2D->rows = numRows;
        grid2D->cols = numCols;
        grid2D->count = numRows * numCols;

        RasterizationSettings settings;
        settings.colorMode = COLORMODE_ID;
        settings.world = mat4::identity();

        rasterizeGrid(grid2D, framebuffer, settings);
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
