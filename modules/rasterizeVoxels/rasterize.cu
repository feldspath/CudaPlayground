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

struct Voxels {
    int count;
    float3 *positions;
    float3 *colors;
};

struct RasterizationSettings {
    int colorMode = COLORMODE_ID;
    mat4 world;
};

struct Ray {
    float3 origin;
    float3 dir;
};

struct Box {
    float3 center;
    float radius;
};

struct HitInfo {
    bool hit;
    float distance;
    float3 normal;
};

float sign(float v) {
    float iszero = v == 0.0f ? 0.0f : 1.0f;
    return iszero * (v > 0.0f ? 1.0f : -1.0f);
}

float3 sign(float3 v) {
    float3 res = {sign(v.x), sign(v.y), sign(v.z)};
    return res;
}

// From A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
// (https://jcgt.org/published/0007/03/04/paper.pdf)
HitInfo rayBoxIntersection(Box box, Ray ray) {
    ray.origin -= box.center;
    float3 sgn = -1.0f * sign(ray.dir);

    float3 d = box.radius * sgn - ray.origin;
    d /= ray.dir;

#define TEST(U, V, W)                                                                              \
    (d.U >= 0.0) && fabs(ray.origin.V + ray.dir.V * d.U) <                                         \
                        box.radius &&fabs(ray.origin.W + ray.dir.W * d.U) < box.radius
    int3 test = int3{TEST(x, y, z), TEST(y, z, x), TEST(z, x, y)};
    sgn = test.x ? float3{sgn.x, 0.0f, 0.0f}
                 : (test.y ? float3{0.0f, sgn.y, 0.0f} : float3{0.0f, 0.0f, test.z ? sgn.z : 0.0f});
#undef TEST

    HitInfo hitInfo;
    hitInfo.distance = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
    hitInfo.normal = sgn;
    hitInfo.hit = (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);
    return hitInfo;
}

// rasterizes voxels
// - each block grabs a voxel
// - all threads of that block process different fragments of the voxel
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeVoxels(Voxels *voxels, uint64_t *framebuffer, RasterizationSettings settings) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int colorMode = settings.colorMode;

    mat4 transform = uniforms.proj * uniforms.view * settings.world;

    uint32_t &processedVoxels = *allocator->alloc<uint32_t *>(4);
    if (grid.thread_rank() == 0) {
        processedVoxels = 0;
    }
    grid.sync();

    {
        __shared__ int sh_voxelIndex;

        block.sync();

        // safety mechanism: each block draws at most <loop_max> voxels
        int loop_max = 10'000;
        for (int loop_i = 0; loop_i < loop_max; loop_i++) {

            // grab the index of the next unprocessed voxel
            block.sync();
            if (block.thread_rank() == 0) {
                sh_voxelIndex = atomicAdd(&processedVoxels, 1);
            }
            block.sync();

            if (sh_voxelIndex >= voxels->count)
                break;

            // project x/y to pixel coords
            // z: whatever
            // w: linear depth
            auto toScreenCoord_SW = [&](float3 p) {
                float4 pos = transform * float4{p.x, p.y, p.z, 1.0f};

                pos.x = pos.x / pos.w;
                pos.y = pos.y / pos.w;

                float4 imgPos = {(pos.x * 0.5f + 0.5f) * uniforms.width,
                                 (pos.y * 0.5f + 0.5f) * uniforms.height, pos.z, pos.w};

                return imgPos;
            };

            float3 voxel_pos_W = voxels->positions[sh_voxelIndex];
            float4 voxel_pos_S = toScreenCoord_SW(voxel_pos_W);

            // cull the voxel if its position is closer than depth 0
            // if(voxel_screen_pos.w < 0.0) continue;

            // TODO: compute screen space bbox of voxel

            // clamp to screen
            float min_x = 0.0f;
            float min_y = 0.0f;
            float max_x = uniforms.width;
            float max_y = uniforms.height;

            int size_x = ceil(max_x) - floor(min_x);
            int size_y = ceil(max_y) - floor(min_y);
            int numFragments = size_x * size_y;

            // iterate through fragments in bounding rectangle and draw if within triangle
            int numProcessedSamples = 0;
            for (int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()) {

                // safety mechanism: don't draw more than <x> pixels per thread
                if (numProcessedSamples > 5'000)
                    break;

                int fragID = fragOffset + block.thread_rank();
                int fragX = fragID % size_x;
                int fragY = fragID / size_x;

                float2 pFrag = {floor(min_x) + float(fragX), floor(min_y) + float(fragY)};

                int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
                int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
                pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

                float3 color = voxels->colors[sh_voxelIndex];

                auto pixelDirection_C = [&](int2 pixelCoords) {
                    float2 px = make_float2(pixelCoords);
                    float4 ndc = {px.x / uniforms.width * 2.0f - 1.0f,
                                  px.y / uniforms.height * 2.0f - 1.0f, 1.0, 1.0};

                    float4 homScreenPos = uniforms.invproj * ndc;
                    float3 screenPos = make_float3(homScreenPos) / homScreenPos.w;
                    return normalize(screenPos);
                };

                float3 dir_C = pixelDirection_C(pixelCoords);

                float3 camPos_W = float3{uniforms.invview.rows[0].w, uniforms.invview.rows[1].w,
                                         uniforms.invview.rows[2].w};

                float3 dir_W = make_float3(uniforms.invview * make_float4(dir_C, 0.0f));
                Ray ray_W = Ray{camPos_W, dir_W};
                Box box = Box{voxel_pos_W, 1.0f};

                HitInfo hitInfo_W = rayBoxIntersection(box, ray_W);
                if (!hitInfo_W.hit)
                    continue;

                float3 lightDir_W = normalize(float3{2.0f, -1.0f, -4.0f});
                float3 fragColor =
                    color * min(max(-dot(hitInfo_W.normal, lightDir_W), 0.0f) + 0.2f, 1.0f);

                // if(colorMode == COLORMODE_ID){
                // 	// TRIANGLE INDEX
                // 	color = sh_voxelIndex * 123456;
                // }else if(colorMode == COLORMODE_TIME || colorMode == COLORMODE_TIME_NORMALIZED){
                // 	// TIME
                // 	uint64_t nanotime;
                // 	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));
                // 	color = (nanotime - nanotime_start) % 0x00ffffffull;
                // }else{
                // 	// WHATEVER
                // 	color = sh_voxelIndex * 123456;
                // }

                float depth = hitInfo_W.distance;
                uint64_t udepth = *((uint32_t *)&depth);
                uint64_t pixel = (udepth << 32ull) | rgb8color(fragColor);

                atomicMin(&framebuffer[pixelID], pixel);

                numProcessedSamples++;
            }
        }
    }
}

extern "C" __global__ void kernel(const Uniforms _uniforms, unsigned int *buffer,
                                  cudaSurfaceObject_t gl_colorbuffer, uint32_t numVoxels,
                                  float3 *voxelPositions) {
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
        Voxels *voxels = allocator->alloc<Voxels *>(sizeof(Voxels));
        voxels->positions = voxelPositions;
        voxels->colors = allocator->alloc<float3 *>(sizeof(float3) * numVoxels);

        for (int i = 0; i < numVoxels; ++i) {
            float3 color;
            if (i % 2 == 0) {
                color = float3{0.0f, 1.0f, 1.0f};
            } else {
                color = float3{1.0f, 1.0f, 0.0f};
            }
            voxels->colors[i] = color;
        }

        voxels->count = numVoxels;

        RasterizationSettings settings;
        settings.colorMode = COLORMODE_ID;
        settings.world = mat4::identity();

        // when drawing time, due to normalization, everything needs to be colored by time
        // lets draw the ground with non-normalized time as well for consistency
        if (uniforms.colorMode == COLORMODE_TIME) {
            settings.colorMode = COLORMODE_TIME_NORMALIZED;
        } else if (uniforms.colorMode == COLORMODE_TIME_NORMALIZED) {
            settings.colorMode = COLORMODE_TIME_NORMALIZED;
        }

        rasterizeVoxels(voxels, framebuffer, settings);
    }

    grid.sync();

    uint32_t &maxNanos = *allocator->alloc<uint32_t *>(4);

    // if colored by normalized time, we compute the max time for normalization
    if (uniforms.colorMode == COLORMODE_TIME_NORMALIZED) {
        if (grid.thread_rank() == 0) {
            maxNanos = 0;
        }
        grid.sync();

        processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex) {
            int x = pixelIndex % int(uniforms.width);
            int y = pixelIndex / int(uniforms.width);

            uint64_t encoded = framebuffer[pixelIndex];
            uint32_t color = encoded & 0xffffffffull;

            if (color != BACKGROUND_COLOR) {
                atomicMax(&maxNanos, color);
            }
        });

        grid.sync();
    }

    // transfer framebuffer to opengl texture
    processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex) {
        int x = pixelIndex % int(uniforms.width);
        int y = pixelIndex / int(uniforms.width);

        uint64_t encoded = framebuffer[pixelIndex];
        uint32_t color = encoded & 0xffffffffull;

        if (uniforms.colorMode == COLORMODE_TIME_NORMALIZED)
            if (color != BACKGROUND_COLOR) {
                color = color / (maxNanos / 255);
            }

        surf2Dwrite(color, gl_colorbuffer, x * 4, y);
    });
}
