#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/Entities/entities.cuh"
#include "World/map.cuh"
#include "builtin_types.h"
#include "common/helper_math.h"
#include "common/matrix_math.h"
#include "framebuffer.cuh"
#include "gui.cuh"
#include "sprite.cuh"
#include "text.cuh"

namespace cg = cooperative_groups;

GameData gameData;
Allocator *allocator;
uint64_t nanotime_start;

constexpr uint32_t BACKGROUND_COLOR = 0x00332211ull;

// https://coolors.co/palette/8cb369-f4e285-f4a259-5b8e7d-bc4b51
float3 LIGHT_GREEN = float3{140.0 / 255, 179.0 / 255, 105.0 / 255};
float3 GREEN = float3{0.8f * 140.0 / 255, 0.8f * 179.0 / 255, 0.8f * 105.0 / 255};
float3 LIGHT_BLUE = float3{91.0 / 255, 142.0 / 255, 125.0 / 255};
float3 RED = float3{188.0 / 255, 75.0 / 255, 81.0 / 255};
float3 YELLOW = float3{244.0 / 255, 226.0 / 255, 133.0 / 255};
float3 GRAY = float3{0.5f, 0.5f, 0.5f};
float3 PURPLE = float3{1.0, 0.0, 1.0};

float3 colorFromId(uint32_t id) {
    switch (id) {
    case GRASS:
        return GREEN;
    case ROAD:
        return GRAY;
    case HOUSE:
        return LIGHT_GREEN;
    case FACTORY:
        return YELLOW;
    case SHOP:
        return LIGHT_BLUE;
    default:
        return PURPLE;
    }
}

// rasterizes the grid
// - Each thread computes the color of a pixel.
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeGrid(Map &map, Entities *entities, SpriteSheet sprites, Framebuffer framebuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    processRange(gameData.uniforms.width * gameData.uniforms.height, [&](int pixelId) {
        int pixelX = pixelId % int(gameData.uniforms.width);
        int pixelY = pixelId / int(gameData.uniforms.width);

        float2 pFrag = make_float2(pixelX, pixelY);

        float3 pos_W = unproject(pFrag, gameData.uniforms.invview * gameData.uniforms.invproj,
                                 gameData.uniforms.width, gameData.uniforms.height);

        MapId cell = map.cellAtPosition(float2{pos_W.x, pos_W.y});
        if (!cell.valid()) {
            return;
        }
        auto &chunk = map.getChunk(cell.chunkId);
        auto sh_cellIndex = cell.cellId;

        float2 cellCenter = chunk.getCellPosition(sh_cellIndex);
        float2 diff = float2{pos_W.x - cellCenter.x, pos_W.y - cellCenter.y};

        float2 diffToCorner = diff + float2{CELL_RADIUS, CELL_RADIUS};
        float u = diffToCorner.x / (CELL_RADIUS * 2);
        float v = diffToCorner.y / (CELL_RADIUS * 2);

        if (abs(diff.x) > CELL_RADIUS || abs(diff.y) > CELL_RADIUS) {
            return;
        }

        float3 color;
        if (gameData.uniforms.renderMode == RENDERMODE_DEFAULT ||
            gameData.uniforms.renderMode == RENDERMODE_FLOWFIELD) {
            switch (chunk.get(sh_cellIndex).tileId) {
            case GRASS:
                color = sprites.grass.sampleFloat(u, v);
                break;
            case HOUSE:
                if (chunk.getTyped<HouseCell>(sh_cellIndex).residentCount > 0) {
                    color = sprites.house.sampleFloat(u, v);
                    break;
                }
            default:
                color = colorFromId(chunk.get(sh_cellIndex).tileId);
                break;
            }
            color *= GameState::instance->gameTime.timeOfDay().toFloat() * 0.5 + 0.5;

        } else if (gameData.uniforms.renderMode == RENDERMODE_NETWORK ||
                   gameData.uniforms.renderMode == RENDERMODE_NETWORK_CHUNK) {
            TileId tileId = chunk.get(sh_cellIndex).tileId;
            if (tileId == GRASS || tileId == UNKNOWN) {
                color = {0.0f, 0.0f, 0.0f};
            } else {
                int colorId;
                if (tileId == HOUSE) {
                    colorId = 0;
                } else if (tileId == ROAD) {
                    if (gameData.uniforms.renderMode == RENDERMODE_NETWORK) {
                        colorId = map.getTyped<RoadCell>(cell).networkRepr.cellId;
                    } else {
                        colorId = map.getTyped<RoadCell>(cell).chunkNetworkRepr;
                    }
                } else {
                    colorId = sh_cellIndex;
                }

                float r = (float)(colorId % 3) / 3.0;
                float g = (float)(colorId % 11) / 11.0;
                float b = (float)(colorId % 37) / 37.0;
                color = float3{r, g, b};
                auto &road = map.getTyped<RoadCell>(cell);
                if (road.chunkNetworkRepr == cell.cellId &&
                    gameData.uniforms.renderMode == RENDERMODE_NETWORK_CHUNK) {
                    color = float3{1.0f, 1.0f, 1.0f};
                }

                // if (tileId == ROAD) {
                //     color *= (float)(map->roadNetworkId(sh_cellIndex)) /
                //              map->roadNetworkId(map->roadNetworkRepr(sh_cellIndex));
                // }

                if (colorId == -1) {
                    color = float3{1.0f, 0.0f, 1.0f};
                }
            }
        } else {
            color = float3{0.0f, 0.0f, 0.0f};
        }
        float3 pixelColor = color;

        float depth = 1.0f;
        uint64_t udepth = *((uint32_t *)&depth);
        uint64_t pixel = (udepth << 32ull) | rgb8color(pixelColor);

        atomicMin(&framebuffer.data[pixelId], pixel);
    });
}

void rasterizeEntities(Entities *entities, Framebuffer framebuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    mat4 viewProj = gameData.uniforms.proj * gameData.uniforms.view;

    float sphereRadius =
        length(projectVectorToScreenPos(float3{ENTITY_RADIUS, 0.0f, 0.0f}, viewProj,
                                        gameData.uniforms.width, gameData.uniforms.height));
    // sphereRadius = 5.0f;
    //  Each thread grabs an entity
    entities->processAll([&](int entityIndex) {
        float2 entityPos = entities->get(entityIndex).position;
        float2 screenPos = projectPosToScreenPos(make_float3(entityPos, 0.0f), viewProj,
                                                 gameData.uniforms.width, gameData.uniforms.height);

        float min_x = screenPos.x - sphereRadius;
        float max_x = screenPos.x + sphereRadius;
        float min_y = screenPos.y - sphereRadius;
        float max_y = screenPos.y + sphereRadius;

        min_x = clamp(min_x, 0.0f, gameData.uniforms.width);
        min_y = clamp(min_y, 0.0f, gameData.uniforms.height);
        max_x = clamp(max_x, 0.0f, gameData.uniforms.width);
        max_y = clamp(max_y, 0.0f, gameData.uniforms.height);

        int size_x = ceil(max_x) - floor(min_x);
        int size_y = ceil(max_y) - floor(min_y);
        int numFragments = size_x * size_y;
        for (int fragID = 0; fragID < numFragments; fragID++) {
            int fragX = fragID % size_x;
            int fragY = fragID / size_x;

            float2 pFrag = {floor(min_x) + float(fragX), floor(min_y) + float(fragY)};

            if (length(pFrag - screenPos) >= sphereRadius) {
                continue;
            }

            int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
            int pixelID = pixelCoords.x + pixelCoords.y * gameData.uniforms.width;
            pixelID =
                clamp(pixelID, 0, int(gameData.uniforms.width * gameData.uniforms.height) - 1);

            float3 color = make_float3(1.0f, 0.0f, 0.0f) *
                           (GameState::instance->gameTime.timeOfDay().toFloat() * 0.5 + 0.5);

            float depth = 0.9f;
            uint64_t udepth = *((uint32_t *)&depth);
            uint64_t pixel = (udepth << 32ull) | rgb8color(color);

            atomicMin(&framebuffer.data[pixelID], pixel);
        }
    });
}

void rasterizeEntitiesFlowfield(Entities *entities, Framebuffer framebuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    mat4 viewProj = gameData.uniforms.proj * gameData.uniforms.view;

    float sphereRadius =
        length(projectVectorToScreenPos(float3{ENTITY_RADIUS, 0.0f, 0.0f}, viewProj,
                                        gameData.uniforms.width, gameData.uniforms.height));
    // sphereRadius = 5.0f;
    //  Each thread grabs an entity
    entities->processAllActive([&](int entityIndex) {
        auto &entity = entities->get(entityIndex);
        if (!entity.path.isValid()) {
            return;
        }

        float2 screenPos = projectPosToScreenPos(make_float3(entity.position, 0.0f), viewProj,
                                                 gameData.uniforms.width, gameData.uniforms.height);

        for (int i = 0; i < 30; ++i) {
            float depth = 0.8f;
            uint64_t udepth = *((uint32_t *)&depth);
            uint64_t pixel = (udepth << 32ull) | rgb8color(make_float3(0.0f, 1.0f, 0.0f));

            int2 pixelCoords =
                make_int2(screenPos + normalize(directionFromEnum(entity.path.nextDir())) * i);

            if (pixelCoords.x < 0 || pixelCoords.x >= framebuffer.width || pixelCoords.y < 0 ||
                pixelCoords.y >= framebuffer.height) {
                continue;
            }

            int pixelID = pixelCoords.x + pixelCoords.y * gameData.uniforms.width;
            atomicMin(&framebuffer.data[pixelID], pixel);
        }
    });
}

void rasterizeFlowfield(Map &map, Framebuffer &framebuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (!(gameData.uniforms.cursorPos.x >= 0 &&
          gameData.uniforms.cursorPos.x < gameData.uniforms.width &&
          gameData.uniforms.cursorPos.y >= 0 &&
          gameData.uniforms.cursorPos.y < gameData.uniforms.height)) {
        return;
    }

    mat4 viewProj = gameData.uniforms.proj * gameData.uniforms.view;

    float cellRadius =
        length(projectVectorToScreenPos(float3{CELL_RADIUS, 0.0f, 0.0f}, viewProj,
                                        gameData.uniforms.width, gameData.uniforms.height));
    float2 px = float2{gameData.uniforms.cursorPos.x,
                       gameData.uniforms.height - gameData.uniforms.cursorPos.y};
    float3 pos_W = unproject(px, gameData.uniforms.invview * gameData.uniforms.invproj,
                             gameData.uniforms.width, gameData.uniforms.height);
    float2 worldPos = float2{pos_W.x, pos_W.y};

    MapId flowfieldId = map.cellAtPosition(worldPos);
    if (!flowfieldId.valid()) {
        return;
    }

    Chunk &chunk = map.getChunk(flowfieldId.chunkId);

    if (chunk.cachedFlowfields[flowfieldId.cellId].state != VALID) {
        return;
    }

    chunk.processEachCell([&](int cellId) {
        float2 dir = directionFromEnum(
            Direction(chunk.cachedFlowfields[flowfieldId.cellId].directions[cellId]));

        float2 cellCenter = chunk.getCellPosition(cellId);
        float2 screenPos = projectPosToScreenPos(make_float3(cellCenter, 0.0f), viewProj,
                                                 gameData.uniforms.width, gameData.uniforms.height);
        for (int i = 0; i < int(cellRadius); ++i) {
            float2 pFrag = screenPos + dir * i;
            int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
            if (pixelCoords.x < 0 || pixelCoords.x >= framebuffer.width || pixelCoords.y < 0 ||
                pixelCoords.y >= framebuffer.height) {
                continue;
            }

            int pixelID = pixelCoords.x + pixelCoords.y * gameData.uniforms.width;

            float3 color = make_float3(0.0f, 0.0f, 1.0f);

            float depth = 0.85f;
            uint64_t udepth = *((uint32_t *)&depth);
            uint64_t pixel = (udepth << 32ull) | rgb8color(color);

            atomicMin(&framebuffer.data[pixelID], pixel);
        }
    });
}

extern "C" __global__ void kernel(GameData _gameData, cudaSurfaceObject_t gl_colorbuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

    gameData = _gameData;
    GameState::instance = gameData.state;

    Allocator _allocator(gameData.buffer, 0);
    allocator = &_allocator;

    Font font(gameData.img_ascii_16);
    TextRenderer textRenderer(font);
    SpriteSheet sprites(gameData.img_spritesheet);

    // allocate framebuffer memory
    int framebufferSize =
        int(gameData.uniforms.width) * int(gameData.uniforms.height) * sizeof(uint64_t);
    uint64_t *framebufferData = allocator->alloc<uint64_t *>(framebufferSize);

    Framebuffer framebuffer(uint32_t(gameData.uniforms.width), uint32_t(gameData.uniforms.height),
                            framebufferData);

    // clear framebuffer
    framebuffer.clear(uint64_t(BACKGROUND_COLOR));

    grid.sync();

    {
        Map *map = allocator->alloc<Map *>(sizeof(Map));
        *map = Map(gameData.numRows, gameData.numCols, gameData.chunks);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(gameData.entitiesBuffer);

        rasterizeGrid(*map, entities, sprites, framebuffer);
        grid.sync();
        rasterizeEntities(entities, framebuffer);
        grid.sync();
        if (gameData.uniforms.renderMode == RENDERMODE_FLOWFIELD) {
            rasterizeFlowfield(*map, framebuffer);
            grid.sync();
        }

        if (gameData.uniforms.displayFlowfield) {
            rasterizeEntitiesFlowfield(entities, framebuffer);
            grid.sync();
        }

        GUI gui(framebuffer, textRenderer, sprites,
                gameData.uniforms.proj * gameData.uniforms.view);
        gui.render(*map, entities);
    }

    grid.sync();

    uint32_t &maxNanos = *allocator->alloc<uint32_t *>(4);

    // transfer framebuffer to opengl texture
    processRange(0, framebuffer.width * framebuffer.height, [&](int pixelIndex) {
        int x = pixelIndex % int(framebuffer.width);
        int y = pixelIndex / int(framebuffer.width);

        uint64_t encoded = framebuffer.data[pixelIndex];
        uint32_t color = encoded & 0xffffffffull;

        surf2Dwrite(color, gl_colorbuffer, x * 4, y);
    });
}
