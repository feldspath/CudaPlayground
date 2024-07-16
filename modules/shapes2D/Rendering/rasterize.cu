#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "HostDeviceInterface.h"
#include "World/map.cuh"
#include "builtin_types.h"
#include "common/helper_math.h"
#include "common/matrix_math.h"
#include "common/utils.cuh"
#include "framebuffer.cuh"
#include "gui.cuh"
#include "sprite.cuh"
#include "text.cuh"

#include "ObjectSelection.cuh"

namespace cg = cooperative_groups;

GameData gamedata;
GameState *GameState::instance;
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
float3 DARK_GRAY = float3{0.15, 0.15, 0.15};
float3 BLUE = float3{0.0f, 0.0f, 1.0f};

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
    case STONE:
        return DARK_GRAY;
    case WATER:
        return BLUE;
    default:
        return PURPLE;
    }
}

// rasterizes the grid
// - Each thread computes the color of a pixel.
// - <framebuffer> stores interleaved 32bit depth and color values
// - The closest fragments are rendered via atomicMin on a combined 64bit depth&color integer
//   atomicMin(&framebuffer[pixelIndex], (depth << 32 | color));
void rasterizeGrid(Map map, Entity* entities, uint32_t numEntities, SpriteSheet sprites, Framebuffer framebuffer) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto &uniforms = gamedata.uniforms;

    uint32_t numObjects = 0;
    auto cursorPos = uniforms.cursorPos;
    ObjectSelectionSprite* objects = ObjectSelection::createPanel(allocator, numObjects, cursorPos.x, uniforms.height - cursorPos.y, gamedata);
    ObjectSelectionSprite object = objects[gamedata.state->buildingType];

    grid.sync();

    auto doesTileIntersectHoveredObject = [&](int tx, int ty){

        int cellIndex = map.cellAtPosition(float2{float(tx), float(ty)});

        if(cellIndex < 0) return false;
        if(!gamedata.state->isPlacingBuilding) return false;

        int pixelX = cursorPos.x;
        int pixelY = uniforms.height - cursorPos.y;

        float2 frag = {
            cursorPos.x,
            (uniforms.height - cursorPos.y),
        };

        float3 pos_W = unproject(frag, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
        int btx = pos_W.x - object.cellSize.x * 0.5;
        int bty = pos_W.y  - object.cellSize.y * 0.5;

        if(tx < btx) return false;
        if(ty < bty) return false;
        if(tx >= btx + object.cellSize.x) return false;
        if(ty >= bty + object.cellSize.y) return false;

        // if(grid.thread_rank() == 0){
        //     printf("%d, %d \n", btx, bty);
        // }

        return true;
    };

    auto doesMouseIntersectTile = [&](int tx, int ty){

        int cellIndex = map.cellAtPosition(float2{float(tx), float(ty)});

        if(cellIndex < 0) return false;

        int pixelX = cursorPos.x;
        int pixelY = uniforms.height - cursorPos.y;

        float2 frag = {
            cursorPos.x,
            (uniforms.height - cursorPos.y),
        };

        float3 pos_W = unproject(frag, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
        int btx = pos_W.x;
        int bty = pos_W.y;

        if(tx < btx) return false;
        if(ty < bty) return false;
        if(tx > btx) return false;
        if(ty > bty) return false;

        // if(grid.thread_rank() == 0){
        //     printf("%d, %d \n", btx, bty);
        // }

        return true;
    };

    grid.sync();




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
        int sh_cellIndex = map.cellAtPosition(float2{pos_W.x, pos_W.y});
        if (sh_cellIndex == -1) {
            continue;
        }

        bool tileIsHovered = doesTileIntersectHoveredObject(pos_W.x, pos_W.y);


        float2 cellCenter = map.getCellPosition(sh_cellIndex);
        float2 diff = float2{pos_W.x - cellCenter.x, pos_W.y - cellCenter.y};

        float2 diffToCorner = diff + float2{CELL_RADIUS, CELL_RADIUS};
        float u = diffToCorner.x / (CELL_RADIUS * 2);
        float v = diffToCorner.y / (CELL_RADIUS * 2);

        if (abs(diff.x) > CELL_RADIUS || abs(diff.y) > CELL_RADIUS) {
            continue;
        }

        float3 color;
        if (uniforms.renderMode == RENDERMODE_DEFAULT) {
            switch (map.getTileId(sh_cellIndex)) {
            case GRASS:
                color = sprites.grass.sampleFloat(u, v);
                break;
            case HOUSE:
                if (*map.houseTileData(sh_cellIndex) != -1) {
                    color = sprites.house.sampleFloat(u, v);
                    break;
                }
            default:
                color = colorFromId(map.getTileId(sh_cellIndex));
                break;
            }
            color *= GameState::instance->gameTime.timeOfDay().toFloat() * 0.5 + 0.5;

        } else if (uniforms.renderMode == RENDERMODE_NETWORK) {
            TileId tileId = map.getTileId(sh_cellIndex);
            if (tileId == GRASS || tileId == UNKNOWN) {
                color = {0.0f, 0.0f, 0.0f};
            } else {
                int colorId;
                if (tileId == HOUSE) {
                    int entityId = *(map.houseTileData(sh_cellIndex));
                    if (entityId == -1) {
                        colorId = -1;
                    } else {
                        colorId = entities[entityId].workplaceId;
                    }
                } else if (tileId == ROAD) {
                    colorId = map.roadNetworkRepr(sh_cellIndex);
                } else {
                    colorId = sh_cellIndex;
                }

                float r = (float)(colorId % 3) / 3.0;
                float g = (float)(colorId % 11) / 11.0;
                float b = (float)(colorId % 37) / 37.0;
                color = float3{r, g, b};

                if (tileId == ROAD) {
                    color *= (float)(map.roadNetworkId(sh_cellIndex)) /
                             map.roadNetworkId(map.roadNetworkRepr(sh_cellIndex));
                }

                if (colorId == -1) {
                    color = float3{1.0f, 0.0f, 1.0f};
                }
            }
        } else if (uniforms.renderMode == RENDERMODE_LANDVALUE) {
            int value = map.cellsData[sh_cellIndex].landValue;
            float a = float(value) / 255.0f;
            color = float3{0.0f, 1.0f, 0.0f} * a + float3{1.0f, 0.0f, 0.0f} * (1 - a);
        }

        bool tileIsFree = map.cellsData[sh_cellIndex].buildingID == -1;
        if(tileIsHovered && tileIsFree){
            color.x = 0.5 * color.x + 0.3;
            color.y = 0.5 * color.y + 0.3;
            color.z = 0.5 * color.z + 0.3;
        }else if(tileIsHovered && !tileIsFree){
            color.x = 0.5 * color.x + 0.3;
            color.y = 0.5 * color.y + 0.1;
            color.z = 0.5 * color.z + 0.1;
        }else if(doesMouseIntersectTile(pos_W.x, pos_W.y)){
            color.x = 0.5 * color.x + 0.3;
            color.y = 0.5 * color.y + 0.3;
            color.z = 0.5 * color.z + 0.3;
        }

        // if(map.cellsData[sh_cellIndex].buildingID >= 0){
        //     color = {0.0, 1.0, 0.0};
        // }

        float3 pixelColor = color;

        float depth = 1.0f;
        uint64_t udepth = *((uint32_t *)&depth);
        uint64_t pixel = (udepth << 32ull) | rgb8color(pixelColor);

        atomicMin(&framebuffer.data[pixelId], pixel);
    }
}

void rasterizeEntities(Entity *entities, uint32_t numEntities, Framebuffer framebuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto& uniforms = gamedata.uniforms;

    mat4 viewProj = uniforms.proj * uniforms.view;

    float sphereRadius = length(projectVectorToScreenPos(
        float3{ENTITY_RADIUS, 0.0f, 0.0f}, viewProj, uniforms.width, uniforms.height));
    // sphereRadius = 5.0f;
    //  Each thread grabs an entity
    for (int offset = 0; offset < numEntities; offset += grid.num_threads()) {
        int entityIndex = offset + grid.thread_rank();
        if (entityIndex >= numEntities) {
            break;
        }

        float2 entityPos = entities[entityIndex].position;
        float2 screenPos = projectPosToScreenPos(make_float3(entityPos, 0.0f), viewProj,
                                                 uniforms.width, uniforms.height);

        float min_x = screenPos.x - sphereRadius;
        float max_x = screenPos.x + sphereRadius;
        float min_y = screenPos.y - sphereRadius;
        float max_y = screenPos.y + sphereRadius;

        min_x = clamp(min_x, 0.0f, uniforms.width);
        min_y = clamp(min_y, 0.0f, uniforms.height);
        max_x = clamp(max_x, 0.0f, uniforms.width);
        max_y = clamp(max_y, 0.0f, uniforms.height);

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
            int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
            pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

            float3 color = make_float3(1.0f, 0.0f, 0.0f) *
                           (GameState::instance->gameTime.timeOfDay().toFloat() * 0.5 + 0.5);

            float depth = 0.9f;
            uint64_t udepth = *((uint32_t *)&depth);
            uint64_t pixel = (udepth << 32ull) | rgb8color(color);

            atomicMin(&framebuffer.data[pixelID], pixel);
        }
    }
}

extern "C" __global__ void kernel(GameData _gamedata, cudaSurfaceObject_t gl_colorbuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // return;
    
    gamedata = _gamedata;
    auto& uniforms = gamedata.uniforms;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

    Allocator _allocator(gamedata.buffer, 0);
    allocator = &_allocator;

    GameState::instance = gamedata.state;

    Font font(gamedata.img_ascii_16);
    TextRenderer textRenderer(font);
    SpriteSheet sprites(gamedata.img_spritesheet);

    // allocate framebuffer memory
    int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
    uint64_t *framebufferData = allocator->alloc<uint64_t *>(framebufferSize);

    Framebuffer framebuffer(uint32_t(uniforms.width), uint32_t(uniforms.height), framebufferData);

    // clear framebuffer
    framebuffer.clear(uint64_t(BACKGROUND_COLOR));

    grid.sync();

    {
        Map map = Map(gamedata.numRows, gamedata.numCols, gamedata.cells);

        // Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        // *entities = Entities(gamedata.entitiesBuffer);

        rasterizeGrid(map, gamedata.entities, gamedata.state->numEntities, sprites, framebuffer);
        grid.sync();
        rasterizeEntities(gamedata.entities, gamedata.state->numEntities, framebuffer);
        grid.sync();
        GUI gui(framebuffer, textRenderer, sprites, uniforms.proj * uniforms.view);
        gui.render(map, gamedata.entities, gamedata.state->numEntities);
    }

    grid.sync();

    { // DRAW CONSTRUCTIONS

        auto cursor = gamedata.uniforms.cursorPos;
        uint32_t numObjects = 0;
        int mouseX = cursor.x;
        int mouseY = uniforms.height - cursor.y;
        ObjectSelectionSprite* objects = ObjectSelection::createPanel(allocator, numObjects, mouseX, mouseY, gamedata);

        for_blockwise(gamedata.constructions->numConstructions, [&](int index){
            Construction construction = gamedata.constructions->items[index];

            mat4 viewProj = uniforms.proj * uniforms.view;
            float3 cellPos = {
                construction.tile_x, construction.tile_y, 0.0f
            };
            float2 screenPos = projectPosToScreenPos(cellPos, viewProj,
                                                 uniforms.width, uniforms.height);

            float2 p0 = projectPosToScreenPos(float3{0.0, 0.0, 0.0}, viewProj, uniforms.width, uniforms.height);
            float2 p1 = projectPosToScreenPos(float3{1.0, 1.0, 0.0}, viewProj, uniforms.width, uniforms.height);

            float cellPixelSize = p1.x - p0.x;    

            ObjectSelectionSprite& object = objects[construction.type];
            object.depth = 0.2;
            object.position.x = screenPos.x;
            object.position.y = screenPos.y;
            object.size.x = object.cellSize.x * cellPixelSize;
            object.size.y = object.cellSize.y * cellPixelSize;

            ObjectSelection::rasterize_blockwise(object, framebuffer);
        });

    }

    grid.sync();


    { // DRAW OBJECT SELECTION PANEL
        auto cursor = gamedata.uniforms.cursorPos;
        uint32_t numObjects = 0;
        int mouseX = cursor.x;
        int mouseY = uniforms.height - cursor.y;
        ObjectSelectionSprite* objects = ObjectSelection::createPanel(allocator, numObjects, mouseX, mouseY, gamedata);

        grid.sync();

        // draw building panel
        for_blockwise(numObjects, [&](int index){
            ObjectSelectionSprite& object = objects[index];

            ObjectSelection::rasterize_blockwise(object, framebuffer);
        });


        grid.sync();

        // Draw black background of hovered panel
        if(grid.block_rank() == 0){

            for(int i = 0; i < numObjects; i++){
                ObjectSelectionSprite object = objects[i];
                if(object.hovered){
                    object.position.x = object.position.x + object.size.x * 0.03;
                    object.position.y = object.position.y - object.size.y * 0.05;
                    object.size = object.size * 1.05;
                    object.depth = 0.07;

                    ObjectSelection::rasterize_blockwise(object, framebuffer, true);
                }
            }
        }

        grid.sync();

        // Draw label of currently hovered building
        Font font(gamedata.img_ascii_16);
        TextRenderer textRenderer(font);

        for(int i = 0; i < numObjects; i++){
            
            ObjectSelectionSprite object = objects[i];

            if(object.hovered){

                
                float x = object.position.x + 50.0f; // - particle.size.x / 2.0f;
                float y = object.position.y;

                Cursor cblack  = textRenderer.newCursor(20.0f, x + 2, y + 20 - 2);
                Cursor cblack2 = textRenderer.newCursor(20.0f, x - 2, y + 20 - 2);
                Cursor cblack3 = textRenderer.newCursor(20.0f, x + 0, y + 20 - 2);
                Cursor cwhite  = textRenderer.newCursor(20.0f, x + 0, y + 20 + 0);
                cwhite.textColor = {1.0f, 1.0f, 1.0f};

                textRenderer.drawText(object.label, cblack, framebuffer);
                textRenderer.drawText(object.label, cblack2, framebuffer);
                textRenderer.drawText(object.label, cblack3, framebuffer);
                textRenderer.drawText(object.label, cwhite, framebuffer);

            }

        }

        grid.sync();

        // Draw building that is currently being placed
        if(gamedata.state->isPlacingBuilding && grid.block_rank() == 0){
            ObjectSelectionSprite object = objects[gamedata.state->buildingType];

            mat4 viewProj = uniforms.proj * uniforms.view;
            float2 p0 = projectPosToScreenPos(float3{0.0, 0.0, 0.0}, viewProj, uniforms.width, uniforms.height);
            float2 p1 = projectPosToScreenPos(float3{1.0, 1.0, 0.0}, viewProj, uniforms.width, uniforms.height);

            float cellPixelSize = p1.x - p0.x;
            float2 objectCellSize = object.cellSize;

            float2 objectPixelSize = {
                objectCellSize.x * cellPixelSize,
                objectCellSize.y * cellPixelSize,
            };

            object.size = objectPixelSize;

            object.position.x = float(mouseX) - object.size.x * 0.5f;
            object.position.y = float(mouseY) - object.size.y * 0.5f;

            ObjectSelection::rasterize_blockwise(object, framebuffer);
        }



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
