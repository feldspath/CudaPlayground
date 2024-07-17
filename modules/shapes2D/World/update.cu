#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/map.cuh"
#include "World/time.h"
#include "builtin_types.h"
#include "common/helper_math.h"
#include "common/matrix_math.h"
#include "pathfinding.cuh"

#include "./Rendering/ObjectSelection.cuh"

namespace cg = cooperative_groups;

GameData gamedata;
Allocator *allocator;
uint64_t nanotime_start;

curandStateXORWOW_t thread_random_state;

struct UpdateInfo {
    bool update;
    int tileToUpdate;
    TileId newTileId;
};

void updateCell(Map *map, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    TileId new_tile = updateInfo.newTileId;
    int id = updateInfo.tileToUpdate;

    grid.sync();
    if (grid.thread_rank() == 0) {
        if (gamedata.uniforms.creativeMode) {
            map->setTileId(id, new_tile);
        } else if (tileCost(new_tile) <= GameState::instance->playerMoney) {
            GameState::instance->playerMoney -= tileCost(new_tile);
            map->setTileId(id, new_tile);
        }
    }
    grid.sync();

    if (map->getTileId(id) != new_tile) {
        // tile was not updated
        return;
    }

    switch (new_tile) {
    case ROAD: {
        int *cumulNeighborNetworksSizes = allocator->alloc<int *>(sizeof(int) * 5);
        int *neighborNetworks = allocator->alloc<int *>(sizeof(int) * 4);

        if (grid.thread_rank() == 0) {
            // check nearby tiles.
            auto neighbors = map->neighborCells(id);
            int neighborNetworksSizes[4];

            for (int i = 0; i < 4; i++) {
                int nId = neighbors.data[i];
                // if one tile is not grass, update the connected components
                if (nId != -1 && map->getTileId(nId) == ROAD) {
                    int repr = map->roadNetworkRepr(nId);
                    // Skip the tile if it was already updated this frame
                    if (map->roadNetworkRepr(repr) == repr) {
                        neighborNetworksSizes[i] = map->roadNetworkId(repr);
                        neighborNetworks[i] = repr;
                        map->roadNetworkRepr(repr) = id;
                        continue;
                    }
                }
                neighborNetworksSizes[i] = 0;
                neighborNetworks[i] = -1;
            }

            cumulNeighborNetworksSizes[0] = 0;
            for (int i = 0; i < 4; i++) {
                cumulNeighborNetworksSizes[i + 1] =
                    cumulNeighborNetworksSizes[i] + neighborNetworksSizes[i];
            }

            // Init the new road tile
            map->roadNetworkRepr(id) = id;
            map->roadNetworkId(id) = cumulNeighborNetworksSizes[4] + 1;
        }

        grid.sync();

        // Flatten network
        map->processEachCell(ROAD, [&](int cellId) {
            int neighborId = -1;
            for (int i = 0; i < 4; ++i) {
                int network = neighborNetworks[i];
                if (map->roadNetworkRepr(cellId) == network || cellId == network) {
                    neighborId = i;
                    break;
                }
            }
            if (neighborId == -1) {
                return;
            }

            map->roadNetworkRepr(cellId) = id;
            map->roadNetworkId(cellId) += cumulNeighborNetworksSizes[neighborId];
        });
        break;
    }
    case FACTORY: {

        if (grid.thread_rank() == 0) {
            // Set capacity
            *map->factoryTileData(id) = FACTORY_CAPACITY;
        }

        processRange(map->count, [&](int cellId) {
            auto diff = map->cellCoords(cellId) - map->cellCoords(id);
            int dist = length(make_float2(diff));
            if (dist < 20) {
                map->cellsData[cellId].landValue =
                    max(map->cellsData[cellId].landValue - 20 + int(dist), 0);
            }
        });

        break;
    }
    case HOUSE:
        if (grid.thread_rank() == 0) {
            // Set house to unassigned
            *map->houseTileData(id) = -1;
        }
        break;
    case SHOP:
        if (grid.thread_rank() == 0) {
            // Set capacities
            map->shopWorkCapacity(id) = SHOP_WORK_CAPACITY;
            map->shopCurrentWorkerCount(id) = 0;
        }
        break;
    default:
        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }

template <typename Function> void printDuration(char *name, Function &&f) {
    if (!gamedata.uniforms.printTimings) {
        f();
        return;
    }

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    uint64_t t_start = nanotime();

    f();

    grid.sync();

    uint64_t t_end = nanotime();

    if (grid.thread_rank() == 0) {
        double nanos = double(t_end) - double(t_start);
        float millis = nanos / 1e6;
        printf("%s: %8.3f ms\n", name, millis);
    }
}

void handleInputs(Map map) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto& uniforms = gamedata.uniforms;

    // if (uniforms.cursorPos.x >= 0 && uniforms.cursorPos.x < uniforms.width &&
    //     uniforms.cursorPos.y >= 0 && uniforms.cursorPos.y < uniforms.height) {
    //     UpdateInfo updateInfo;

    //     bool mouseClicked =
    //         (uniforms.mouseButtons & 1) & ((~GameState::instance->previousMouseButtons) & 1);
    //     bool mousePressed = uniforms.mouseButtons & 1;
    //     updateInfo.update = false;

    //     float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
    //     float3 pos_W = unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
    //     int id = map->cellAtPosition(float2{pos_W.x, pos_W.y});

    //     if (mouseClicked) {
    //         if (grid.thread_rank() == 0) {
    //             if (map->getTileId(id) & (HOUSE | FACTORY | SHOP)) {
    //                 if (GameState::instance->buildingDisplay == id) {
    //                     GameState::instance->buildingDisplay = -1;
    //                 } else {
    //                     GameState::instance->buildingDisplay = id;
    //                 }
    //             }
    //         }
    //     } else if (mousePressed) {
    //         if (id != -1 && map->getTileId(id) == GRASS) {
    //             updateInfo.update = true;
    //             updateInfo.tileToUpdate = id;
    //             updateInfo.newTileId = (TileId)uniforms.modeId;
    //         }
    //     }
    //     if (updateInfo.update) {
    //         updateCell(map, updateInfo);
    //     }
    // }

    uint32_t numObjects = 0;
    auto cursorPos = uniforms.cursorPos;
    ObjectSelectionSprite* objects = ObjectSelection::createPanel(allocator, numObjects, cursorPos.x, uniforms.height - cursorPos.y, gamedata);

    grid.sync();

    bool wasPlacingBuilding = gamedata.state->isPlacingBuilding;
    bool nothingHovered = true;
    
    { // Handle Clicking the Object Selection Panel
        bool mouseClicked =
            (uniforms.mouseButtons & 1) & ((~GameState::instance->previousMouseButtons) & 1);

        if (mouseClicked) {
            for(int i = 0; i < numObjects; i++){
                ObjectSelectionSprite object = objects[i];

                if(object.hovered){
                    gamedata.state->isPlacingBuilding = true;
                    gamedata.state->buildingType = i;
                    nothingHovered = false;
                }
            }
        }
    }

    grid.sync();

    // Handle exiting placement mode
    if(grid.thread_rank() == 0)
    if(uniforms.mouseButtons == 2){
        gamedata.state->isPlacingBuilding = false;
    }

    grid.sync();

    // Handle placing an object
    if(grid.thread_rank() == 0)
    if(gamedata.state->isPlacingBuilding)
    {

        ObjectSelectionSprite object = objects[gamedata.state->buildingType];

        int pixelX = cursorPos.x;
        int pixelY = uniforms.height - cursorPos.y;

        float2 pFrag = make_float2(pixelX, pixelY);
        float3 pos_W = unproject(pFrag, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
        int btx = pos_W.x - object.cellSize.x * 0.5;
        int bty = pos_W.y  - object.cellSize.y * 0.5;


        if(uniforms.mouseButtons == 1 && GameState::instance->previousMouseButtons == 0){

            bool allCellsFree = true;
            for(int ox = 0; ox < object.cellSize.x; ox++)
            for(int oy = 0; oy < object.cellSize.y; oy++)
            {
                int tx = btx + ox;
                int ty = bty + oy;
                int cellID = map.cellAtPosition({tx, ty});

                if(map.cellsData[cellID].buildingID >= 0) allCellsFree = false;
            }

            if(allCellsFree && nothingHovered){
                for(int ox = 0; ox < object.cellSize.x; ox++)
                for(int oy = 0; oy < object.cellSize.y; oy++)
                {
                    int tx = btx + ox;
                    int ty = bty + oy;
                    int cellID = map.cellAtPosition({tx, ty});

                    Construction construction;
                    construction.type = gamedata.state->buildingType;
                    construction.tile_x = tx;
                    construction.tile_y = ty;

                    map.cellsData[cellID].buildingID = gamedata.state->buildingType;

                    if(ox == 0 && oy == 0){
                        uint32_t constructionIndex = atomicAdd(&gamedata.constructions->numConstructions, 1);
                        printf("constructionIndex %i \n", constructionIndex);
                        gamedata.constructions->items[constructionIndex] = construction;
                    }
                    
                }
            }
            

        }
    }

    // handle selecting entity target
    // if(grid.thread_rank() == 0)
    {
        int pixelX = cursorPos.x;
        int pixelY = uniforms.height - cursorPos.y;

        float2 pFrag = make_float2(pixelX, pixelY);
        float3 pos_W = unproject(pFrag, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);

        // printf("%f, %f \n", pos_W.x, pos_W.y);

        // if(uniforms.mouseButtons == 1 && GameState::instance->previousMouseButtons == 0){
        //     printf("%u \n", gamedata.state->numEntities);
        // }

        processRange(gamedata.state->numEntities, [&](int index){

            Entity& entity = gamedata.entities[index];

            int cell_x = pos_W.x;
            int cell_y = pos_W.y;
            uint32_t cellID = cell_x + cell_y * gamedata.numCols;

            entity.destination = cellID;

        });

    }


}



void updateGrid(Map map) {
    nanotime_start = nanotime();

    if (gamedata.uniforms.printTimings && cg::this_grid().thread_rank() == 0) {
        printf("================================\n");
    }

    printDuration("handleInputs            ", [&]() { handleInputs(map); });
    
    // if(cg::this_grid().thread_rank() == 0) printf("abc");
    // for(int i = 0; i < gamedata.state->numEntities; i++){
    //     Entity& entity = gamedata.entities[i];
        
    //     int2 start = {entity.position.x, entity.position.y};
    //     int2 end = {
    //         entity.destination % map.cols,
    //         entity.destination / map.cols,
    //     };
        
    //     findPath(start, end, map, gamedata);
    // }

    // printDuration("fillCells               ", [&]() { fillCells(map, entities); });
    // printDuration("assignOneHouse          ", [&]() { assignOneHouse(map, entities); });
    // printDuration("assignOneCustomerToShop ", [&]() { assignOneCustomerToShop(map, entities); });
    // printDuration("performPathFinding      ", [&]() { performPathFinding(map, entities, allocator); });
    // printDuration("moveEntities            ", [&]() { moveEntities(map, entities, allocator, GameState::instance->gameTime.getDt());});
    // printDuration("updateEntitiesState     ", [&]() { updateEntitiesState(map, entities); });
    // printDuration("entitiesInteractions    ", [&]() { entitiesInteractions(map, entities); });
    // printDuration("updateGameState         ", [&]() { updateGameState(); });
}



extern "C" __global__ void kernel_update(GameData _gamedata) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    gamedata = _gamedata;
    GameState::instance = gamedata.state;

    *gamedata.dbg_numLabels = 0;
    *gamedata.numLines = 0;

    Allocator _allocator(gamedata.buffer, 0);
    allocator = &_allocator;

    curand_init(grid.thread_rank() + gamedata.state->currentTime_ms, 0, 0,
                &thread_random_state);

    grid.sync();

    Map map = Map(gamedata.numRows, gamedata.numCols, gamedata.cells);

    updateGrid(map);
    
}

extern "C" __global__ 
void kernel_pathfinding(GameData gamedata) {


    // for(int i = 0; i < gamedata.state->numEntities; i++){
    //     Entity& entity = gamedata.entities[i];
        
    //     int2 start = {entity.position.x, entity.position.y};
    //     int2 end = {
    //         entity.destination % map.cols,
    //         entity.destination / map.cols,
    //     };
        
    //     findPath(start, end, map, gamedata);
    // }
    Map map = Map(gamedata.numRows, gamedata.numCols, gamedata.cells);

    int entityIndex = blockIdx.x;
    Entity& entity = gamedata.entities[entityIndex];

    int numCells = map.rows * map.cols;

    // Allocator allocator(gamedata.buffer, 1'000'000);
    // uint32_t* grid_costmap       = allocator.alloc<uint32_t*>(gamedata.state->numEntities * 4 * numCells);
	// uint32_t* grid_distancefield = allocator.alloc<uint32_t*>(gamedata.state->numEntities * 4 * numCells);
	// uint32_t* grid_flowfield     = allocator.alloc<uint32_t*>(gamedata.state->numEntities * 4 * numCells);

    // uint32_t* block_costmap       = grid_costmap       + entityIndex * 4 * numCells;
    // uint32_t* block_distancefield = grid_distancefield + entityIndex * 4 * numCells;
    // uint32_t* block_flowfield     = grid_flowfield     + entityIndex * 4 * numCells;

    // if(threadIdx.x == 0) printf("entityIndex: %d \n", entityIndex);

    // if(entityIndex != 0) return;

    int2 start = {entity.position.x, entity.position.y};
    int2 end = {
        entity.destination % map.cols,
        entity.destination / map.cols,
    };
    
    findPath(start, end, map, gamedata);
    // findPath(start, end, map, gamedata, block_costmap, block_distancefield, block_flowfield);


    // printf("test")
}


extern "C" __global__
void kernel_updateGameState(GameData gamedata) {
    auto grid = cg::this_grid();

    GameState::instance = gamedata.state;

    if (grid.thread_rank() == 0) {
        float dt = ((float)(nanotime_start - GameState::instance->previousFrameTime_ns)) / 1e9;
        GameState::instance->previousFrameTime_ns = nanotime_start;
        GameState::instance->currentTime_ms = currentTime_ms();
        GameState::instance->population = gamedata.state->numEntities;
        GameState::instance->previousMouseButtons = gamedata.uniforms.mouseButtons;

        if (GameState::instance->firstFrame) {
            GameState::instance->firstFrame = false;
            GameState::instance->gameTime.dt = 0.0f;
        } else {
            GameState::instance->gameTime.incrementRealTime(dt * gamedata.uniforms.timeMultiplier);
        }
    }
}