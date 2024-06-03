#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "entities.h"
#include "helper_math.h"
#include "matrix_math.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
GameState *gameState;
Allocator *allocator;
uint64_t nanotime_start;

struct UpdateInfo {
    bool update;
    int tileToUpdate;
    TileId newTileId;
};

int cellNetworkId(int cellId, Grid2D *grid2D) {
    if (cellId != -1 && grid2D->getTileId(cellId) == ROAD) {
        return grid2D->roadNetworkRepr(cellId);
    } else {
        return -1;
    }
}

int4 neighborCells(int cellId, Grid2D *grid2D) {
    int2 coords = grid2D->cellCoords(cellId);
    int right = grid2D->idFromCoords(coords.x + 1, coords.y);
    int left = grid2D->idFromCoords(coords.x - 1, coords.y);
    int up = grid2D->idFromCoords(coords.x, coords.y + 1);
    int down = grid2D->idFromCoords(coords.x, coords.y - 1);

    return int4{right, left, up, down};
}

int4 neighborNetworks(int cellId, Grid2D *grid2D) {
    int4 neighbors = neighborCells(cellId, grid2D);
    int4 comps = {cellNetworkId(neighbors.x, grid2D), cellNetworkId(neighbors.y, grid2D),
                  cellNetworkId(neighbors.z, grid2D), cellNetworkId(neighbors.w, grid2D)};

    return comps;
}

void updateCell(Grid2D *grid2D, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    TileId new_tile = updateInfo.newTileId;
    int id = updateInfo.tileToUpdate;

    grid.sync();
    if (grid.thread_rank() == 0) {
        grid2D->setTileId(id, new_tile);
    }
    grid.sync();

    switch (new_tile) {
    case ROAD: {
        // TODO: update using the entire grid
        if (grid.block_rank() == 0) {

            __shared__ int cumulNeighborNetworksSizes[5];
            __shared__ int neighborNetworksReprs[4];

            if (block.thread_rank() == 0) {

                // check nearby tiles.
                int4 neighbors = neighborCells(id, grid2D);
                int neighborNetworksSizes[4];

                for (int i = 0; i < 4; i++) {
                    int nId = ((int *)(&neighbors))[i];
                    // if one tile is not grass, update the connected components
                    if (nId != -1 && grid2D->getTileId(nId) == ROAD) {
                        int repr = grid2D->roadNetworkRepr(nId);
                        // Skip the tile if it was already updated this frame
                        if (grid2D->roadNetworkRepr(repr) == repr) {
                            neighborNetworksSizes[i] = grid2D->roadNetworkId(repr);
                            neighborNetworksReprs[i] = repr;
                            grid2D->roadNetworkRepr(repr) = id;
                            continue;
                        }
                    }
                    neighborNetworksSizes[i] = 0;
                    neighborNetworksReprs[i] = -1;
                }

                cumulNeighborNetworksSizes[0] = 0;
                for (int i = 0; i < 4; i++) {
                    cumulNeighborNetworksSizes[i + 1] =
                        cumulNeighborNetworksSizes[i] + neighborNetworksSizes[i];
                }

                // Init the new road tile
                grid2D->roadNetworkRepr(id) = id;
                grid2D->roadNetworkId(id) = cumulNeighborNetworksSizes[4] + 1;
            }

            block.sync();

            // Flatten network
            for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
                int cellId = block.thread_rank() + offset;
                if (cellId >= grid2D->count || grid2D->getTileId(cellId) != ROAD) {
                    continue;
                }

                int neighborId = -1;
                for (int i = 0; i < 4; ++i) {
                    if (grid2D->roadNetworkRepr(cellId) == neighborNetworksReprs[i] ||
                        cellId == neighborNetworksReprs[i]) {
                        neighborId = i;
                        break;
                    }
                }
                if (neighborId == -1) {
                    continue;
                }

                grid2D->roadNetworkRepr(cellId) = id;
                grid2D->roadNetworkId(cellId) += cumulNeighborNetworksSizes[neighborId];
            }
        }
        break;
    }
    case FACTORY:
        if (grid.thread_rank() == 0) {
            // Set capacity
            *grid2D->factoryTileData(id) = FACTORY_CAPACITY;
        }
        break;

    case HOUSE:
        if (grid.thread_rank() == 0) {
            // Set house to unassigned
            *grid2D->houseTileData(id) = -1;
        }
        break;
    default:
        break;
    }
}

void assignHouseToFactory(Grid2D *grid2D, Entities *entities, int32_t houseId, int32_t factoryId) {
    int32_t newEntity = entities->newEntity(grid2D->getCell(houseId).center, houseId, factoryId);
    int32_t *houseData = grid2D->houseTileData(houseId);
    *houseData = newEntity;

    *grid2D->factoryTileData(factoryId) -= 1;
}

void assignOneHouse(Grid2D *grid2D, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    if (grid.thread_rank() == 0) {
        assigned = 0;
    }
    grid.sync();

    __shared__ uint64_t targetFactory;
    for (int offset = 0; offset < grid2D->count; offset += grid.num_blocks()) {
        int currentTile = offset + grid.block_rank();
        if (currentTile > grid2D->count) {
            break;
        }

        // Skip if tile is not a house or already assigned
        if (grid2D->getTileId(currentTile) != HOUSE || *grid2D->houseTileData(currentTile) != -1) {
            continue;
        }

        // Get neighbor networks
        int4 houseNets = neighborNetworks(currentTile, grid2D);
        int2 tileCoords = grid2D->cellCoords(currentTile);

        if (block.thread_rank() == 0) {
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        // Check all tiles for factories
        for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
            int factoryId = block.thread_rank() + offset;
            if (factoryId >= grid2D->count) {
                break;
            }

            // Look for factories ...
            if (grid2D->getTileId(factoryId) != FACTORY) {
                continue;
            }
            // ... with some capacity
            if (*grid2D->factoryTileData(factoryId) == 0) {
                continue;
            }

            // Get the networks the factory is connected to
            int4 factoryNets = neighborNetworks(factoryId, grid2D);
            for (int i = 0; i < 16; i++) {
                int f = ((int *)(&factoryNets))[i % 4];
                int h = ((int *)(&houseNets))[i / 4];
                if (f != -1 && f == h) {
                    // This factory shares the same networ
                    int2 factoryCoords = grid2D->cellCoords(factoryId);
                    int2 diff = factoryCoords - tileCoords;
                    uint32_t distance = abs(diff.x) + abs(diff.y);
                    uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(factoryId);
                    // keep the closest factory
                    atomicMin(&targetFactory, target);
                    break;
                }
            }
        }

        block.sync();

        if (block.thread_rank() == 0) {
            int32_t *houseData = grid2D->houseTileData(currentTile);
            if (targetFactory != uint64_t(Infinity) << 32ull && !atomicAdd(&assigned, 1)) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                assignHouseToFactory(grid2D, entities, currentTile, factoryId);
            } else {
                *houseData = -1;
            }
        }

        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }
float frameTime() { return ((float)(nanotime_start - gameState->previousFrameTime_ns)) / 1e9; }

void updateEntities(Grid2D *grid2D, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Each thread handles an entity
    for (int offset = 0; offset < entities->getCount(); offset += grid.num_threads()) {
        int entityIndex = offset + grid.thread_rank();
        if (entityIndex >= entities->getCount()) {
            break;
        }

        switch (entities->entityState(entityIndex)) {
        case GoToWork:
            uint32_t factoryId = entities->entityFactory(entityIndex);
            float2 factoryPos = grid2D->getCell(factoryId).center;

            if (entities->moveEntityTo(entityIndex, factoryPos, CELL_RADIUS * 0.5f, frameTime())) {
                entities->entityState(entityIndex) = Work;
                entities->stateStart_ms(entityIndex) = currentTime_ms();
            }
            break;
        case Work:
            if (currentTime_ms() - entities->stateStart_ms(entityIndex) >= WORK_TIME_MS) {
                entities->entityState(entityIndex) = GoHome;
            }
            break;
        case GoHome:
            uint32_t houseId = entities->entityHouse(entityIndex);
            float2 housePos = grid2D->getCell(houseId).center;

            if (entities->moveEntityTo(entityIndex, housePos, CELL_RADIUS * 0.5f, frameTime())) {
                entities->entityState(entityIndex) = Rest;
                entities->stateStart_ms(entityIndex) = currentTime_ms();
            }
            break;
        case Rest:
            if (currentTime_ms() - entities->stateStart_ms(entityIndex) >= REST_TIME_MS) {
                entities->entityState(entityIndex) = GoToWork;
            }
        default:
            break;
        }
    }
}

void updateGameState() { gameState->previousFrameTime_ns = nanotime_start; }

void updateGrid(Grid2D *grid2D, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

    if (uniforms.cursorPos.x >= 0 && uniforms.cursorPos.x < uniforms.width &&
        uniforms.cursorPos.y >= 0 && uniforms.cursorPos.y < uniforms.height) {
        UpdateInfo updateInfo;

        bool mousePressed = uniforms.mouseButtons & 1;
        updateInfo.update = false;

        if (mousePressed) {
            float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
            float3 pos_W =
                unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
            int id = grid2D->cellAtPosition(float2{pos_W.x, pos_W.y});

            if (id != -1 && grid2D->getTileId(id) == GRASS) {
                updateInfo.update = true;
                updateInfo.tileToUpdate = id;
                updateInfo.newTileId = (TileId)uniforms.modeId;
            }
        }

        if (updateInfo.update) {
            updateCell(grid2D, updateInfo);
        }
    }

    assignOneHouse(grid2D, entities);

    updateEntities(grid2D, entities);

    if (grid.thread_rank() == 0) {
        updateGameState();
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, GameState *_gameState,
                                  unsigned int *buffer, uint32_t numRows, uint32_t numCols,
                                  char *cells, void *entitiesBuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    gameState = _gameState;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(entitiesBuffer);

        updateGrid(grid2D, entities);
    }
}
