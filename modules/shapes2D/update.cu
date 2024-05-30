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
Allocator *allocator;

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

int4 neighborNetworks(int cellId, Grid2D *grid2D) {
    int2 coords = grid2D->cellCoords(cellId);
    int right = grid2D->idFromCoords(coords.x + 1, coords.y);
    int left = grid2D->idFromCoords(coords.x - 1, coords.y);
    int up = grid2D->idFromCoords(coords.x, coords.y + 1);
    int down = grid2D->idFromCoords(coords.x, coords.y - 1);

    int4 comps = {cellNetworkId(right, grid2D), cellNetworkId(left, grid2D),
                  cellNetworkId(up, grid2D), cellNetworkId(down, grid2D)};

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
    case ROAD:
        if (grid.thread_rank() == 0) {
            *grid2D->roadTileData(id) = id;

            // check nearby tiles.
            int2 coords = grid2D->cellCoords(id);

            int right = grid2D->idFromCoords(coords.x + 1, coords.y);
            int left = grid2D->idFromCoords(coords.x - 1, coords.y);
            int up = grid2D->idFromCoords(coords.x, coords.y + 1);
            int down = grid2D->idFromCoords(coords.x, coords.y - 1);

            // if one tile is not grass, update the connected components
            if (right != -1 && grid2D->getTileId(right) == ROAD) {
                grid2D->updateNetworkRepr(right, id);
            }
            if (left != -1 && grid2D->getTileId(left) == ROAD) {
                grid2D->updateNetworkRepr(left, id);
            }
            if (up != -1 && grid2D->getTileId(up) == ROAD) {
                grid2D->updateNetworkRepr(up, id);
            }
            if (down != -1 && grid2D->getTileId(down) == ROAD) {
                grid2D->updateNetworkRepr(down, id);
            }
        }

        block.sync();

        // Flatten network
        if (grid.block_rank() == 0) {
            for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
                int cellId = block.thread_rank() + offset;
                if (cellId >= grid2D->count || grid2D->getTileId(cellId) != ROAD) {
                    continue;
                }

                int newRepr = grid2D->roadNetworkRepr(grid2D->roadNetworkRepr(cellId));
                *grid2D->roadTileData(cellId) = newRepr;
            }
        }

        block.sync();
        break;

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

void updateGrid(Grid2D *grid2D, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    assignOneHouse(grid2D, entities);

    if (uniforms.cursorPos.x < 0 || uniforms.cursorPos.x >= uniforms.width ||
        uniforms.cursorPos.y < 0 || uniforms.cursorPos.y >= uniforms.height) {
        return;
    }

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

extern "C" __global__ void update(const Uniforms _uniforms, unsigned int *buffer, uint32_t numRows,
                                  uint32_t numCols, char *cells, void *entitiesBuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(entitiesBuffer);

        updateGrid(grid2D, entities);
    }
}
