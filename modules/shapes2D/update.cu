#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
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

    if (grid.thread_rank() == 0) {
        grid2D->setTileId(id, new_tile);
        *grid2D->roadTileData(id) = id;
    }

    switch (new_tile) {
    case ROAD:
        if (grid.thread_rank() == 0) {
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
                if (cellId >= grid2D->count) {
                    break;
                }

                int newRepr = grid2D->roadNetworkRepr(grid2D->roadNetworkRepr(cellId));
                *grid2D->roadTileData(cellId) = newRepr;
            }
        }

        break;

    case FACTORY:
        if (grid.thread_rank() == 0) {
            // Set capacity
            *grid2D->factoryTileData(id) = FACTORY_CAPACITY;
        }
        break;

    case HOUSE:
        __shared__ uint64_t targetFactory;
        if (grid.thread_rank() == 0) {
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        if (grid.block_rank() == 0) {
            // Check nearby tiles.
            int4 tileComps = neighborNetworks(id, grid2D);
            int2 tileCoords = grid2D->cellCoords(id);

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

                // For each unique connected comp, look for factories connected to it.
                int4 factoryComps = neighborNetworks(factoryId, grid2D);
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        int f = ((int *)(&factoryComps))[i];
                        int t = ((int *)(&tileComps))[j];
                        if (f != -1 && f == t) {
                            // Keep the closest factory
                            int2 factoryCoords = grid2D->cellCoords(factoryId);
                            int2 diff = factoryCoords - tileCoords;
                            uint32_t distance = abs(diff.x) + abs(diff.y);
                            uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(factoryId);
                            atomicMin(&targetFactory, target);
                        }
                    }
                }
            }
        }
        if (grid.thread_rank() == 0) {
            int32_t *houseData = grid2D->houseTileData(id);
            if (targetFactory != uint64_t(Infinity) << 32ull) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                *houseData = factoryId;
                *grid2D->factoryTileData(factoryId) -= 1;
            } else {
                *houseData = -1;
            }
        }

        break;
    default:
        break;
    }
}

void updateGrid(Grid2D *grid2D) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (uniforms.cursorPos.x < 0 || uniforms.cursorPos.x >= uniforms.width ||
        uniforms.cursorPos.y < 0 || uniforms.cursorPos.y >= uniforms.height) {
        return;
    }

    __shared__ UpdateInfo updateInfo;

    if (grid.thread_rank() == 0) {
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
    }

    block.sync();

    if (updateInfo.update) {
        updateCell(grid2D, updateInfo);
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, unsigned int *buffer, uint32_t numRows,
                                  uint32_t numCols, char *cells) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        updateGrid(grid2D);
    }
}
