#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "helper_math.h"
#include "matrix_math.h"
#include "network.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator *allocator;

struct UpdateInfo {
    bool update;
    int tileToUpdate;
    TileId newTileId;
};

int cellNetworkId(int cellId, Network *network, Grid2D *grid2D) {
    if (cellId != -1 && grid2D->getTileId(cellId) == ROAD) {
        return network->cellRepr(cellId);
    } else {
        return -1;
    }
}

int4 neighborNetworks(int cellId, Network *network, Grid2D *grid2D) {
    int2 coords = grid2D->cellCoords(cellId);
    int right = grid2D->idFromCoords(coords.x + 1, coords.y);
    int left = grid2D->idFromCoords(coords.x - 1, coords.y);
    int up = grid2D->idFromCoords(coords.x, coords.y + 1);
    int down = grid2D->idFromCoords(coords.x, coords.y - 1);

    int4 comps = {cellNetworkId(right, network, grid2D), cellNetworkId(left, network, grid2D),
                  cellNetworkId(up, network, grid2D), cellNetworkId(down, network, grid2D)};

    return comps;
}

void updateCell(Grid2D *grid2D, Network *network, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    TileId new_tile = updateInfo.newTileId;
    int id = updateInfo.tileToUpdate;

    if (grid.thread_rank() == 0) {
        grid2D->setTileId(id, new_tile);
        network->parents[id] = id;
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
                network->parents[network->parents[right]] = id;
                // network->update(right, id);
            }
            if (left != -1 && grid2D->getTileId(left) == ROAD) {
                network->parents[network->parents[left]] = id;
                // network->update(left, id);
            }
            if (up != -1 && grid2D->getTileId(up) == ROAD) {
                network->parents[network->parents[up]] = id;
                // network->update(up, id);
            }
            if (down != -1 && grid2D->getTileId(down) == ROAD) {
                network->parents[network->parents[down]] = id;
                // network->update(down, id);
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

                network->parents[cellId] = network->cellRepr(cellId);
            }
        }

        break;

    case FACTORY:
        if (grid.thread_rank() == 0) {
            // Set worker count to 0.
            char *data = grid2D->tileData(id);
            *(uint32_t *)data = FACTORY_CAPACITY;
        }
        break;

    case HOUSE:
        __shared__ uint64_t targetFactory;
        if (grid.thread_rank() == 0) {
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        if (grid.block_rank() == 0) {
            // Check nearby tiles.
            int4 tileComps = neighborNetworks(id, network, grid2D);
            int2 tileCoords = grid2D->cellCoords(id);

            for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
                int factoryId = block.thread_rank() + offset;
                if (factoryId >= grid2D->count) {
                    break;
                }

                if (grid2D->getTileId(factoryId) != FACTORY) {
                    continue;
                }
                if (*(uint32_t *)(grid2D->tileData(factoryId)) == 0) {
                    continue;
                }

                // For each unique connected comp, look for factories connected to it.
                int4 factoryComps = neighborNetworks(factoryId, network, grid2D);
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        int f = ((int *)(&factoryComps))[i];
                        int t = ((int *)(&tileComps))[j];
                        if (f != -1 && f == t) {
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
            char *data = grid2D->tileData(id);
            if (targetFactory != uint64_t(Infinity) << 32ull) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                *(int32_t *)data = factoryId;
                *(uint32_t *)(grid2D->tileData(factoryId)) -= 1;
            } else {
                *(int32_t *)data = -1;
            }
        }

        // Choose the closest factory that has space.
        break;
    default:
        break;
    }
}

void updateGrid(Grid2D *grid2D, Network *network) {

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
        updateCell(grid2D, network, updateInfo);
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, unsigned int *buffer, uint32_t numRows,
                                  uint32_t numCols, char *cells, uint32_t *networkParents) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        Network *network = allocator->alloc<Network *>(sizeof(Network));
        network->parents = networkParents;

        updateGrid(grid2D, network);
    }
}
