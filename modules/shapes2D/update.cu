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

void updateCell(Grid2D *grid2D, Network *network, uint32_t id) {
    uint32_t new_tile = uniforms.modeId;
    grid2D->cellIndices[id] = new_tile;
    network->parents[id] = id;
    switch (new_tile) {
    case ROAD:
        // check nearby tiles.
        int2 coords = grid2D->cellCoords(id);

        int right = grid2D->idFromCoords(coords.x + 1, coords.y);
        int left = grid2D->idFromCoords(coords.x - 1, coords.y);
        int up = grid2D->idFromCoords(coords.x, coords.y + 1);
        int down = grid2D->idFromCoords(coords.x, coords.y - 1);

        // if one tile is not grass, update the connected components
        if (right != -1 && grid2D->cellIndices[right] != GRASS) {
            network->parents[network->parents[right]] = id;
            // network->update(right, id);
        }
        if (left != -1 && grid2D->cellIndices[left] != GRASS) {
            network->parents[network->parents[left]] = id;
            // network->update(left, id);
        }
        if (up != -1 && grid2D->cellIndices[up] != GRASS) {
            network->parents[network->parents[up]] = id;
            // network->update(up, id);
        }
        if (down != -1 && grid2D->cellIndices[down] != GRASS) {
            network->parents[network->parents[down]] = id;
            // network->update(down, id);
        }
        break;
    case FACTORY:
    case HOUSE:
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

    if (grid.thread_rank() == 0) {
        bool mousePressed = uniforms.mouseButtons & 1;

        if (mousePressed) {
            float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
            float3 pos_W =
                unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
            int id = grid2D->cellAtPosition(float2{pos_W.x, pos_W.y});
            if (id == -1) {
                return;
            }
            if (grid2D->cellIndices[id] == GRASS) {
                updateCell(grid2D, network, id);
            }
        }
    }

    // Flatten network
    if (grid.block_rank() == 0) {
        block.sync();
        for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
            int cellId = block.thread_rank() + offset;
            if (cellId >= grid2D->count) {
                break;
            }

            network->parents[cellId] = network->cellRepr(cellId);
        }
        block.sync();
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, unsigned int *buffer, uint32_t numRows,
                                  uint32_t numCols, uint32_t *gridCells, uint32_t *networkParents) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, gridCells);

        Network *network = allocator->alloc<Network *>(sizeof(Network));
        network->parents = networkParents;

        updateGrid(grid2D, network);
    }
}
