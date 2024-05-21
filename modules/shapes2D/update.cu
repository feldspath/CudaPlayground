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

void updateGrid(Grid2D *grid2D) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.thread_rank() == 0) {
        bool mousePressed = uniforms.mouseButtons & 1;

        // Reset grid
        // for (int i = 0; i < grid2D->rows * grid2D->cols; ++i) {
        //    grid2D->cellIndices[i] = 0;
        //}

        if (mousePressed) {
            float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
            float3 pos_W =
                unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
            int id = grid2D->cellIdFromPosition(float2{pos_W.x, pos_W.y});
            if (id != -1) {
                grid2D->cellIndices[id] = uniforms.modeId;
            }
        }
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, unsigned int *buffer, uint32_t numRows,
                                  uint32_t numCols, uint32_t *gridCells) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    grid.sync();

    {
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, gridCells);
        updateGrid(grid2D);
    }
}
