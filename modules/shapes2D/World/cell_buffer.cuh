#pragma once

#include "cell.h"
#include "map.cuh"

class CellBuffer {
private:
    uint32_t *cellIds;
    uint32_t count;

public:
    uint32_t getCount() { return count; }
    template <typename Function> void fill(Map &map, Allocator *allocator, Function &&filter) {
        auto grid = cg::this_grid();

        // count occurences
        uint32_t &count = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
        if (grid.thread_rank() == 0) {
            count = 0;
        }
        grid.sync();
        map.processEachCell(UNKNOWN, [&](int cellId) {
            if (!filter(cellId)) {
                return;
            }
            atomicAdd(&count, 1);
        });
        grid.sync();
        this->count = count;

        if (count == 0) {
            return;
        }

        // allocate buffer
        cellIds = allocator->alloc<uint32_t *>(sizeof(uint32_t) * count);
        grid.sync();

        // fill buffer
        if (grid.thread_rank() == 0) {
            count = 0;
        }
        grid.sync();

        map.processEachCell(UNKNOWN, [&](int cellId) {
            if (!filter(cellId)) {
                return;
            }
            int idx = atomicAdd(&count, 1);
            cellIds[idx] = cellId;
        });
    }

    // each thread of the block handles a cell
    template <typename Function> void processEachCellBlock(Function &&function) {
        processRangeBlock(count, [&](int idx) { function(cellIds[idx]); });
    }

    // each thread of the grid handles a cell
    template <typename Function> void processEachCell(Function &&function) {
        processRange(count, [&](int idx) { function(cellIds[idx]); });
    }

    // each block of the grid handles a cell
    template <typename Function> void processEachCell_blockwise(Function &&function) {
        for_blockwise(count, [&](int idx) { function(cellIds[idx]); });
    }

    template <typename Function> CellBuffer subBuffer(Allocator *allocator, Function &&filter) {
        CellBuffer result;

        auto grid = cg::this_grid();

        // count occurences
        uint32_t &count = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
        if (grid.thread_rank() == 0) {
            count = 0;
        }
        grid.sync();
        processEachCell([&](int cellId) {
            if (!filter(cellId)) {
                return;
            }
            atomicAdd(&count, 1);
        });
        grid.sync();
        result.count = count;

        if (count == 0) {
            return result;
        }

        // allocate buffer
        result.cellIds = allocator->alloc<uint32_t *>(sizeof(uint32_t) * count);

        grid.sync();

        // fill buffer
        if (grid.thread_rank() == 0) {
            count = 0;
        }
        grid.sync();
        processEachCell([&](int cellId) {
            if (!filter(cellId)) {
                return;
            }
            int idx = atomicAdd(&count, 1);
            result.cellIds[idx] = cellId;
        });

        return result;
    }

    template <typename Function>
    int32_t findClosestOnNetworkBlockwise(Map &map, uint32_t cellId, Function &&filter) {
        auto block = cg::this_thread_block();

        __shared__ uint64_t targetCell;
        if (block.thread_rank() == 0) {
            targetCell = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        auto cellNets = map.neighborNetworks(cellId);
        int2 cellCoords = map.cellCoords(cellId);

        processEachCellBlock([&](int targetCellId) {
            auto targetNets = map.neighborNetworks(targetCellId);
            if (map.sharedNetworks(cellNets, targetNets).data[0] != -1 && filter(targetCellId)) {
                int2 targetCoords = map.cellCoords(targetCellId);
                int2 diff = targetCoords - cellCoords;
                uint32_t distance = abs(diff.x) + abs(diff.y);
                uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(targetCellId);
                // keep the closest
                atomicMin(&targetCell, target);
            }
        });

        block.sync();

        if (targetCell != uint64_t(Infinity) << 32ull) {
            return targetCell & 0xffffffffull;
        } else {
            return -1;
        }
    }

    template <typename Function> uint32_t findRandom(Function &&filter) {}
};