#pragma once

#include "cell.h"

template <typename T> class CellBuffer {
private:
    T *content;
    uint32_t count;

public:
    CellBuffer() : content(nullptr), count(0) {}
    CellBuffer(T *content, uint32_t count) : content(content), count(count) {}

    uint32_t getCount() { return count; }

    // each thread of the block handles a cell
    template <typename Function> void processEachCellBlock(Function &&function) {
        processRangeBlock(count, [&](int idx) { function(content[idx]); });
    }

    // each thread of the grid handles a cell
    template <typename Function> void processEachCell(Function &&function) {
        processRange(count, [&](int idx) { function(content[idx]); });
    }

    // each block of the grid handles a cell
    template <typename Function> void processEachCell_blockwise(Function &&function) {
        for_blockwise(count, [&](int idx) { function(content[idx]); });
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
        processEachCell([&](T value) {
            if (!filter(value)) {
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
        result.content = allocator->alloc<T *>(sizeof(T) * count);

        grid.sync();

        // fill buffer
        if (grid.thread_rank() == 0) {
            count = 0;
        }
        grid.sync();
        processEachCell([&](T value) {
            if (!filter(value)) {
                return;
            }
            int idx = atomicAdd(&count, 1);
            result.content[idx] = value;
        });

        return result;
    }

    template <typename Function> uint32_t findRandom(Function &&filter) {}
};