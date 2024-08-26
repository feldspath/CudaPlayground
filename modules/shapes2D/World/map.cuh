#pragma once

#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cell_buffer.cuh"
#include "config.h"
#include "direction.h"

static unsigned int tileCost(TileId tile) {
    switch (tile) {
    case UNKNOWN:
        return 0;
    case GRASS:
        return 0;
    case ROAD:
        return ROAD_COST;
    case HOUSE:
        return HOUSE_COST;
    case FACTORY:
        return FACTORY_COST;
    default:
        return 0;
    }
}

class Map {
private:
    Chunk *chunks;
    int rows;
    int cols;
    int count;

public:
    CellBuffer<MapId> shops;
    CellBuffer<MapId> houses;
    CellBuffer<MapId> factories;
    CellBuffer<MapId> workplaces;

    inline int getCount() const { return count; }

    Map(uint32_t numRows, uint32_t numCols, Chunk *chunks)
        : chunks(chunks), rows(numRows), cols(numCols), count(numCols * numRows) {}

    Chunk &getChunk(int chunkId) { return chunks[chunkId]; }
    const Chunk &getChunk(int chunkId) const { return chunks[chunkId]; }

    template <typename T> T &getTyped(MapId cell) {
        return getChunk(cell.chunkId).getTyped<T>(cell.cellId);
    }
    template <typename T> const T &getTyped(MapId cell) const {
        return getChunk(cell.chunkId).getTyped<T>(cell.cellId);
    }
    BaseCell &get(MapId cell) { return getChunk(cell.chunkId).get(cell.cellId); }
    const BaseCell &get(MapId cell) const { return getChunk(cell.chunkId).get(cell.cellId); }

    int chunkIdFromWorldPos(float2 pos) const {
        int row = floor(pos.x / (CHUNK_X * 2.0f * CELL_RADIUS));
        int col = floor(pos.y / (CHUNK_Y * 2.0f * CELL_RADIUS));

        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return -1;
        }

        return row * cols + col;
    }

    MapId cellAtPosition(float2 pos) const {
        int chunkId = chunkIdFromWorldPos(pos);
        if (chunkId == -1) {
            return {-1, -1};
        }

        int cellId = getChunk(chunkId).cellAtPosition(pos);

        return {chunkId, cellId};
    }

    int2 cellCoords(MapId cell) const { return getChunk(cell.chunkId).cellCoords(cell.cellId); }
    MapId cellFromCoords(int2 coords) const {
        int row = coords.x / CHUNK_X;
        int col = coords.y / CHUNK_Y;
        int chunkId = row * cols + col;
        int cellId = getChunk(chunkId).idFromCoords(coords);
        return {chunkId, cellId};
    }

    template <typename Function> void processEachCell(Function &&f) {
        // Each block handles a chunk
        for_blockwise(count, [&](int chunkId) {
            // Each thread in the block handles a cell
            getChunk(chunkId).processEachCellBlock([&](int cellId) { f({chunkId, cellId}); });
        });
    }

    // Buffers
    // TODO: cells in order of chunkId
    template <typename Function>
    CellBuffer<MapId> selectCells(Allocator *allocator, Function &&filter) {
        auto grid = cg::this_grid();

        // count occurences
        uint32_t &cellCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
        if (grid.thread_rank() == 0) {
            cellCount = 0;
        }
        grid.sync();

        processEachCell([&](MapId cell) {
            if (!filter(cell)) {
                return;
            }
            atomicAdd(&cellCount, 1);
        });

        grid.sync();

        if (cellCount == 0) {
            return CellBuffer<MapId>();
        }

        // allocate buffer
        MapId *cellIds = allocator->alloc<MapId *>(sizeof(MapId) * cellCount);
        grid.sync();

        // fill buffer
        if (grid.thread_rank() == 0) {
            cellCount = 0;
        }
        grid.sync();

        processEachCell([&](MapId cell) {
            if (!filter(cell)) {
                return;
            }
            int idx = atomicAdd(&cellCount, 1);
            cellIds[idx] = cell;
        });

        grid.sync();

        return CellBuffer(cellIds, cellCount);
    }

    template <typename Function>
    int32_t findClosestOnNetworkBlockwise(CellBuffer<MapId> &buffer, MapId cell,
                                          Function &&filter) {
        auto block = cg::this_thread_block();

        __shared__ uint64_t targetCell;
        if (block.thread_rank() == 0) {
            targetCell = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        auto &chunk = getChunk(cell.chunkId);
        auto cellNets = chunk.neighborNetworks(cell.cellId);
        int2 cellCoords = chunk.cellCoords(cell.cellId);

        buffer.processEachCellBlock([&](MapId other) {
            auto targetNets = chunk.neighborNetworks(other.cellId);
            if (chunk.sharedNetworks(cellNets, targetNets).data[0] != -1 && filter(other)) {
                int2 targetCoords = chunk.cellCoords(other.cellId);
                int2 diff = targetCoords - cellCoords;
                uint32_t distance = abs(diff.x) + abs(diff.y);
                uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(other.cellId);
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
};
