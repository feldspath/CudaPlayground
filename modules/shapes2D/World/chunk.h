#pragma once

#ifdef __CUDACC__
#include "common/utils.cuh"

#include "builtin_types.h"
#include "cell.h"
#include "common/helper_math.h"
#include "config.h"
#include "direction.h"
#endif

enum FlowfieldState {
    VALID = 0,
    INVALID = 1,
    // marked for computation this frame
    MARKED = 2,
};

struct Flowfield {
    FlowfieldState state;
    uint8_t directions[CHUNK_SIZE];
};

struct IntegrationField {
    uint32_t distances[CHUNK_SIZE];
    uint32_t iterations[CHUNK_SIZE];
    MapId target;
    bool ongoingComputation;
};

struct Chunk {
    Cell cells[CHUNK_SIZE];
    Flowfield cachedFlowfields[CHUNK_SIZE];
    int2 offset;

#ifdef __CUDACC__
    constexpr int size() { return CHUNK_SIZE; }

    void invalidateCachedFlowfields() {
        processEachCell(UNKNOWN, [&](int cellId) { cachedFlowfields[cellId].state = INVALID; });
    }

    template <typename Function> void processEachCell(TileId filter, Function &&f) {
        processRange(CHUNK_SIZE, [&](int cellId) {
            // use 0 (UNKNOWN) to disable filtering
            if (!filter || cells[cellId].cell.tileId & filter) {
                f(cellId);
            }
        });
    }

    template <typename Function> void processEachCell(Function &&f) {
        processRange(CHUNK_SIZE, [&](int cellId) { f(cellId); });
    }

    template <typename Function> void processEachCellBlock(TileId filter, Function &&f) {
        processRangeBlock(CHUNK_SIZE, [&](int cellId) {
            // use 0 (UNKNOWN) to disable filtering
            if (!filter || cells[cellId].cell.tileId & filter) {
                f(cellId);
            }
        });
    }

    template <typename Function> void processEachCellBlock(Function &&f) {
        processRangeBlock(CHUNK_SIZE, [&](int cellId) { f(cellId); });
    }

    float2 getCellPosition(int cellId) const {
        int x = cellId % CHUNK_X;
        int y = cellId / CHUNK_X;

        float2 center = 2.0f * CELL_RADIUS *
                        (float2{float(x) + 0.5f + float(offset.x) * CHUNK_X,
                                float(y) + 0.5f + float(offset.y) * CHUNK_Y});
        return center;
    }

    int cellAtPosition(float2 position) const {
        int x = floor(position.x / (CELL_RADIUS * 2.0f));
        int y = floor(position.y / (CELL_RADIUS * 2.0f));

        int localX = x - offset.x * CHUNK_X;
        int localY = y - offset.y * CHUNK_Y;

        if (!isCoordValid(localX, localY)) {
            return -1;
        }

        return localY * CHUNK_X + localX;
    }

    inline int2 cellCoords(int cellId) const {
        return int2{cellId % CHUNK_X, cellId / CHUNK_X} + offset * int2{CHUNK_X, CHUNK_Y};
    }

    inline int idFromCoords(int x, int y) const {
        int localX = x - offset.x * CHUNK_X;
        int localY = y - offset.y * CHUNK_Y;
        if (!isCoordValid(localX, localY)) {
            return -1;
        }

        return localY * CHUNK_X + localX;
    }

    inline int idFromCoords(int2 coords) const { return idFromCoords(coords.x, coords.y); }

    template <typename T> T &getTyped(int cellId) { return *((T *)&cells[cellId]); }
    template <typename T> const T &getTyped(int cellId) const { return *((T *)&cells[cellId]); }
    BaseCell &get(int cellId) { return cells[cellId].cell; }
    const BaseCell &get(int cellId) const { return cells[cellId].cell; }

    // WORKPLACE DATA
    bool isWorkplace(int cellId) const { return get(cellId).tileId & (SHOP | FACTORY); }

    // Util functions
    Neighbors neighborCells(int cellId) {
        int2 coords = cellCoords(cellId);
        Neighbors result;
        result.setDir([&](Direction dir) {
            int2 dirCoord = coordFromEnum(dir);
            return idFromCoords(coords.x + dirCoord.x, coords.y + dirCoord.y);
        });
        return result;
    }

    ExtendedNeighbors extendedNeighborCells(int cellId) {
        int2 coords = cellCoords(cellId);
        ExtendedNeighbors result;
        result.setDir([&](Direction dir) {
            int2 dirCoord = coordFromEnum(dir);
            return idFromCoords(coords.x + dirCoord.x, coords.y + dirCoord.y);
        });
        return result;
    }

    int32_t neighborCell(int cellId, Direction dir) {
        int2 coords = cellCoords(cellId);
        int2 dirCoord = coordFromEnum(dir);
        return idFromCoords(coords.x + dirCoord.x, coords.y + dirCoord.y);
    }

    Neighbors neighborNetworks(int32_t cell) {
        auto neighbors = neighborCells(cell);
        return neighbors.apply([&](int32_t neighborCell) {
            if (get(neighborCell).tileId == ROAD) {
                return getTyped<RoadCell>(neighborCell).chunkNetworkRepr;
            } else {
                return -1;
            }
        });
    }

    Neighbors sharedNetworks(Neighbors nets1, Neighbors nets2) {
        Neighbors result;
        int count = 0;
        nets1.forEach([&](int networkRepr) {
            if (nets2.contains(networkRepr)) {
                result.data[count] = networkRepr;
                count++;
            }
        });
        return result;
    }

    Neighbors sharedNetworks(int cell1, int cell2) {
        Neighbors nets1 = neighborNetworks(cell1);
        Neighbors nets2 = neighborNetworks(cell2);
        return sharedNetworks(nets1, nets2);
    }

    Neighbors neighborNetworkIds(int cell) {
        return neighborNetworks(cell).apply(
            [&](int networkRepr) { return getTyped<RoadCell>(networkRepr).networkId; });
    }

private:
    inline bool isCoordValid(int x, int y) const {
        return x >= 0 && x < CHUNK_X && y >= 0 && y < CHUNK_Y;
    }
#endif
};
