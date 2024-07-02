#pragma once

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "config.h"
#include "direction.h"

typedef NeighborInfo<int32_t> NeighborCells;
typedef NeighborInfo<int32_t> NeighborNetworks;

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

struct Map {
    Cell *cellsData;
    int rows;
    int cols;
    int count;

    Map(uint32_t numRows, uint32_t numCols, char *data) {
        cellsData = (Cell *)data;
        rows = numRows;
        cols = numCols;
        count = numRows * numCols;
    }

    float2 getCellPosition(int cellId) {
        int x = cellId % cols;
        int y = cellId / cols;

        float2 center = make_float2(x + CELL_RADIUS + CELL_PADDING, y + CELL_RADIUS + CELL_PADDING);
        return center;
    }

    int cellAtPosition(float2 position) const {
        int x = floor(position.x);
        int y = floor(position.y);

        if (x >= cols || x < 0 || y >= rows || y < 0) {
            return -1;
        }

        return y * cols + x;
    }

    int2 cellCoords(int cellId) { return int2{cellId % cols, cellId / cols}; }

    inline int idFromCoords(int x, int y) {
        if (x >= cols || x < 0 || y >= rows || y < 0) {
            return -1;
        }
        return y * cols + x;
    }
    int idFromCoords(int2 coords) { return idFromCoords(coords.x, coords.y); }

    TileId getTileId(int cellId) const { return cellsData[cellId].tileId; }
    void setTileId(int cellId, TileId tile) { cellsData[cellId].tileId = tile; }

    char *tileData(int cellId) { return cellsData[cellId].additionalData; }

    // ROAD DATA

    int32_t *roadTileData(int cellId) { return (int32_t *)(tileData(cellId)); }

    // We assume that the network is flattened
    int32_t &roadNetworkRepr(int cellId) { return roadTileData(cellId)[0]; }
    int32_t &roadNetworkId(int cellId) { return roadTileData(cellId)[1]; }

    // FACTORY DATA
    int32_t *factoryTileData(int cellId) { return (int32_t *)(tileData(cellId)); }

    // HOUSE DATA
    int32_t *houseTileData(int cellId) { return (int32_t *)(tileData(cellId)); }

    int32_t rentCost(int cellId) { return cellsData[cellId].landValue / 20 + 10; }

    // SHOP DATA
    int32_t *shopTileData(int cellId) { return (int32_t *)(tileData(cellId)); }

    int32_t &shopWorkCapacity(int cellId) { return shopTileData(cellId)[0]; }
    int32_t &shopCurrentWorkerCount(int cellId) { return shopTileData(cellId)[1]; }

    // Network logic
    NeighborNetworks neighborNetworks(int cellId) {
        auto neighbors = neighborCells(cellId);
        return neighbors.apply([&](int neighborCellId) {
            if (getTileId(neighborCellId) == ROAD) {
                return roadNetworkRepr(neighborCellId);
            } else {
                return -1;
            }
        });
    }

    NeighborNetworks sharedNetworks(NeighborNetworks nets1, NeighborNetworks nets2) {
        NeighborNetworks result;
        int count = 0;
        nets1.forEach([&](int network) {
            if (nets2.contains(network)) {
                result.data[count] = network;
                count++;
            }
        });
        return result;
    }

    NeighborNetworks sharedNetworks(int cellId1, int cellId2) {
        NeighborNetworks nets1 = neighborNetworks(cellId1);
        NeighborNetworks nets2 = neighborNetworks(cellId2);
        return sharedNetworks(nets1, nets2);
    }

    // Util functions
    template <typename Function> void processEachCell(TileId filter, Function &&f) {
        processRange(count, [&](int cellId) {
            if (getTileId(cellId) & filter) {
                f(cellId);
            }
        });
    }

    NeighborCells neighborCells(int cellId) {
        int2 coords = cellCoords(cellId);
        NeighborCells result;
        result.data[0] = idFromCoords(coords.x + 1, coords.y);
        result.data[1] = idFromCoords(coords.x - 1, coords.y);
        result.data[2] = idFromCoords(coords.x, coords.y + 1);
        result.data[3] = idFromCoords(coords.x, coords.y - 1);
        return result;
    }
};
