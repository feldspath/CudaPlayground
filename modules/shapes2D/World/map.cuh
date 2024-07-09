#pragma once

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
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

    int32_t &shopCurrentWorkerCount(int cellId) { return shopTileData(cellId)[1]; }

    // WORKPLACE DATA
    bool isWorkplace(int cellId) const { return getTileId(cellId) & (SHOP | FACTORY); }
    int32_t &workplaceCapacity(int cellId) { return *(int32_t *)(tileData(cellId)); }

    // Network logic
    Neighbors neighborNetworks(int cellId) {
        auto neighbors = neighborCells(cellId);
        return neighbors.apply([&](int neighborCellId) {
            if (getTileId(neighborCellId) == ROAD) {
                return roadNetworkRepr(neighborCellId);
            } else {
                return -1;
            }
        });
    }

    Neighbors sharedNetworks(Neighbors nets1, Neighbors nets2) {
        Neighbors result;
        int count = 0;
        nets1.forEach([&](int network) {
            if (nets2.contains(network)) {
                result.data[count] = network;
                count++;
            }
        });
        return result;
    }

    Neighbors sharedNetworks(int cellId1, int cellId2) {
        Neighbors nets1 = neighborNetworks(cellId1);
        Neighbors nets2 = neighborNetworks(cellId2);
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

    Neighbors neighborCells(int cellId) {
        int2 coords = cellCoords(cellId);
        Neighbors result;
        result.setDir([&](Direction dir) {
            int2 dirCoord = coordFromEnum(dir);
            return idFromCoords(coords.x + dirCoord.x, coords.y + dirCoord.y);
        });
        return result;
    }
};
