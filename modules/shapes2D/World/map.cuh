#pragma once

#include "common/utils.cuh"

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

class Map {
private:
    Cell *cellsData;
    int rows;
    int cols;
    int count;

public:
    inline int getCount() const { return count; }

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

    TileId getTileId(int cellId) const { return cellsData[cellId].cell.tileId; }
    void setTileId(int cellId, TileId tile) { cellsData[cellId].cell.tileId = tile; }

    template <typename T> T &getTyped(int cellId) { return *((T *)&cellsData[cellId]); }
    BaseCell &get(int cellId) { return cellsData[cellId].cell; }

    // ROAD DATA
    // We assume that the network is flattened
    int32_t &roadNetworkRepr(int cellId) { return getTyped<RoadCell>(cellId).networkRepr; }
    int32_t &roadNetworkId(int cellId) { return getTyped<RoadCell>(cellId).networkId; }

    // HOUSE DATA
    int32_t rentCost(int cellId) { return cellsData[cellId].cell.landValue / 20 + 10; }

    // WORKPLACE DATA
    bool isWorkplace(int cellId) const { return getTileId(cellId) & (SHOP | FACTORY); }
    int32_t &workplaceCapacity(int cellId) {
        return getTyped<WorkplaceCell>(cellId).workplaceCapacity;
    }

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
            // use 0 (UNKNOWN) to disable filtering
            if ((getTileId(cellId) & filter) || !filter) {
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
};
