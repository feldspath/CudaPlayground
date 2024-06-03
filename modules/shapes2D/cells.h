#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"

float CELL_RADIUS = 0.45f;
float CELL_PADDING = 0.05f;

struct Grid2D {
    Cell *cellsData;
    int rows;
    int cols;
    int count;

    Grid2D(uint32_t numRows, uint32_t numCols, char *data) {
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

    int idFromCoords(int x, int y) {
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
};
