#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"

float CELL_RADIUS = 0.45f;
float CELL_PADDING = 0.05f;

struct Cell {
    int id;
    float2 center;
};

struct Grid2D {
    char *cellsData;
    int rows;
    int cols;
    int count;

    Grid2D(uint32_t numRows, uint32_t numCols, char *data) {
        cellsData = data;
        rows = numRows;
        cols = numCols;
        count = numRows * numCols;
    }

    Cell getCell(int cellId) {
        int x = cellId % cols;
        int y = cellId / cols;

        Cell cell;
        cell.center = make_float2(x + CELL_RADIUS + CELL_PADDING, y + CELL_RADIUS + CELL_PADDING);
        cell.id = cellId;
        return cell;
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

    TileId getTileId(int cellId) const {
        return (TileId)(*(int32_t *)(cellsData + cellId * BYTES_PER_CELL));
    }
    void setTileId(int cellId, TileId tile) {
        *(int32_t *)(cellsData + cellId * BYTES_PER_CELL) = tile;
    }

    char *tileData(int cellId) { return cellsData + cellId * BYTES_PER_CELL + 4; }

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
