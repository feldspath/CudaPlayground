#pragma once

#include "builtin_types.h"

float CELL_RADIUS = 0.49f;
float CELL_PADDING = 0.01f;

enum TileId {
    GRASS = 0,
    ROAD = 1,
    HOUSE = 2,
    FACTORY = 3,
    UNKNOWN = -1,
};

struct Cell {
    int id;
    float2 center;
};

struct Grid2D {
    uint32_t *cellIndices;
    int rows;
    int cols;
    int count;

    Grid2D(uint32_t numRows, uint32_t numCols, uint32_t *gridCells) {
        cellIndices = gridCells;
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

    int cellAtPosition(float2 position) {
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
};
