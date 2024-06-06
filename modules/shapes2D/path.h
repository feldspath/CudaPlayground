#pragma once

#include "direction.h"

struct Path {
    // first bit not used
    // 5 bits for path length
    // 2x29 bits for the directions. LSBs correspond to the last directions.
    uint64_t path;

    static const uint32_t MAX_LENGTH = 29;

    void reset();
    bool isValid();
    uint32_t length();
    void setLength(uint32_t newPathLength);
    void setDirId(Direction dir, uint32_t dirId);
    Direction nextDir();
    void pop();
    void append(Direction dir);
};
