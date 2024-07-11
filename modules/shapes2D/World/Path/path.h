#pragma once

#include "World/direction.h"

struct Path {
    // first bit not used
    // 5 bits for path length
    // 3x19 bits for the directions. LSBs correspond to the last directions.
    uint64_t path;

    static const uint32_t MAX_LENGTH = 19;
    static constexpr unsigned long long BITS_PER_DIR = 3ull;
    static constexpr unsigned long long DIR_MASK = 0b111ull;

    void reset();
    bool isValid();
    uint32_t length();
    void setLength(uint32_t newPathLength);
    void setDirId(Direction dir, uint32_t dirId);
    Direction nextDir();
    Direction getDir(int dirId);
    void pop();
    void append(Direction dir);

    Path() { reset(); }
};
