#include "./../common/utils.cuh"
#include "path.h"

void Path::reset() { path = 0; }

bool Path::isValid() { return path != 0; }

uint32_t Path::length() { return uint32_t((path >> 58ull) & 0b11111ull); }

void Path::setLength(uint32_t newPathLength) {
    if (newPathLength > MAX_LENGTH) {
        return;
    }

    // Set path bits to 0
    path = path & ~(uint64_t(0b11111ull) << 58ull);
    // Set path
    path = path | (uint64_t(newPathLength) << 58ull);
}

void Path::setDirId(Direction dir, uint32_t dirId) {
    if (dirId > MAX_LENGTH) {
        return;
    }
    path = path & ~(0b11ull << uint64_t(2 * dirId));
    path = path | (uint64_t(dir) << uint64_t(2 * dirId));
}

Direction Path::nextDir() {
    uint32_t pathLength = length();
    return Direction(path >> uint64_t(2 * (pathLength - 1)) & 0b11ull);
}

void Path::pop() {
    int newPathLength = length() - 1;
    if (newPathLength == 0) {
        reset();
    } else {
        setLength(newPathLength);
    }
}

void Path::append(Direction dir) {
    int currentLength = length();
    setLength(0);
    path = path << 2ull;
    setDirId(dir, 0);
    setLength(currentLength + 1);
}