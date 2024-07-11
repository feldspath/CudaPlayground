#include "common/helper_math.h"
#include "common/utils.cuh"

#include "path.h"

void Path::reset() { path = 0; }

bool Path::isValid() { return path != 0; }

uint32_t Path::length() { return uint32_t((path >> 58ull) & 0b11111ull); }

void Path::setLength(uint32_t newPathLength) {
    if (newPathLength > MAX_LENGTH) {
        return;
    }

    // Set path length bits to 0
    path = path & ~(uint64_t(0b11111ull) << 58ull);
    // Set path length
    path = path | (uint64_t(newPathLength) << 58ull);
}

void Path::setDirId(Direction dir, uint32_t dirId) {
    if (dirId > MAX_LENGTH) {
        return;
    }
    path = path & ~(DIR_MASK << uint64_t(BITS_PER_DIR * dirId));
    path = path | (uint64_t(dir) << uint64_t(BITS_PER_DIR * dirId));
}

Direction Path::nextDir() { return getDir(0); }

Direction Path::getDir(int dirId) {
    uint32_t pathLength = this->length();
    return Direction(path >> uint64_t(BITS_PER_DIR * (pathLength - 1 - dirId)) & DIR_MASK);
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
    path = path << BITS_PER_DIR;
    setDirId(dir, 0);
    setLength(currentLength + 1);
}