#pragma once
#include "builtin_types.h"

enum Direction {
    RIGHT = 0,
    LEFT = 1,
    UP = 2,
    DOWN = 3,
    DIAG_UR = 4,
    DIAG_DL = 5,
    DIAG_UL = 6,
    DIAG_DR = 7,
    COUNT = 8,
};

static float2 directionFromEnum(Direction dir) {
    switch (dir) {
    case RIGHT:
        return float2{1.0f, 0.0f};
    case LEFT:
        return float2{-1.0f, 0.0f};
    case UP:
        return float2{0.0f, 1.0f};
    case DOWN:
        return float2{0.0f, -1.0f};
    case DIAG_UR:
        return float2{1.0f, 1.0f};
    case DIAG_DL:
        return float2{-1.0f, -1.0f};
    case DIAG_UL:
        return float2{-1.0f, 1.0f};
    case DIAG_DR:
        return float2{1.0f, -1.0f};
    default:
        break;
    }
}

static Direction enumFromCoord(int2 coord) {
    int x = coord.x;
    int y = coord.y;
    int sum = x + y;
    int lsb = y < 0 || sum < 0;
    int osb = x == 0 || sum == 0;
    int msb = x != 0 && y != 0;

    return Direction(msb << 2 | osb << 1 | lsb);
}

static int2 coordFromEnum(Direction dir) {
    switch (dir) {
    case RIGHT:
        return int2{1, 0};
    case LEFT:
        return int2{-1, 0};
    case UP:
        return int2{0, 1};
    case DOWN:
        return int2{0, -1};
    case DIAG_UR:
        return int2{1, 1};
    case DIAG_DL:
        return int2{-1, -1};
    case DIAG_UL:
        return int2{-1, 1};
    case DIAG_DR:
        return int2{1, -1};
    default:
        break;
    }
}

template <typename T, size_t SIZE> struct NeighborInfo {
    // Indices follow the Direction enum values
    T data[SIZE];

    bool contains(T other) const {
        return oneTrue([&](T val) { return val == other; });
    }

    T getDir(Direction dir) const { return data[static_cast<int32_t>(dir)]; }

    NeighborInfo() { set(-1); }

    static constexpr size_t size() { return SIZE; }

    // Apply f to every value in data that is not -1
    template <typename Function> NeighborInfo<T, SIZE> apply(Function &&f) const {
        NeighborInfo<T, SIZE> result;
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == -1) {
                continue;
            }
            result.data[i] = f(data[i]);
        }
        return result;
    }

    template <typename Function> NeighborInfo<T, SIZE> applyDir(Function &&f) const {
        NeighborInfo<T, SIZE> result;
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == -1) {
                continue;
            }
            result.data[i] = f(Direction(i), data[i]);
        }
        return result;
    }

    // Run f to every value in data that is not -1
    template <typename Function> void forEach(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == -1) {
                continue;
            }
            f(data[i]);
        }
    }

    template <typename Function> void forEachDir(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == -1) {
                continue;
            }
            f(Direction(i), data[i]);
        }
    }

    template <typename Function> void setDir(Function &&f) {
        for (int i = 0; i < SIZE; ++i) {
            data[i] = f(Direction(i));
        }
    }

    void set(T val) {
        for (int i = 0; i < SIZE; ++i) {
            data[i] = val;
        }
    }

    template <typename Function> bool oneTrue(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == -1) {
                continue;
            }
            if (f(data[i])) {
                return true;
            }
        }
        return false;
    }
};

typedef NeighborInfo<int32_t, 4> Neighbors;
typedef NeighborInfo<int32_t, 8> ExtendedNeighbors;