#pragma once
#include "builtin_types.h"

enum Direction { RIGHT = 0, LEFT = 1, UP = 2, DOWN = 3 };

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
    default:
        break;
    }
}

template <typename T> struct NeighborInfo {
    // Indices follow the Direction enum values
    // 0 -> right
    // 1 -> left
    // 2 -> up
    // 3 -> down
    T data[4];

    bool contains(T other) const {
        return oneTrue([&](T val) { return val == other; });
    }

    T getDir(Direction dir) const { return data[static_cast<int32_t>(dir)]; }

    NeighborInfo() {
        data[0] = -1;
        data[1] = -1;
        data[2] = -1;
        data[3] = -1;
    }

    // Apply f to every value in data that is not -1
    template <typename Function> NeighborInfo<T> apply(Function &&f) const {
        NeighborInfo<T> result;
        for (int i = 0; i < 4; ++i) {
            if (data[i] == -1) {
                continue;
            }
            result.data[i] = f(data[i]);
        }
        return result;
    }

    template <typename Function> void applyDir(Function &&f) const {
        NeighborInfo<T> result;
        for (int i = 0; i < 4; ++i) {
            if (data[i] == -1) {
                continue;
            }
            result.data[i] = f(Direction(i), data[i]);
        }
        return result;
    }

    // Run f to every value in data that is not -1
    template <typename Function> void forEach(Function &&f) const {
        for (int i = 0; i < 4; ++i) {
            if (data[i] == -1) {
                continue;
            }
            f(data[i]);
        }
    }

    template <typename Function> void forEachDir(Function &&f) const {
        for (int i = 0; i < 4; ++i) {
            if (data[i] == -1) {
                continue;
            }
            f(Direction(i), data[i]);
        }
    }

    template <typename Function> bool oneTrue(Function &&f) const {
        for (int i = 0; i < 4; ++i) {
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
