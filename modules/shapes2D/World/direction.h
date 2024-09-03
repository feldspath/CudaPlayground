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

inline float2 directionFromEnum(Direction dir) {
    static float2 dirs[] = {
        float2{1.0f, 0.0f}, float2{-1.0f, 0.0f},  float2{0.0f, 1.0f},  float2{0.0f, -1.0f},
        float2{1.0f, 1.0f}, float2{-1.0f, -1.0f}, float2{-1.0f, 1.0f}, float2{1.0f, -1.0f},
    };
    return dirs[int(dir)];
}

Direction enumFromCoord(int2 coord);

inline int2 coordFromEnum(Direction dir) {
    static int2 dirs[] = {
        int2{1, 0}, int2{-1, 0},  int2{0, 1},  int2{0, -1},
        int2{1, 1}, int2{-1, -1}, int2{-1, 1}, int2{1, -1},
    };
    return dirs[int(dir)];
}

template <typename T, size_t SIZE> struct NeighborInfo {
    // Indices follow the Direction enum values
    T data[SIZE];

    bool contains(T other) const {
        return oneTrue([&](T val) { return val == other; });
    }

    T getDir(Direction dir) const { return data[static_cast<int32_t>(dir)]; }

    NeighborInfo() { set(T(-1)); }

    static constexpr size_t size() { return SIZE; }

    // Apply f to every value in data that is not -1
    template <typename Function> NeighborInfo<T, SIZE> apply(Function &&f) const {
        NeighborInfo<T, SIZE> result;
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            result.data[i] = f(data[i]);
        }
        return result;
    }

    template <typename Function> NeighborInfo<T, SIZE> applyDir(Function &&f) const {
        NeighborInfo<T, SIZE> result;
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            result.data[i] = f(Direction(i), data[i]);
        }
        return result;
    }

    // Run f to every value in data that is not -1
    template <typename Function> void forEach(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            f(data[i]);
        }
    }

    template <typename Function> void forEachDir(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            f(Direction(i), data[i]);
        }
    }

    template <typename U, typename Function> NeighborInfo<U, SIZE> convert(Function &&f) const {
        NeighborInfo<U, SIZE> result;
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            result.data[i] = f(data[i]);
        }
        return result;
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

    T min() {
        T minValue = T(Infinity);
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
                continue;
            }
            if (data[i] < minValue) {
                minValue = data[i];
            }
        }
        return minValue;
    }

    template <typename Function> bool oneTrue(Function &&f) const {
        for (int i = 0; i < SIZE; ++i) {
            if (data[i] == T(-1)) {
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