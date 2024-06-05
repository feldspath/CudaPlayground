#pragma once

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

    bool contains(T other) {
        for (int i = 0; i < 4; ++i) {
            if (data[i] == other) {
                return true;
            }
        }
        return false;
    }

    T getDir(Direction dir) { return data[static_cast<int32_t>(dir)]; }

    NeighborInfo() {
        data[0] = -1;
        data[1] = -1;
        data[2] = -1;
        data[3] = -1;
    }
};