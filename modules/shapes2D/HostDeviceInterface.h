
#pragma once

#include "builtin_types.h"
#include "path.h"

struct mat4 {
    float4 rows[4];

    static mat4 identity() {
        mat4 id;

        id.rows[0] = {1.0f, 0.0f, 0.0f, 0.0f};
        id.rows[1] = {0.0f, 1.0f, 0.0f, 0.0f};
        id.rows[2] = {0.0f, 0.0f, 1.0f, 0.0f};
        id.rows[3] = {0.0f, 0.0f, 0.0f, 1.0f};

        return id;
    }

    static mat4 rotate(float angle, float3 axis) {
        // see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

        float cosa = cos(-angle);
        float sina = sin(-angle);

        float ux = axis.x;
        float uy = axis.y;
        float uz = axis.z;

        mat4 rot;
        rot.rows[0].x = cosa + ux * ux * (1.0f - cosa);
        rot.rows[0].y = ux * uy * (1.0f - cosa) - uz * sina;
        rot.rows[0].z = ux * uz * (1.0f - cosa) + uy * sina;
        rot.rows[0].w = 0.0f;

        rot.rows[1].x = uy * ux * (1.0f - cosa) + uz * sina;
        rot.rows[1].y = cosa + uy * uy * (1.0f - cosa);
        rot.rows[1].z = uy * uz * (1.0f - cosa) - ux * sina;
        rot.rows[1].w = 0.0f;

        rot.rows[2].x = uz * ux * (1.0f - cosa) - uy * sina;
        rot.rows[2].y = uz * uy * (1.0f - cosa) + ux * sina;
        rot.rows[2].z = cosa + uz * uz * (1.0f - cosa);
        rot.rows[2].w = 0.0f;

        rot.rows[3].x = 0.0f;
        rot.rows[3].y = 0.0f;
        rot.rows[3].z = 0.0f;
        rot.rows[3].w = 1.0f;

        return rot;
    }

    static mat4 translate(float x, float y, float z) {

        mat4 trans = mat4::identity();

        trans.rows[0].w = x;
        trans.rows[1].w = y;
        trans.rows[2].w = z;

        return trans;
    }

    static mat4 scale(float sx, float sy, float sz) {

        mat4 scaled = mat4::identity();

        scaled.rows[0].x = sx;
        scaled.rows[1].y = sy;
        scaled.rows[2].z = sz;

        return scaled;
    }

    mat4 transpose() {
        mat4 result;

        result.rows[0] = {rows[0].x, rows[1].x, rows[2].x, rows[3].x};
        result.rows[1] = {rows[0].y, rows[1].y, rows[2].y, rows[3].y};
        result.rows[2] = {rows[0].z, rows[1].z, rows[2].z, rows[3].z};
        result.rows[3] = {rows[0].w, rows[1].w, rows[2].w, rows[3].w};

        return result;
    }
};

static int RENDERMODE_DEFAULT = 0;
static int RENDERMODE_NETWORK = 1;

// DATA FORMATS

// CELLS

enum TileId {
    UNKNOWN = 0,
    GRASS = 1,
    ROAD = 2,
    HOUSE = 4,
    FACTORY = 8,
};

inline TileId operator|(TileId a, TileId b) {
    return static_cast<TileId>(static_cast<int>(a) | static_cast<int>(b));
}

#define ENTITIES_PER_CELL 8

struct Cell {
    TileId tileId;
    int32_t entities[ENTITIES_PER_CELL];
    char additionalData[8];
};

static int BYTES_PER_CELL = sizeof(Cell);

// ENTITIES

enum EntityState {
    Rest,
    GoToWork,
    Work,
    GoHome,
};

// alignment = 8 bytes. No padding
struct Entity {
    float2 position;
    float2 velocity;
    uint32_t houseId;
    uint32_t factoryId;
    EntityState state;
    uint32_t stateStart_ms;
    // Path is a uint64_t
    Path path;

    inline bool isLost() { return (state == GoHome || state == GoToWork) && !path.isValid(); }
};

static int BYTES_PER_ENTITY = sizeof(Entity);

struct Uniforms {
    float width;
    float height;
    float time;
    mat4 view;
    mat4 proj;
    mat4 invproj;
    mat4 invview;
    int renderMode;
    bool printTimings;

    // Inputs
    double2 cursorPos;
    int mouseButtons;
    int modeId;
};

// GAME RELATED

struct GameState {
    uint64_t previousFrameTime_ns;
    uint32_t currentTime_ms;
    float dt;

    float assignOneHouse_ms;
    float performPathFinding_ms;
    float fillCells_ms;
    float moveEntities_ms;
    float updateEntitiesState_ms;
};
