
#pragma once

#include "World/Path/path.h"
#include "World/time.h"
#include "builtin_types.h"

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
static int RENDERMODE_LANDVALUE = 2;

// DATA FORMATS

struct GameState {
    bool firstFrame;
    int previousMouseButtons;

    // Time
    uint64_t previousFrameTime_ns;
    uint32_t currentTime_ms;
    GameTime gameTime;

    // Game stuff
    unsigned int playerMoney;
    unsigned int population;
    int32_t buildingDisplay;

    static GameState *instance;

    // Timings
    float assignOneHouse_ms;
    float performPathFinding_ms;
    float fillCells_ms;
    float moveEntities_ms;
    float updateEntitiesState_ms;

    bool isPlacingBuilding;
    int buildingType;
};

// CELLS

enum TileId {
    UNKNOWN = 0,
    GRASS = 1,
    ROAD = 2,
    HOUSE = 4,
    FACTORY = 8,
    SHOP = 16,
};

inline TileId operator|(TileId a, TileId b) {
    return static_cast<TileId>(static_cast<int>(a) | static_cast<int>(b));
}

#define ENTITIES_PER_CELL 8

struct Cell {
    TileId tileId;
    int32_t landValue;
    int32_t entities[ENTITIES_PER_CELL];
    char additionalData[8];
    int32_t buildingID; // -1: empty
};

static int BYTES_PER_CELL = sizeof(Cell);

// ENTITIES

enum EntityState {
    Rest,
    GoToWork,
    Work,
    GoHome,
    Shop,
    GoShopping,
};

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
    bool creativeMode;
    float timeMultiplier;

    // Inputs
    double2 cursorPos;
    int mouseButtons;
    int modeId;
};

struct Construction{
    int32_t type;
    int tile_x;
    int tile_y;
};

struct ConstructionList{
    uint32_t numConstructions;
    Construction items[10'000];
};

struct Entity {
    // movement
    float2 position;
    float2 velocity;

    // state logic
    uint32_t houseId;
    uint32_t workplaceId;
    EntityState state;
    uint32_t money;
    uint32_t happiness;

    GameTime stateStart;
    // entity id of the current interaction. -1 if the entity is not engaged in any interaction.
    uint32_t interaction;

    // Path is a uint64_t
    Path path;
    int32_t destination;

    inline bool isLost() { return destination != -1 && !path.isValid(); }

    void resetStateStart() { stateStart = GameState::instance->gameTime; }

    void changeState(EntityState newState) {
        path.reset();
        resetStateStart();
        destination = -1;
        interaction = -1;
        state = newState;
    }
};

static int BYTES_PER_ENTITY = sizeof(Entity);


struct GameData{
    Uniforms uniforms;   // Args passed form host to device every frame
    GameState *state;          // Managed only by device
    unsigned int *buffer;      // Some buffer where we can allocate each frame
    uint32_t numRows;
    uint32_t numCols;
    char *cells;
    void *entitiesBuffer;
    uint32_t *img_ascii_16;
    uint32_t *img_spritesheet;
    uint32_t *img_spritesheet_buildings;
    ConstructionList* constructions;
};