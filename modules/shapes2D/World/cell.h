#pragma once

#include "builtin_types.h"

#include "config.h"
#include "direction.h"

#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#define ENTITIES_PER_CELL 8

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

struct ALIGN(8) MapId {
    int32_t chunkId = -1;
    int32_t cellId = -1;

    MapId() {}

    inline bool valid() const { return chunkId != -1 && cellId != -1; }
    void reset() {
        chunkId = -1;
        cellId = -1;
    }
    MapId(int32_t chunkId) {
        if (chunkId == -1) {
            MapId();
        } else {
            printf("Wrong use of MapId constructor");
        }
    }
    MapId(int32_t chunkId, int32_t cellId) : chunkId(chunkId), cellId(cellId) {}

    int64_t &as_int64() { return *(int64_t *)(&chunkId); }

    static MapId invalidId() { return {-1, -1}; }
};

typedef NeighborInfo<MapId, 4> MapNeighbors;

inline bool operator==(const MapId &lhs, const MapId &rhs) {
    return lhs.chunkId == rhs.chunkId && lhs.cellId == rhs.cellId;
}

struct BaseCell {
    TileId tileId;
    int32_t entities[ENTITIES_PER_CELL];
};

struct HouseCell : public BaseCell {
    int32_t residentCount;
    int32_t woodCount;
    int32_t level;

    HouseCell() {
        residentCount = 0;
        woodCount = 0;
        level = 0;
    }

    int upgradeCost() const { return HOUSE_BASE_UPGRADE_WOOD_COUNT << level; }

    void levelUp() {
        woodCount -= upgradeCost();
        level++;
    }

    int maxResidents() const { return 1 << level; }

    static TileId type() { return HOUSE; }
};

struct RoadCell : public BaseCell {
    MapId networkRepr;
    int chunkNetworkRepr;

    static TileId type() { return ROAD; }
};

struct WorkplaceCell : public BaseCell {
    int32_t workplaceCapacity;
    int32_t currentWorkerCount;

    WorkplaceCell(int32_t workplaceCapacity) {
        currentWorkerCount = 0;
        this->workplaceCapacity = workplaceCapacity;
    }

    bool isOpen() const { return currentWorkerCount > 0; }

    static TileId type() { return SHOP | FACTORY; }
};

struct ShopCell : public WorkplaceCell {
    int32_t woodCount;

    ShopCell() : WorkplaceCell(SHOP_WORK_CAPACITY) { woodCount = 0; }

    static TileId type() { return SHOP; }
};

struct FactoryCell : public WorkplaceCell {
    int32_t stockCount;
    int32_t level;

    FactoryCell() : WorkplaceCell(FACTORY_CAPACITY) {
        stockCount = 0;
        level = 0;
    }

    static TileId type() { return FACTORY; }
};

union Cell {
    BaseCell cell;
    HouseCell house;
    FactoryCell factory;
    RoadCell road;
    ShopCell shop;
    WorkplaceCell workplace{0};
};