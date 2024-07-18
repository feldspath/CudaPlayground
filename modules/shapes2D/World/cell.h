#pragma once

#include "builtin_types.h"

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

struct BaseCell {
    TileId tileId;
    int32_t landValue;
    int32_t entities[ENTITIES_PER_CELL];
};

struct HouseCell : public BaseCell {
    int32_t residentEntityIdx;

    static TileId type() { return HOUSE; }
};

struct RoadCell : public BaseCell {
    int32_t networkRepr;
    int32_t networkId;

    static TileId type() { return ROAD; }
};

struct WorkplaceCell : public BaseCell {
    int32_t workplaceCapacity;
    int32_t currentWorkerCount;

    bool isOpen() const { return currentWorkerCount > 0; }

    static TileId type() { return SHOP | FACTORY; }
};

struct ShopCell : public WorkplaceCell {
    int32_t woodCount;

    static TileId type() { return SHOP; }
};

struct FactoryCell : public WorkplaceCell {
    int32_t stockCount;

    static TileId type() { return FACTORY; }
};

union Cell {
    BaseCell cell;
    HouseCell house;
    FactoryCell factory;
    RoadCell road;
    ShopCell shop;
    WorkplaceCell workplace;
};