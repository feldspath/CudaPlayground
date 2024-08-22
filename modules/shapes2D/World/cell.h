#pragma once

#include "builtin_types.h"

#include "config.h"

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
    int32_t networkRepr;

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