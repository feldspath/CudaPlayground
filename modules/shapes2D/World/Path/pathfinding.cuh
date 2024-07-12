#pragma once
#include "World/Entities/entities.cuh"
#include "World/map.cuh"

void performPathFinding(Map *map, Entities *entities, Allocator *allocator);

struct FieldCell {
    uint32_t distance;
    int32_t cellId;
};

struct FlowField {
    Map *map;
    Neighbors networkIds;

    FieldCell *integrationField;
    uint32_t fieldSize;
    uint32_t target;

    FlowField() {}

    FlowField(Map *map, uint32_t target, uint32_t origin) {
        this->map = map;
        this->target = target;
        networkIds = map->sharedNetworks(origin, target);

        fieldSize = 0;
        networkIds.forEach([&](int network) { fieldSize += map->roadNetworkId(network); });
    }

    void setBuffer(FieldCell *integrationField) { this->integrationField = integrationField; }

    inline uint32_t length() const { return fieldSize; }
    inline uint32_t size() const { return fieldSize * sizeof(FieldCell); }

    // The cell should be a road in one of the networks in networkIds
    inline bool isCellInField(uint32_t cellId) const {
        return map->getTileId(cellId) == ROAD && networkIds.contains(map->roadNetworkRepr(cellId));
    }

    int32_t fieldId(uint32_t cellId) const;

    void resetCell(uint32_t fieldId);

    inline FieldCell &getFieldCell(uint32_t fieldId) { return integrationField[fieldId]; }
    inline const FieldCell &getFieldCell(uint32_t fieldId) const {
        return integrationField[fieldId];
    }

    Path extractPath(uint32_t originId) const;

    bool isNeighborValid(uint32_t fieldId, Direction neighborDir) const;
};

struct Pathfinding {
    FlowField flowField;
    uint32_t entityIdx;
    uint32_t origin;
    uint32_t target;
};
