#pragma once
#include "World/Entities/entities.cuh"
#include "World/map.cuh"

void performPathFinding(Map *map, Entities *entities, Allocator *allocator);

struct FieldCell {
    uint32_t distance;
};

struct FlowField {
    Map *map;
    FieldCell *integrationField;
    uint32_t targetId;

    FlowField() {}

    FlowField(Map *map, uint32_t target, FieldCell *integrationField) {
        this->map = map;
        this->targetId = target;
        this->integrationField = integrationField;
    }

    static constexpr uint32_t size() { return 64 * 64; }

    void resetCell(uint32_t cellId);

    inline FieldCell &getFieldCell(uint32_t cellId) { return integrationField[cellId]; }
    inline const FieldCell &getFieldCell(uint32_t cellId) const { return integrationField[cellId]; }

    Path extractPath(uint32_t originId) const;

    bool isNeighborValid(uint32_t fieldId, uint32_t neighborId, Direction neighborDir) const;
};

struct Pathfinding {
    uint32_t entityIdx;
    uint32_t origin;
    uint32_t target;
};
