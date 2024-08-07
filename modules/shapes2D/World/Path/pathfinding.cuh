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

    FlowField(Map *map, uint32_t target) {
        this->map = map;
        this->targetId = target;
    }

    void setBuffer(FieldCell *integrationField) { this->integrationField = integrationField; }

    constexpr uint32_t length() const { return 64 * 64; }
    constexpr uint32_t size() const { return length() * sizeof(FieldCell); }

    void resetCell(uint32_t cellId);

    inline FieldCell &getFieldCell(uint32_t cellId) { return integrationField[cellId]; }
    inline const FieldCell &getFieldCell(uint32_t cellId) const { return integrationField[cellId]; }

    Path extractPath(uint32_t originId) const;

    bool isNeighborValid(uint32_t fieldId, Direction neighborDir) const;
};

struct Pathfinding {
    FlowField flowField;
    uint32_t entityIdx;
    uint32_t origin;
    uint32_t target;
};
