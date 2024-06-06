#pragma once
#include "map.h"

struct FieldCell {
    uint32_t distance;
    int32_t cellId;
};

struct FlowField {
    Map *map;
    NeighborNetworks networkIds;

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

    void setBuffer(FieldCell *integrationField) {
        this->integrationField = integrationField;
        this->fieldSize = fieldSize;
    }

    uint32_t length() const { return fieldSize; }
    uint32_t size() const { return fieldSize * sizeof(FieldCell); }

    // The cell should be a road in one of the networks in networkIds
    bool isCellInField(uint32_t cellId) {
        return map->getTileId(cellId) == ROAD && networkIds.contains(map->roadNetworkRepr(cellId));
    }

    int32_t fieldId(uint32_t cellId) {
        if (!isCellInField(cellId)) {
            return -1;
        }
        int offset = 0;
        int32_t cellNetwork = map->roadNetworkRepr(cellId);
        for (int i = 0; i < 4; i++) {
            int ni = networkIds.data[i];
            if (cellNetwork == ni) {
                break;
            }
            offset += map->roadNetworkId(ni);
        }
        return offset + map->roadNetworkId(cellId) % map->roadNetworkId(cellNetwork);
    }

    void resetCell(uint32_t fieldId) {
        auto &fieldCell = getFieldCell(fieldId);
        fieldCell.cellId = -1;
        fieldCell.distance = uint32_t(Infinity);
    }

    FieldCell &getFieldCell(uint32_t fieldId) { return integrationField[fieldId]; }
};

struct Pathfinding {
    FlowField flowField;
    uint32_t entityIdx;
    uint32_t origin;
    uint32_t target;
};
