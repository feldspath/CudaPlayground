#pragma once

#include "HostDeviceInterface.h"
#include "World/map.cuh"
#include "builtin_types.h"

class Entities {
private:
    uint32_t *count;
    uint32_t *holesCount;
    Entity *buffer;

public:
    Entities(void *entitiesBuffer) {
        count = (uint32_t *)entitiesBuffer;
        holesCount = (uint32_t *)entitiesBuffer + 1;
        buffer = (Entity *)((uint32_t *)entitiesBuffer + 2);
    }

    uint32_t getCount() { return *count; }
    uint32_t holes() { return *holesCount; }
    uint32_t newEntity(float2 position, uint32_t house, uint32_t workplace);
    Entity &get(uint32_t entityId) { return buffer[entityId]; }
    void remove(uint32_t entityId) {
        *((uint32_t *)(&buffer[MAX_ENTITY_COUNT]) - 1 - *holesCount) = entityId;
        (*holesCount)++;
        get(entityId).active = false;
    }

    template <typename Function> void processAllActive(Function &&function) {
        processRange(getCount(), [&](int idx) {
            if (get(idx).active) {
                function(idx);
            }
        });
    }

    template <typename Function> void processAll(Function &&function) {
        processRange(getCount(), [&](int idx) { function(idx); });
    }
};