#pragma once

#include "HostDeviceInterface.h"
#include "World/map.cuh"
#include "builtin_types.h"

struct Entities {
    uint32_t *count;
    Entity *buffer;

    Entities(void *entitiesBuffer) {
        count = (uint32_t *)entitiesBuffer;
        buffer = (Entity *)((char *)entitiesBuffer + 8);
    }

    uint32_t getCount() { return *count; }
    uint32_t newEntity(float2 position, uint32_t house, uint32_t workplace);
    Entity &get(uint32_t entityId) { return buffer[entityId]; }
};