#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"

float ENTITY_RADIUS = 0.2f;

struct Entities {
    uint32_t *count;
    void *buffer;

    static const int houseOffset = 8;
    static const int entityOffset = 12;

    Entities(void *entitiesBuffer) {
        count = (uint32_t *)entitiesBuffer;
        buffer = &(((float2 *)entitiesBuffer)[1]);
    }

    uint32_t getCount() { return *count; }

    uint32_t newEntity(float2 position, uint32_t house, uint32_t factory) {
        int32_t id = getCount();
        *count += 1;

        entityPosition(id) = position;
        entityHouse(id) = house;
        entityFactory(id) = factory;

        return id;
    }

    float2 &entityPosition(uint32_t entityId) {
        return *(float2 *)((char *)buffer + entityId * BYTES_PER_ENTITY);
    }

    uint32_t &entityHouse(uint32_t entityId) {
        return *(uint32_t *)((char *)buffer + entityId * BYTES_PER_ENTITY + houseOffset);
    }

    uint32_t &entityFactory(uint32_t entityId) {
        return *(uint32_t *)((char *)buffer + entityId * BYTES_PER_ENTITY + entityOffset);
    }
};