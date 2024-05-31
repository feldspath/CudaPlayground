#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "helper_math.h"

float ENTITY_RADIUS = 0.2f;
float ENTITY_SPEED = 1.0f;

uint32_t WORK_TIME_MS = 5000;
uint32_t REST_TIME_MS = 5000;

enum EntityState {
    Rest,
    GoToWork,
    Work,
    GoHome,
};

struct Entities {
    uint32_t *count;
    void *buffer;

    static const int houseOffset = 8;
    static const int entityOffset = 12;
    static const int stateOffset = 16;
    static const int stateStartOffset = 20;

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
        entityState(id) = GoToWork;

        return id;
    }

    char *entityPtr(uint32_t entityId) { return (char *)buffer + entityId * BYTES_PER_ENTITY; }

    float2 &entityPosition(uint32_t entityId) { return *(float2 *)entityPtr(entityId); }

    uint32_t &entityHouse(uint32_t entityId) {
        return *(uint32_t *)(entityPtr(entityId) + houseOffset);
    }

    uint32_t &entityFactory(uint32_t entityId) {
        return *(uint32_t *)(entityPtr(entityId) + entityOffset);
    }

    EntityState &entityState(uint32_t entityId) {
        return *(EntityState *)(entityPtr(entityId) + stateOffset);
    }

    uint32_t &stateStart_ms(uint32_t entityId) {
        return *(uint32_t *)(entityPtr(entityId) + stateStartOffset);
    }

    // Returns true if entity is within clampRadius distance of target.
    bool moveEntityTo(uint32_t entityId, float2 target, float clampRadius) {
        float2 &entityPos = entityPosition(entityId);
        float2 movementVector = normalize(target - entityPos);
        entityPos += 0.05f * movementVector;

        if (length(entityPos - target) < clampRadius) {
            entityPos = target;
            return true;
        }
        return false;
    }
};