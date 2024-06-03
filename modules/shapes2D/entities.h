#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "helper_math.h"

float ENTITY_RADIUS = 0.2f;
float ENTITY_SPEED = 3.0f;

uint32_t WORK_TIME_MS = 5000;
uint32_t REST_TIME_MS = 5000;

struct Entities {
    uint32_t *count;
    Entity *buffer;

    static const int houseOffset = 8;
    static const int factoryOffset = 12;
    static const int stateOffset = 16;
    static const int stateStartOffset = 20;

    Entities(void *entitiesBuffer) {
        count = (uint32_t *)entitiesBuffer;
        buffer = (Entity *)((char *)entitiesBuffer + 8);
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

    Entity *entityPtr(uint32_t entityId) { return &buffer[entityId]; }

    float2 &entityPosition(uint32_t entityId) { return entityPtr(entityId)->position; }

    uint32_t &entityHouse(uint32_t entityId) { return entityPtr(entityId)->houseId; }

    uint32_t &entityFactory(uint32_t entityId) { return entityPtr(entityId)->factoryId; }

    EntityState &entityState(uint32_t entityId) { return entityPtr(entityId)->state; }

    uint32_t &stateStart_ms(uint32_t entityId) { return entityPtr(entityId)->stateStart; }

    // Returns true if entity is within clampRadius distance of target.
    bool moveEntityTo(uint32_t entityId, float2 target, float clampRadius, float dt) {
        float2 &entityPos = entityPosition(entityId);
        float2 movementVector = normalize(target - entityPos);
        entityPos += dt * movementVector * ENTITY_SPEED;

        if (length(entityPos - target) < clampRadius) {
            entityPos = target;
            return true;
        }
        return false;
    }
};