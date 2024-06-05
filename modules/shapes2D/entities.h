#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "helper_math.h"
#include "map.h"

struct Entities {
    uint32_t *count;
    Entity *buffer;

    Entities(void *entitiesBuffer) {
        count = (uint32_t *)entitiesBuffer;
        buffer = (Entity *)((char *)entitiesBuffer + 8);
    }

    uint32_t getCount() { return *count; }

    uint32_t newEntity(float2 position, uint32_t house, uint32_t factory) {
        int32_t id = getCount();
        *count += 1;

        Entity &entity = get(id);
        entity.position = position;
        entity.houseId = house;
        entity.factoryId = factory;
        entity.state = GoToWork;
        entity.path.reset();

        return id;
    }

    Entity &get(uint32_t entityId) { return buffer[entityId]; }

    // Returns true if entity is within clampRadius distance of target.
    bool moveEntityTo(uint32_t entityId, float2 target, float clampRadius, float dt) {
        float2 &entityPos = get(entityId).position;
        float2 movementVector = normalize(target - entityPos);
        entityPos += dt * movementVector * ENTITY_SPEED;

        if (length(entityPos - target) < clampRadius) {
            entityPos = target;
            return true;
        }
        return false;
    }

    // Move entity in specified direction. If a tile is crossed, return true.
    // direction is assumed to be normalized.
    bool moveEntityDir(uint32_t entityId, Direction dir, float dt, Map *map) {
        float2 direction = directionFromEnum(dir);
        float2 &entityPos = get(entityId).position;
        int previousCellId = map->cellAtPosition(entityPos - direction * ENTITY_RADIUS * 1.2);

        entityPos += direction * ENTITY_SPEED * dt;

        return map->cellAtPosition(entityPos - direction * ENTITY_RADIUS * 1.2) != previousCellId;
    }
};