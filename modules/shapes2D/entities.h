#pragma once

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "helper_math.h"

float ENTITY_RADIUS = 0.2f;
float ENTITY_SPEED = 3.0f;

uint32_t WORK_TIME_MS = 5000;
uint32_t REST_TIME_MS = 5000;

int MAX_PATH_LENGTH = 29;

enum Direction { RIGHT = 0, LEFT = 1, UP = 2, DOWN = 3 };

float2 directionFromEnum(Direction dir) {
    switch (dir) {
    case RIGHT:
        return float2{1.0f, 0.0f};
    case LEFT:
        return float2{-1.0f, 0.0f};
    case UP:
        return float2{0.0f, 1.0f};
    case DOWN:
        return float2{0.0f, -1.0f};
    default:
        break;
    }
}

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
        resetPath(id);

        return id;
    }

    Entity *entityPtr(uint32_t entityId) { return &buffer[entityId]; }

    float2 &entityPosition(uint32_t entityId) { return entityPtr(entityId)->position; }

    uint32_t &entityHouse(uint32_t entityId) { return entityPtr(entityId)->houseId; }

    uint32_t &entityFactory(uint32_t entityId) { return entityPtr(entityId)->factoryId; }

    EntityState &entityState(uint32_t entityId) { return entityPtr(entityId)->state; }

    uint32_t &stateStart_ms(uint32_t entityId) { return entityPtr(entityId)->stateStart_ms; }

    // Path
    void resetPath(uint32_t entityId) { entityPtr(entityId)->path = 0; }

    bool isPathValid(uint32_t entityId) { return entityPtr(entityId)->path != 0; }

    uint32_t getPathLength(uint32_t entityId) {
        return (uint32_t)((entityPtr(entityId)->path >> 58ull) & 0b11111ull);
    }

    void setPathLength(uint32_t entityId, uint32_t newPathLength) {
        if (newPathLength > MAX_PATH_LENGTH) {
            return;
        }
        uint64_t &path = entityPtr(entityId)->path;

        // Set path bits to 0
        path = path & ~(uint64_t(0b11111ull) << 58ull);
        // Set path
        path = path | (uint64_t(newPathLength) << 58ull);
    }

    void setPathDir(uint32_t entityId, Direction dir, uint32_t dirId) {
        if (dirId > 29) {
            return;
        }
        uint64_t &path = entityPtr(entityId)->path;
        path = path & ~(0b11ull << uint64_t(2 * dirId));
        path = path | (uint64_t(dir) << uint64_t(2 * dirId));
    }

    Direction nextPathDirection(uint32_t entityId) {
        uint32_t pathLength = getPathLength(entityId);
        return (Direction)((entityPtr(entityId)->path >> uint64_t(2 * (pathLength - 1))) & 0b11ull);
    }

    void advancePath(uint32_t entityId) {
        int newPathLength = getPathLength(entityId) - 1;
        uint64_t path = entityPtr(entityId)->path;
        if (newPathLength == 0) {
            resetPath(entityId);
        } else {
            setPathLength(entityId, newPathLength);
        }
    }

    void pushBackPath(uint32_t entityId, Direction dir) {
        // Retrieve current path
        int currentLength = getPathLength(entityId);

        setPathLength(entityId, 0);
        uint64_t &path = entityPtr(entityId)->path;
        path = (path << 2ull);

        setPathDir(entityId, dir, 0);
        setPathLength(entityId, currentLength + 1);
    }

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

    // Move entity in specified direction. If a tile is crossed, return true.
    // direction is assumed to be normalized.
    bool moveEntityDir(uint32_t entityId, Direction dir, float dt, Grid2D *grid2D) {
        float2 direction = directionFromEnum(dir);
        float2 &entityPos = entityPosition(entityId);
        int previousCellId = grid2D->cellAtPosition(entityPos - direction * ENTITY_RADIUS * 1.2);

        entityPos += direction * ENTITY_SPEED * dt;

        return grid2D->cellAtPosition(entityPos - direction * ENTITY_RADIUS * 1.2) !=
               previousCellId;
    }
};