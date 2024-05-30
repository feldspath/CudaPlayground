#pragma once

#include "builtin_types.h"

float ENTITY_RADIUS = 0.2f;

struct Entities {
    uint32_t count;
    float2 *positions;

    Entities(uint32_t entitiesCount, float2 *entities) {
        positions = entities;
        count = entitiesCount;
    }
};