#pragma once
#include <curand_kernel.h>

#include "World/Entities/entities.cuh"
#include "World/map.cuh"

void fillCells(Map &map, Entities &entities);
void moveEntities(Map &map, Entities &entities, Allocator allocator, float dt,
                  curandStateXORWOW_t &rng);