#include "common/utils.cuh"

#include "common/helper_math.h"
#include "movement.cuh"

void fillCells(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Reset cell occupation
    processRange(map->count, [&](int cellId) {
        int32_t *cellEntities = map->cellsData[cellId].entities;
        for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
            cellEntities[i] = -1;
        }
    });

    grid.sync();

    entities->processAll([&](int entityIdx) {
        Entity &entity = entities->get(entityIdx);
        EntityState state = entity.state;
        if (state == GoHome || state == GoToWork || state == GoShopping || state == Shop) {
            int cell = map->cellAtPosition(entity.position);
            int32_t *cellEntities = map->cellsData[cell].entities;

            for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
                if (atomicCAS(&cellEntities[i], -1, entityIdx) == -1) {
                    break;
                }
            }
        }
    });

    grid.sync();
}

void moveEntities(Map *map, Entities *entities, Allocator *allocator, float dt) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    // Count entities to move
    uint32_t &entitiesToMoveCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &bufferIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        entitiesToMoveCount = 0;
        bufferIdx = 0;
    }

    grid.sync();

    // Count entities to move
    entities->processAll([&](int entityIdx) {
        Entity &entity = entities->get(entityIdx);
        if (entity.path.isValid()) {
            atomicAdd(&entitiesToMoveCount, 1);
        }
    });

    grid.sync();

    if (entitiesToMoveCount == 0) {
        return;
    }

    // Allocate buffer and store entities to move
    uint32_t *entitiesToMove = allocator->alloc<uint32_t *>(sizeof(uint32_t) * entitiesToMoveCount);
    entities->processAll([&](int entityIdx) {
        Entity &entity = entities->get(entityIdx);
        if (entity.path.isValid()) {
            int idx = atomicAdd(&bufferIdx, 1);
            entitiesToMove[idx] = entityIdx;
        }
    });

    grid.sync();

    float pressure_normalization = 15.0f / (3.141592 * powf(KERNEL_RADIUS, 6.0f));

    // Update velocities
    processRange(entitiesToMoveCount, [&](int idx) {
        auto &entity = entities->get(entitiesToMove[idx]);

        float2 forces = {0.0f, 0.0f};

        int cellId = map->cellAtPosition(entity.position);
        int2 coords = map->cellCoords(cellId);

        // Compute repulsive force of other entities
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int neighborCellId = map->idFromCoords(coords.x + i, coords.y + j);
                if (neighborCellId == -1) {
                    continue;
                }

                for (int k = 0; k < ENTITIES_PER_CELL; ++k) {
                    int otherIdx = map->cellsData[neighborCellId].entities[k];
                    if (otherIdx == -1) {
                        break;
                    }
                    if (otherIdx == idx) {
                        continue;
                    }

                    Entity &other = entities->get(otherIdx);

                    float2 diffVector = entity.position - other.position;
                    float norm = length(diffVector);
                    if (norm < 1e-6) {
                        continue;
                    }
                    if (norm < KERNEL_RADIUS) {
                        forces += diffVector / norm * powf((KERNEL_RADIUS - norm), 3.0) *
                                  REPULSIVE_STRENGTH * pressure_normalization;
                    }
                }
            }
        }

        // Stirring force
        // TODO: improve by choosing a continuous location, not discrete
        float2 dirVector = directionFromEnum(entity.path.nextDir());
        float2 parVector = {dirVector.y, dirVector.x};
        float2 targetCenter =
            map->getCellPosition(map->cellAtPosition(entity.position)) + dirVector * CELL_RADIUS;

        float2 targetSide1 = targetCenter + parVector * CELL_RADIUS * 0.25;
        float2 targetSide2 = targetCenter - parVector * CELL_RADIUS * 0.25;

        float2 target = targetCenter;
        float targetDistance = targetDistance;

        float dist1 = length(targetSide1 - entity.position);
        if (dist1 < targetDistance) {
            target = targetSide1;
            targetDistance = dist1;
        }
        float dist2 = length(targetSide2 - entity.position);
        if (dist2 < targetDistance) {
            target = targetSide2;
            targetDistance = dist2;
        }

        forces += normalize(target - entity.position) * STIR_STRENGTH;

        forces -= 10.0 * entity.velocity;

        // float angle = generateRandomNumber() * 2.0 * 3.141592;
        // forces += length(forces) * 0.5 * float2{cos(angle), sin(angle)};

        entity.velocity += forces * dt;

        // Clamp velocity
        float velocityNorm = length(entity.velocity);
        if (velocityNorm > ENTITY_SPEED) {
            entity.velocity = entity.velocity / velocityNorm * ENTITY_SPEED;
        }
    });

    grid.sync();

    // Update positions
    processRange(entitiesToMoveCount, [&](int idx) {
        uint32_t entityId = entitiesToMove[idx];
        auto &entity = entities->get(entityId);

        if (!entity.path.isValid()) {
            return;
        }

        uint32_t previousCellId = map->cellAtPosition(entity.position);
        float2 previousCellPosition = map->getCellPosition(previousCellId);
        auto neighborCells = map->neighborCells(previousCellId);
        entity.position += entity.velocity * dt;

        // check each side of the entity for wall collision
        neighborCells.forEachDir([&](Direction direction, uint32_t cellId) {
            // no collision with roads, house, workplace and shops.
            TileId tile = map->getTileId(cellId);
            if ((tile == ROAD && map->neighborNetworks(entity.destination)
                                     .contains(map->roadNetworkRepr(cellId))) ||
                tile == SHOP || cellId == entity.workplaceId || cellId == entity.houseId) {
                return;
            }

            float2 vectorDir = directionFromEnum(direction);
            float2 posToPreviousCellCenter =
                entity.position + vectorDir * ENTITY_RADIUS - previousCellPosition;
            // Collision with wall, project back to boundary
            entity.position -=
                max((dot(posToPreviousCellCenter, vectorDir) - CELL_RADIUS), 0.0f) * vectorDir;
        });

        uint32_t newCellId = map->cellAtPosition(entity.position);
        if (newCellId != previousCellId) {
            int dir = -1;
            neighborCells.forEachDir([&](Direction direction, uint32_t cellId) {
                if (cellId == newCellId) {
                    dir = int(direction);
                }
            });

            if (dir != -1 && Direction(dir) == entity.path.nextDir()) {
                // Entity is on intended road
                entity.path.pop();
            } else {
                // if the entity went too far, or is not an intended road, invalidate path
                entity.path.reset();
            }
        }
    });
}