#include "common/utils.cuh"

#include "common/helper_math.h"
#include "movement.cuh"

void fillCells(Map &map, Entities &entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Reset cell occupation
    map.processEachCell([&](MapId cell) {
        int32_t *cellEntities = map.get(cell).entities;
        for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
            cellEntities[i] = -1;
        }
    });

    grid.sync();

    entities.processAll([&](int entityIdx) {
        Entity &entity = entities.get(entityIdx);
        EntityState state = entity.state;
        if (state == Rest || state == WorkAtFactory) {
            return;
        }

        auto cell = map.cellAtPosition(entity.position);
        if (!cell.valid()) {
            printf("Error: entity %d is not on a valid tile, placing it back at home\n", entityIdx);
            entity.position = map.getCellPosition(entity.house);
            return;
        }
        int32_t *cellEntities = map.get(cell).entities;

        for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
            if (atomicCAS(&cellEntities[i], -1, entityIdx) == -1) {
                break;
            }
        }
    });

    grid.sync();
}

void moveEntities(Map &map, Entities &entities, Allocator allocator, float dt,
                  curandStateXORWOW_t &rng) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    // Count entities to move
    uint32_t &entitiesToMoveCount = *allocator.alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &bufferIdx = *allocator.alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        entitiesToMoveCount = 0;
        bufferIdx = 0;
    }

    grid.sync();

    entities.processAllActive([&](int entityIdx) {
        Entity &entity = entities.get(entityIdx);
        if (entity.path.isValid()) {
            atomicAdd(&entitiesToMoveCount, 1);
        }
    });

    grid.sync();

    if (entitiesToMoveCount == 0) {
        return;
    }

    // Allocate buffer and store entities to move
    uint32_t *entitiesToMove = allocator.alloc<uint32_t *>(sizeof(uint32_t) * entitiesToMoveCount);
    entities.processAllActive([&](int entityIdx) {
        Entity &entity = entities.get(entityIdx);
        if (entity.path.isValid()) {
            int idx = atomicAdd(&bufferIdx, 1);
            entitiesToMove[idx] = entityIdx;
        }
    });

    grid.sync();

    float pressure_normalization = 15.0f / (3.141592 * powf(KERNEL_RADIUS, 6.0f));

    // Update velocities
    processRange(entitiesToMoveCount, [&](int idx) {
        auto &entity = entities.get(entitiesToMove[idx]);

        float2 forces = {0.0f, 0.0f};

        auto cell = map.cellAtPosition(entity.position);

        // Compute repulsive force of other entities
        int2 coords = map.cellCoords(cell);
        for (int i = -1; i < 1; i++) {
            for (int j = -1; j < 1; j++) {
                auto neighborCell = map.cellFromCoords({coords.x + i, coords.y + j});
                if (!neighborCell.valid()) {
                    continue;
                }
                for (int k = 0; k < ENTITIES_PER_CELL; ++k) {
                    int otherIdx = map.get(neighborCell).entities[k];
                    if (otherIdx == -1) {
                        break;
                    }
                    if (otherIdx == entitiesToMove[idx]) {
                        continue;
                    }

                    Entity &other = entities.get(otherIdx);

                    float2 diffVector = entity.position - other.position;
                    float norm = length(diffVector);
                    if (norm < 1e-12) {
                        continue;
                    }
                    if (norm < KERNEL_RADIUS) {
                        float otherMult = 1.0f + 0.5f * (other.mult - 0.5f);
                        forces += diffVector / norm * powf((KERNEL_RADIUS - norm), 3.0) *
                                  REPULSIVE_STRENGTH * pressure_normalization * otherMult;
                    }
                }
            }
        }

        // Stirring force
        Direction nextDir = entity.path.nextDir();
        float2 dirVector = directionFromEnum(nextDir);

        float angle = (curand_uniform(&rng) - 0.5f) * 3.141592 / 4;
        float c = cos(angle);
        float s = sin(angle);
        dirVector = {dirVector.x * c - dirVector.y * s, dirVector.x * s + dirVector.y * c};

        forces += normalize(dirVector) * STIR_STRENGTH;
        forces -= DAMPING_STRENGTH * entity.velocity;

        entity.velocity += forces * dt;

        // Clamp velocity
        float velocityNorm = length(entity.velocity);
        float mult = 1.0f + 0.3f * (entity.mult - 0.5f);
        float maxSpeed = mult * ENTITY_SPEED;
        if (velocityNorm > maxSpeed) {
            entity.velocity = entity.velocity / velocityNorm * maxSpeed;
        }
    });

    grid.sync();

    // Update positions
    processRange(entitiesToMoveCount, [&](int idx) {
        uint32_t entityId = entitiesToMove[idx];
        auto &entity = entities.get(entityId);

        if (!entity.path.isValid()) {
            return;
        }

        auto previousCell = map.cellAtPosition(entity.position);
        float2 previousCellPosition = map.getCellPosition(previousCell);
        auto neighborCells = map.neighborCells(previousCell);
        entity.position += entity.velocity * dt;

        // check each side of the entity for wall collision
        neighborCells.forEachDir([&](Direction direction, MapId neighbor) {
            // no collision with roads, house, workplace, shops, and destination
            TileId tile = map.get(neighbor).tileId;
            if ((tile == ROAD && map.neighborNetworks(entity.destination)
                                     .contains(map.getTyped<RoadCell>(neighbor).networkRepr)) ||
                tile == SHOP || neighbor == entity.workplace || neighbor == entity.house ||
                neighbor == entity.destination) {
                return;
            }

            float2 vectorDir = directionFromEnum(direction);
            float2 posToPreviousCellCenter =
                entity.position + vectorDir * ENTITY_RADIUS - previousCellPosition;
            // Collision with wall, project back to boundary
            entity.position -=
                max((dot(posToPreviousCellCenter, vectorDir) - CELL_RADIUS), 0.0f) * vectorDir;
        });

        auto newCell = map.cellAtPosition(entity.position);
        if (newCell != previousCell) {
            int dir = -1;
            neighborCells.forEachDir([&](Direction direction, MapId cell) {
                if (cell == newCell) {
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