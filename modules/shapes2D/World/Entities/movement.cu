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
        int32_t *cellEntities = map.get(cell).entities;

        for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
            if (atomicCAS(&cellEntities[i], -1, entityIdx) == -1) {
                break;
            }
        }
    });

    grid.sync();
}

void moveEntities(Map &map, Entities &entities, Allocator *allocator, float dt) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    // Saving allocator offset
    int64_t allocatorOffset = allocator->offset;
    // Count entities to move
    uint32_t &entitiesToMoveCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &bufferIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        entitiesToMoveCount = 0;
        bufferIdx = 0;
    }

    grid.sync();

    // Count entities to move
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
    uint32_t *entitiesToMove = allocator->alloc<uint32_t *>(sizeof(uint32_t) * entitiesToMoveCount);
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

        auto cellId = map.cellAtPosition(entity.position);
        int2 coords = map.cellCoords(cellId);

        // Compute repulsive force of other entities
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                auto neighborCell = map.cellFromCoords({coords.x + i, coords.y + j});
                if (!neighborCell.valid()) {
                    continue;
                }

                for (int k = 0; k < ENTITIES_PER_CELL; ++k) {
                    int otherIdx = map.get(neighborCell).entities[k];
                    if (otherIdx == -1) {
                        break;
                    }
                    if (otherIdx == idx) {
                        continue;
                    }

                    Entity &other = entities.get(otherIdx);

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
        Direction nextDir = entity.path.nextDir();
        float2 dirVector = directionFromEnum(nextDir);
        forces += normalize(dirVector) * STIR_STRENGTH;
        forces -= DAMPING_STRENGTH * entity.velocity;

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
        auto &entity = entities.get(entityId);

        if (!entity.path.isValid()) {
            return;
        }

        auto previousCell = map.cellAtPosition(entity.position);
        Chunk &chunk = map.getChunk(previousCell.chunkId);
        float2 previousCellPosition = chunk.getCellPosition(previousCell.cellId);
        auto neighborCells = chunk.neighborCells(previousCell.cellId);
        entity.position += entity.velocity * dt;

        // check each side of the entity for wall collision
        neighborCells.forEachDir([&](Direction direction, uint32_t cellId) {
            // no collision with roads, house, workplace, shops, and destination
            TileId tile = chunk.get(cellId).tileId;
            if ((tile == ROAD && chunk.neighborNetworks(entity.destination)
                                     .contains(chunk.roadNetworkRepr(cellId))) ||
                tile == SHOP || cellId == entity.workplaceId || cellId == entity.houseId ||
                cellId == entity.destination) {
                return;
            }

            float2 vectorDir = directionFromEnum(direction);
            float2 posToPreviousCellCenter =
                entity.position + vectorDir * ENTITY_RADIUS - previousCellPosition;
            // Collision with wall, project back to boundary
            entity.position -=
                max((dot(posToPreviousCellCenter, vectorDir) - CELL_RADIUS), 0.0f) * vectorDir;
        });

        uint32_t newCellId = chunk.cellAtPosition(entity.position);
        if (newCellId != previousCell.cellId) {
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

    // Reset allocator offset
    allocator->offset = allocatorOffset;
}