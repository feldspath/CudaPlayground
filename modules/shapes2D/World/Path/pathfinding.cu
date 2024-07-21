#include "common/helper_math.h"
#include "common/utils.cuh"

#include "pathfinding.cuh"

struct PathfindingList {
    Pathfinding *data;
    uint32_t count;
};

// Locate all entities that required pathfinding
static PathfindingList locateLostEntities(Map *map, Entities *entities, Allocator *allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    // One flow field per lost entity
    PathfindingList pathfindingList;
    pathfindingList.data =
        allocator->alloc<Pathfinding *>(entities->getCount() * sizeof(Pathfinding));
    uint32_t &lostCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        lostCount = 0;
    }
    grid.sync();

    entities->processAllActive([&](int entityIndex) {
        Entity &entity = entities->get(entityIndex);
        if (entity.isLost()) {
            uint32_t targetId = entity.destination;
            int originId = map->cellAtPosition(entity.position);
            Pathfinding p;
            p.flowField = FlowField(map, targetId, originId);
            if (p.flowField.length() == 0) {
                printf("Error: entity %d cannot reach its destination\n", entityIndex);
                return;
            }
            p.entityIdx = entityIndex;
            p.origin = originId;
            p.target = targetId;
            uint32_t id = atomicAdd(&lostCount, 1);
            pathfindingList.data[id] = p;
        }
    });

    grid.sync();

    pathfindingList.count = lostCount;

    // Allocate the field for each pathfinding
    for (int i = 0; i < pathfindingList.count; ++i) {
        FieldCell *field = allocator->alloc<FieldCell *>(pathfindingList.data[i].flowField.size());
        if (grid.thread_rank() == 0) {
            pathfindingList.data[i].flowField.setBuffer(field);
        }
    }

    grid.sync();

    return pathfindingList;
}

void performPathFinding(Map *map, Entities *entities, Allocator *allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    // Each block handles a lost entity
    for_blockwise(pathfindingList.count, [&](int bufferIdx) {
        Pathfinding info = pathfindingList.data[bufferIdx];
        FlowField &flowField = info.flowField;

        // Init buffer
        processRangeBlock(info.flowField.length(), [&](int idx) { flowField.resetCell(idx); });

        if (block.thread_rank() == 0) {
            // Init neighbor tiles of target
            map->neighborCells(info.target).forEach([&flowField](int cellId) {
                int fieldId = flowField.fieldId(cellId);
                if (fieldId != -1) {
                    auto &cell = flowField.getFieldCell(fieldId);
                    cell.cellId = cellId;
                    cell.distance = 0;
                }
            });
        }

        block.sync();

        auto shortestPathLength = [](int cellId, Map *map, Pathfinding &info) {
            return map->neighborCells(cellId)
                .apply([&](int cellId) {
                    int fieldId = info.flowField.fieldId(cellId);
                    if (fieldId == -1) {
                        return uint32_t(Infinity);
                    }
                    return info.flowField.getFieldCell(fieldId).distance;
                })
                .min();
        };

        // Build flowfield
        int iterations = 0;
        // The first path found is the smallest in size but not necessarily the shortest, because
        // there are different distance values. This condition ensures that it continues enough
        // to ensure path optimality.
        while (2 * iterations <= shortestPathLength(info.origin, map, info)) {
            iterations++;
            // The field is split accross the threads of the block
            processRangeBlock(info.flowField.length(), [&](int currentFieldId) {
                FieldCell &fieldCell = info.flowField.getFieldCell(currentFieldId);
                if (fieldCell.cellId == -1) {
                    // cell has not been explored yet
                    return;
                }

                map->extendedNeighborCells(fieldCell.cellId)
                    .forEachDir([&](Direction dir, int neighborId) {
                        if (!flowField.isNeighborValid(fieldCell.cellId, dir)) {
                            return;
                        }

                        int neighborFieldId = flowField.fieldId(neighborId);
                        if (neighborFieldId == -1) {
                            return;
                        }

                        auto &neighborFieldCell = flowField.getFieldCell(neighborFieldId);
                        int32_t neighborDistance = 10 * length(directionFromEnum(dir));

                        // Atomically update neighbor tiles id if not set
                        int oldNeighborId = atomicCAS(&neighborFieldCell.cellId, -1, neighborId);

                        if (oldNeighborId != -1) {
                            // Set distance value
                            fieldCell.distance = min(fieldCell.distance,
                                                     neighborFieldCell.distance + neighborDistance);
                        }
                    });
            });

            // Ensure that the iteration is completed by all threads before the next one
            block.sync();
        }

        // Extract path
        if (block.thread_rank() == 0) {
            entities->get(info.entityIdx).path = info.flowField.extractPath(info.origin);
        }
    });
}

Path FlowField::extractPath(uint32_t originId) const {
    int current = originId;
    bool reached = false;
    Path path;

    while (!reached && path.length() < Path::MAX_LENGTH) {
        // Retrieve path
        uint32_t min = uint32_t(Infinity);
        Direction dir;
        int nextCell;

        // We assume that there is always a possible path to the target
        map->extendedNeighborCells(current).forEachDir([&](Direction neighborDir, int neighborId) {
            if (!isNeighborValid(current, neighborDir) || reached) {
                return;
            }

            if (neighborId == target) {
                reached = true;
                dir = neighborDir;
                min = 0;
                return;
            }

            int neighborFieldId = fieldId(neighborId);
            if (neighborFieldId == -1) {
                return;
            }

            uint32_t distance = getFieldCell(neighborFieldId).distance;
            if (distance < min) {
                min = distance;
                dir = neighborDir;
                nextCell = neighborId;
            }
        });

        if (min == uint32_t(Infinity)) {
            printf("Pathfinding error\n");
            return Path();
        }

        path.append(dir);
        current = nextCell;
    }
    return path;
}

int32_t FlowField::fieldId(uint32_t cellId) const {
    if (!isCellInField(cellId)) {
        return -1;
    }
    int offset = 0;
    int32_t cellNetwork = map->roadNetworkRepr(cellId);
    for (int i = 0; i < 4; i++) {
        int ni = networkIds.data[i];
        if (cellNetwork == ni) {
            break;
        }
        offset += map->roadNetworkId(ni);
    }
    return offset + map->roadNetworkId(cellId) % map->roadNetworkId(cellNetwork);
}

void FlowField::resetCell(uint32_t fieldId) {
    auto &fieldCell = getFieldCell(fieldId);
    fieldCell.cellId = -1;
    fieldCell.distance = uint32_t(Infinity);
}

bool FlowField::isNeighborValid(uint32_t cellId, Direction neighborDir) const {
    if (int(neighborDir) < 4) {
        return true;
    }

    int2 currentCellCoord = map->cellCoords(cellId);
    int2 dirCoords = coordFromEnum(neighborDir);
    int id1 = map->idFromCoords(currentCellCoord + int2{dirCoords.x, 0});
    int id2 = map->idFromCoords(currentCellCoord + int2{0, dirCoords.y});
    return !((id1 == -1 || map->getTileId(id1) != ROAD) &&
             (id2 == -1 || map->getTileId(id2) != ROAD));
}