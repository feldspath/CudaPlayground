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
            if (map->sharedNetworks(originId, targetId).data[0] == -1) {
                printf("Error: entity %d cannot reach its destination\n", entityIndex);
                return;
            }
            Pathfinding p;
            p.entityIdx = entityIndex;
            p.origin = originId;
            p.target = targetId;
            uint32_t id = atomicAdd(&lostCount, 1);
            pathfindingList.data[id] = p;
        }
    });

    grid.sync();

    pathfindingList.count = lostCount;

    grid.sync();

    return pathfindingList;
}

void performPathFinding(Map *map, Entities *entities, Allocator *allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    __shared__ FieldCell fieldBuffer[FlowField::size()];

    // Each block handles a lost entity
    for_blockwise(min(pathfindingList.count, 500), [&](int bufferIdx) {
        Pathfinding info = pathfindingList.data[bufferIdx];
        FlowField flowField(map, info.target, fieldBuffer);

        // Init buffer
        processRangeBlock(FlowField::size(), [&](int idx) { flowField.resetCell(idx); });

        block.sync();

        if (block.thread_rank() == 0) {
            // Init target tile
            flowField.getFieldCell(info.target).distance = 0;
        }

        block.sync();

        // Build flowfield
        int iterations = 0;
        // The first path found is the smallest in size but not necessarily the shortest, because
        // there are different distance values. This condition ensures that it continues enough
        // to ensure path optimality.
        while (10 * iterations <= flowField.getFieldCell(info.origin).distance) {
            iterations++;
            // The field is split accross the threads of the block
            processRangeBlock(FlowField::size(), [&](int currentCellId) {
                FieldCell &fieldCell = flowField.getFieldCell(currentCellId);

                map->extendedNeighborCells(currentCellId)
                    .forEachDir([&](Direction dir, int neighborId) {
                        if (!flowField.isNeighborValid(currentCellId, neighborId, dir)) {
                            return;
                        }
                        uint32_t neighborDistance = flowField.getFieldCell(neighborId).distance;
                        if (neighborDistance == uint32_t(Infinity)) {
                            return;
                        }
                        uint32_t newDistance = neighborDistance + (int(dir) < 4 ? 10u : 14u);
                        fieldCell.distance = min(fieldCell.distance, newDistance);
                    });
            });

            // Ensure that the iteration is completed by all threads before the next one
            block.sync();
        }

        // Extract path
        if (block.thread_rank() == 0) {
            entities->get(info.entityIdx).path = flowField.extractPath(info.origin);
        }
        block.sync();
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
            if (!isNeighborValid(current, neighborId, neighborDir) || reached) {
                return;
            }
            if (neighborId == targetId) {
                reached = true;
                dir = neighborDir;
                min = 0;
                return;
            }

            uint32_t distance = getFieldCell(neighborId).distance;
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

void FlowField::resetCell(uint32_t cellId) {
    auto &fieldCell = getFieldCell(cellId);
    fieldCell.distance = uint32_t(Infinity);
}

bool FlowField::isNeighborValid(uint32_t cellId, uint32_t neighborId, Direction neighborDir) const {
    if (neighborId != targetId && map->getTileId(neighborId) != ROAD) {
        return false;
    }

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
