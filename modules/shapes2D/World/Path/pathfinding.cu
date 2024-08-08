#include "common/helper_math.h"
#include "common/utils.cuh"

#include "pathfinding.cuh"

// Locate all entities that required pathfinding
PathfindingList PathfindingManager::locateLostEntities(Map &map, Entities &entities,
                                                       Allocator &allocator) const {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    PathfindingList pathfindingList;
    pathfindingList.data =
        allocator.alloc<PathfindingInfo *>(entities.getCount() * sizeof(PathfindingInfo));
    uint32_t &lostCount = *allocator.alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        lostCount = 0;
    }
    grid.sync();

    entities.processAllActive([&](int entityIndex) {
        Entity &entity = entities.get(entityIndex);
        if (entity.isLost()) {
            uint32_t targetId = entity.destination;
            int originId = map.cellAtPosition(entity.position);
            if (map.sharedNetworks(originId, targetId).data[0] == -1) {
                printf("Error: entity %d cannot reach its destination\n", entityIndex);
                return;
            }
            PathfindingInfo info;
            info.entityIdx = entityIndex;
            info.origin = originId;
            info.target = targetId;
            uint32_t id = atomicAdd(&lostCount, 1);
            pathfindingList.data[id] = info;
        }
    });

    grid.sync();

    pathfindingList.count = lostCount;

    grid.sync();

    return pathfindingList;
}

void PathfindingManager::update(Map &map, Entities &entities, Allocator &allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    __shared__ uint32_t fieldBuffer[IntegrationField::size()];

    // Each block handles a lost entity
    for_blockwise(min(pathfindingList.count, 500), [&](int bufferIdx) {
        PathfindingInfo info = pathfindingList.data[bufferIdx];
        IntegrationField field(info.target, fieldBuffer);

        // Init buffer
        processRangeBlock(IntegrationField::size(), [&](int idx) { field.resetCell(idx); });

        block.sync();

        if (block.thread_rank() == 0) {
            // Init target tile
            field.getCell(info.target) = 0;
        }

        block.sync();

        // Build integration field
        int iterations = 0;
        // The first path found is the smallest in size but not necessarily the shortest, because
        // there are different distance values. This condition ensures that it continues enough
        // to ensure path optimality.
        while (10 * iterations <= field.getCell(info.origin)) {
            iterations++;
            // The field is split accross the threads of the block
            processRangeBlock(IntegrationField::size(), [&](int currentCellId) {
                auto &fieldCell = field.getCell(currentCellId);

                map.extendedNeighborCells(currentCellId)
                    .forEachDir([&](Direction dir, int neighborId) {
                        if (!isNeighborValid(map, currentCellId, neighborId, dir, info.target)) {
                            return;
                        }
                        uint32_t neighborDistance = field.getCell(neighborId);
                        if (neighborDistance == uint32_t(Infinity)) {
                            return;
                        }
                        uint32_t newDistance = neighborDistance + (int(dir) < 4 ? 10u : 14u);
                        fieldCell = min(fieldCell, newDistance);
                    });
            });

            // Ensure that the iteration is completed by all threads before the next one
            block.sync();
        }

        // Extract path
        if (block.thread_rank() == 0) {
            entities.get(info.entityIdx).path = extractPath(map, field, info);
        }
        block.sync();
    });
}

Path PathfindingManager::extractPath(Map &map, const IntegrationField &field,
                                     const PathfindingInfo &info) const {
    int current = info.origin;
    bool reached = false;
    Path path;

    while (!reached && path.length() < Path::MAX_LENGTH) {
        // Retrieve path
        uint32_t min = uint32_t(Infinity);
        Direction dir;
        int nextCell;

        // We assume that there is always a possible path to the target
        map.extendedNeighborCells(current).forEachDir([&](Direction neighborDir, int neighborId) {
            if (!isNeighborValid(map, current, neighborId, neighborDir, info.target) || reached) {
                return;
            }
            if (neighborId == info.target) {
                reached = true;
                dir = neighborDir;
                min = 0;
                return;
            }

            uint32_t distance = field.getCell(neighborId);
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

void IntegrationField::resetCell(uint32_t cellId) {
    auto &fieldCell = getCell(cellId);
    fieldCell = uint32_t(Infinity);
}

bool PathfindingManager::isNeighborValid(Map &map, uint32_t cellId, uint32_t neighborId,
                                         Direction neighborDir, uint32_t targetId) const {
    if (neighborId != targetId && map.getTileId(neighborId) != ROAD) {
        return false;
    }

    if (int(neighborDir) < 4) {
        return true;
    }

    int2 currentCellCoord = map.cellCoords(cellId);
    int2 dirCoords = coordFromEnum(neighborDir);
    int id1 = map.idFromCoords(currentCellCoord + int2{dirCoords.x, 0});
    int id2 = map.idFromCoords(currentCellCoord + int2{0, dirCoords.y});
    return !((id1 == -1 || map.getTileId(id1) != ROAD) &&
             (id2 == -1 || map.getTileId(id2) != ROAD));
}
