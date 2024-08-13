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
        if (lostCount > MAX_PATHS_PER_FRAME) {
            return;
        }

        Entity &entity = entities.get(entityIndex);
        if (entity.isLost()) {
            uint32_t targetId = entity.destination;
            int originId = map.cellAtPosition(entity.position);
            if (map.sharedNetworks(originId, targetId).data[0] == -1) {
                printf("Error: entity %d cannot reach its destination. Removing it.\n",
                       entityIndex);
                int workplaceId = entity.workplaceId;
                atomicAdd(&map.workplaceCapacity(workplaceId), 1);
                entities.remove(entityIndex);
                return;
            }
            uint32_t id = atomicAdd(&lostCount, 1);
            if (id > MAX_PATHS_PER_FRAME) {
                return;
            }

            PathfindingInfo info;
            info.entityIdx = entityIndex;
            info.origin = originId;
            info.target = targetId;
            pathfindingList.data[id] = info;
        }
    });

    grid.sync();

    pathfindingList.count = min(lostCount, MAX_PATHS_PER_FRAME);

    grid.sync();

    return pathfindingList;
}

#define PROFILE_START()                                                                            \
    block.sync();                                                                                  \
    uint64_t t_start = nanotime();

#define PROFILE_END(name)                                                                          \
    block.sync();                                                                                  \
    uint64_t t_end = nanotime();                                                                   \
    if (block.thread_rank() == 0) {                                                                \
        double nanos = double(t_end) - double(t_start);                                            \
        float millis = nanos / 1e6;                                                                \
        printf("%s: %8.3f ms\n", name, millis);                                                    \
    }

void PathfindingManager::update(Map &map, Entities &entities, Allocator &allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    if (grid.thread_rank() == 0 && pathfindingList.count > 0) {
        printf("pathfindings to compute count: %d\n", pathfindingList.count);
    }

    // list all the flowfields that have to be computed this frame
    uint32_t &flowfieldsToComputeCount = *allocator.alloc<uint32_t *>(sizeof(uint32_t));
    if (grid.thread_rank() == 0) {
        flowfieldsToComputeCount = 0;
    }
    grid.sync();

    uint32_t *flowfieldsToCompute =
        allocator.alloc<uint32_t *>(sizeof(uint32_t) * flowfieldsToComputeCount);

    processRange(pathfindingList.count, [&](int idx) {
        PathfindingInfo info = pathfindingList.data[idx];
        if (cachedFlowfields[info.target].state == VALID) {
            return;
        }

        FlowfieldState oldState = FlowfieldState(
            atomicCAS((int *)(&cachedFlowfields[info.target].state), int(INVALID), int(MARKED)));

        if (oldState == INVALID) {
            int flowfieldIdx = atomicAdd(&flowfieldsToComputeCount, 1);
            if (flowfieldIdx >= MAX_FLOWFIELDS_PER_FRAME) {
                return;
            }
            flowfieldsToCompute[flowfieldIdx] = info.target;
        }
    });

    grid.sync();

    if (grid.thread_rank() == 0 && flowfieldsToComputeCount > 0) {
        printf("flowfield to compute count: %d\n", flowfieldsToComputeCount);
    }

    // Compute the flowfields
    __shared__ uint32_t fieldBuffer[IntegrationField::size()];
    __shared__ TileId tilesBuffer[IntegrationField::size()];

    this->tileIds = tilesBuffer;

    processRangeBlock(IntegrationField::size(),
                      [&](int cellId) { tilesBuffer[cellId] = map.getTileId(cellId); });

    // Each block handles a flowfield
    for_blockwise(min(flowfieldsToComputeCount, MAX_FLOWFIELDS_PER_FRAME), [&](int bufferIdx) {
        uint32_t target = flowfieldsToCompute[bufferIdx];
        IntegrationField field(target, fieldBuffer);

        // Init buffer
        processRangeBlock(IntegrationField::size(), [&](int idx) {
            field.resetCell(idx);
            if (idx == target) {
                field.getCell(idx) = 0;
            }
        });

        block.sync();

        // Build integration field
        // The first path found is the smallest in size but not necessarily the shortest, because
        // there are different distance values. This condition ensures that it continues enough
        // to ensure path optimality.

        // simplifying the termination for now
        int iteration = 0;
        while (iteration < 64) {
            iteration++;
            // The field is split accross the threads of the block
            processRangeBlock(IntegrationField::size(), [&](int currentCellId) {
                auto &fieldCell = field.getCell(currentCellId);

                map.extendedNeighborCells(currentCellId)
                    .forEachDir([&](Direction dir, int neighborId) {
                        if (!isNeighborValid(map, currentCellId, neighborId, dir, target)) {
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

        // Create flowfield from integration field
        processRangeBlock(IntegrationField::size(), [&](int cellId) {
            uint32_t minDistance = uint32_t(Infinity);
            Direction dir;
            map.extendedNeighborCells(cellId).forEachDir(
                [&](Direction neighborDir, int neighborId) {
                    if (!isNeighborValid(map, cellId, neighborId, neighborDir, target)) {
                        return;
                    }

                    uint32_t distance = field.getCell(neighborId);
                    if (distance < minDistance) {
                        minDistance = distance;
                        dir = neighborDir;
                    }
                });

            cachedFlowfields[target].directions[cellId] = uint8_t(dir);
        });

        if (block.thread_rank() == 0) {
            cachedFlowfields[target].state = VALID;
        }
    });

    grid.sync();

    // Extract the paths
    processRange(min(pathfindingList.count, 2000), [&](int idx) {
        auto &info = pathfindingList.data[idx];
        if (cachedFlowfields[info.target].state == VALID) {
            entities.get(info.entityIdx).path = extractPath(map, info);
        }
    });
}

Path PathfindingManager::extractPath(Map &map, const PathfindingInfo &info) const {
    int current = info.origin;
    bool reached = false;
    Path path;

    while (!reached && path.length() < Path::MAX_LENGTH) {
        Direction dir = Direction(cachedFlowfields[info.target].directions[current]);
        path.append(dir);
        current = map.neighborCell(current, dir);
        if (current == -1) {
            printf("pathfinding error\n");
            return Path();
        }
    }
    return path;
}

void IntegrationField::resetCell(uint32_t cellId) {
    auto &fieldCell = getCell(cellId);
    fieldCell = uint32_t(Infinity);
}

bool PathfindingManager::isNeighborValid(Map &map, uint32_t cellId, uint32_t neighborId,
                                         Direction neighborDir, uint32_t targetId) const {
    if (neighborId != targetId && tileIds[neighborId] != ROAD) {
        return false;
    }

    if (int(neighborDir) < 4) {
        return true;
    }

    int2 currentCellCoord = map.cellCoords(cellId);
    int2 dirCoords = coordFromEnum(neighborDir);
    int id1 = map.idFromCoords(currentCellCoord + int2{dirCoords.x, 0});
    int id2 = map.idFromCoords(currentCellCoord + int2{0, dirCoords.y});
    return !((id1 == -1 || tileIds[id1] != ROAD) && (id2 == -1 || tileIds[id2] != ROAD));
}
