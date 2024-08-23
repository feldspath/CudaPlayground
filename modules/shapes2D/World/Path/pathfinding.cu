#include "common/helper_math.h"
#include "common/utils.cuh"

#include "pathfinding.cuh"

// Locate all entities that required pathfinding
PathfindingList PathfindingManager::locateLostEntities(Chunk &chunk, Entities &entities,
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
            int originId = chunk.cellAtPosition(entity.position);
            if (chunk.sharedNetworks(originId, targetId).data[0] == -1) {
                printf("Error: entity %d cannot reach its destination. Placing it back at home.\n",
                       entityIndex);
                entity.position = chunk.getCellPosition(entity.houseId);
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

#ifdef PROFILE
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
#else
#define PROFILE_START()
#define PROFILE_END(name)
#endif

static int2 closeNeighbors[] = {int2{-1, 0}, int2{1, 0}, int2{0, -1}, int2{0, 1}};
static int2 impactedNeighbors[] = {int2{0, 2}, int2{1, 3}, int2{0, 1}, int2{2, 3}};
static int2 farNeighbors[] = {int2{-1, -1}, int2{1, -1}, int2{-1, 1}, int2{1, 1}};

void PathfindingManager::update(Chunk &chunk, Entities &entities, Allocator &allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(chunk, entities, allocator);

    // if (grid.thread_rank() == 0 && pathfindingList.count > 0) {
    //     printf("pathfindings to compute count: %d\n", pathfindingList.count);
    // }

    // Remove remaning marks from the previous frame
    chunk.processEachCell([&](int idx) {
        if (chunk.cachedFlowfields[idx].state == MARKED) {
            chunk.cachedFlowfields[idx].state = INVALID;
        }
    });

    // list all the flowfields that have to be computed this frame
    uint32_t &flowfieldsToComputeCount = *allocator.alloc<uint32_t *>(sizeof(uint32_t));
    if (grid.thread_rank() == 0) {
        flowfieldsToComputeCount = 0;
    }
    grid.sync();

    uint32_t *flowfieldsToCompute =
        allocator.alloc<uint32_t *>(sizeof(uint32_t) * maxFlowfieldsPerFrame());

    // First, the saved integrations fields
    processRange(gridDim.x, [&](int idx) {
        if (savedFields[idx].ongoingComputation) {
            int target = savedFields[idx].target;
            chunk.cachedFlowfields[target].state = MARKED;
            int flowfieldIdx = atomicAdd(&flowfieldsToComputeCount, 1);
            flowfieldsToCompute[flowfieldIdx] = target;
        }
    });

    grid.sync();

    processRange(pathfindingList.count, [&](int idx) {
        PathfindingInfo info = pathfindingList.data[idx];
        if (chunk.cachedFlowfields[info.target].state == VALID) {
            return;
        }

        FlowfieldState oldState = FlowfieldState(atomicCAS(
            (int *)(&chunk.cachedFlowfields[info.target].state), int(INVALID), int(MARKED)));

        if (oldState == INVALID) {
            int flowfieldIdx = atomicAdd(&flowfieldsToComputeCount, 1);
            if (flowfieldIdx >= maxFlowfieldsPerFrame()) {
                return;
            }
            flowfieldsToCompute[flowfieldIdx] = info.target;
        }
    });

    grid.sync();

    // if (grid.thread_rank() == 0 && flowfieldsToComputeCount > 0) {
    //     printf("flowfield to compute count: %d\n", flowfieldsToComputeCount);
    // }

    // Compute the flowfields
    __shared__ uint32_t fieldBuffer[CHUNK_SIZE];
    __shared__ uint8_t tilesBuffer[CHUNK_SIZE];
    __shared__ uint32_t iterations[CHUNK_SIZE];

    this->tileIds = tilesBuffer;

    chunk.processEachCellBlock(
        [&](int cellId) { tilesBuffer[cellId] = uint8_t(chunk.get(cellId).tileId); });

    // Each block handles a flowfield
    for_blockwise(min(flowfieldsToComputeCount, maxFlowfieldsPerFrame()), [&](int bufferIdx) {
        uint32_t target = flowfieldsToCompute[bufferIdx];

        __shared__ int32_t savedFieldId;
        if (block.thread_rank() == 0) {
            savedFieldId = -1;
        }
        block.sync();
        processRangeBlock(gridDim.x, [&](int idx) {
            if (savedFields[idx].target == target && savedFields[idx].ongoingComputation) {
                savedFieldId = idx;
            }
        });
        block.sync();

        if (savedFieldId != -1) {
            // Load saved field
            chunk.processEachCellBlock([&](int idx) {
                fieldBuffer[idx] = savedFields[savedFieldId].distances[idx];
                iterations[idx] = savedFields[savedFieldId].iterations[idx];
            });
            if (block.thread_rank() == 0) {
                savedFields[savedFieldId].ongoingComputation = false;
            }
        } else {
            // Init buffer
            chunk.processEachCellBlock([&](int idx) {
                if (idx == target) {
                    fieldBuffer[idx] = 0;
                } else {
                    fieldBuffer[idx] = uint32_t(Infinity);
                }
                iterations[idx] = 0;
            });
        }
        block.sync();

        // Build integration field
        PROFILE_START();
        __shared__ bool updated;
        if (block.thread_rank() == 0) {
            updated = true;
        }
        block.sync();

        int threadIterations = 0;
        while (updated && threadIterations < 64) {
            if (block.thread_rank() == 0) {
                updated = false;
            }
            block.sync();
            threadIterations++;
            // The field is split accross the threads of the block
            for (int currentCellId = block.thread_rank(); currentCellId < CHUNK_SIZE;
                 currentCellId += block.size()) {
                if (currentCellId >= CHUNK_SIZE) {
                    return;
                }

                // Check if cell is reachable
                if (chunk.sharedNetworks(currentCellId, target).data[0] == -1) {
                    continue;
                }

                // The first path found is the smallest in size but not necessarily the shortest,
                // because there are different distance values. This condition ensures that it
                // continues enough to ensure path optimality.
                if (10 * iterations[currentCellId] > fieldBuffer[currentCellId] ||
                    iterations[currentCellId] > CHUNK_SIZE) {
                    // This cell is done
                    continue;
                }
                updated = true;
                iterations[currentCellId]++;

                int2 currentCellCoord = chunk.cellCoords(currentCellId);
                uint32_t minDistance = fieldBuffer[currentCellId];

                int2 toVisit[] = {closeNeighbors[0], closeNeighbors[1], closeNeighbors[2],
                                  closeNeighbors[3], int2{0, 0},        int2{0, 0},
                                  int2{0, 0},        int2{0, 0}};
                int size = 4;
                int idx = 0;
                bool pushed[] = {false, false, false, false};
                while (idx < size) {
                    int2 neighborDir = toVisit[idx];
                    int neighborId = chunk.idFromCoords(currentCellCoord + neighborDir);
                    if (neighborId == currentCellId || neighborId == -1 ||
                        (neighborId != target && TileId(tileIds[neighborId]) != ROAD)) {
                        idx++;
                        continue;
                    }
                    bool diag = idx >= 4;
                    if (!diag) {
                        int2 impacted = impactedNeighbors[idx];
                        if (!pushed[impacted.x]) {
                            pushed[impacted.x] = true;
                            toVisit[size] = farNeighbors[impacted.x];
                            size++;
                        }
                        if (!pushed[impacted.y]) {
                            pushed[impacted.y] = true;
                            toVisit[size] = farNeighbors[impacted.y];
                            size++;
                        }
                    }
                    uint32_t neighborDistance = fieldBuffer[neighborId];
                    uint32_t newDistance = neighborDistance + (diag ? 14 : 10);
                    minDistance = min(minDistance, newDistance);
                    idx++;
                }

                fieldBuffer[currentCellId] = minDistance;
            }
            // Ensure that the iteration is completed by all threads before the next one
            block.sync();
        }
        PROFILE_END("integration field");

        if (updated) {
            // Integration field is not complete, save it to reuse it next frame
            chunk.processEachCellBlock([&](int idx) {
                savedFields[bufferIdx].distances[idx] = fieldBuffer[idx];
                savedFields[bufferIdx].iterations[idx] = iterations[idx];
                savedFields[bufferIdx].target = target;
                savedFields[bufferIdx].ongoingComputation = true;
            });
        } else {
            // Integration field is complete, create flowfield
            chunk.processEachCellBlock([&](int cellId) {
                uint32_t minDistance = uint32_t(Infinity);
                Direction dir;
                chunk.extendedNeighborCells(cellId).forEachDir(
                    [&](Direction neighborDir, int neighborId) {
                        if (!isNeighborValid(chunk, cellId, neighborId, coordFromEnum(neighborDir),
                                             target)) {
                            return;
                        }

                        uint32_t distance = fieldBuffer[neighborId];
                        if (distance < minDistance) {
                            minDistance = distance;
                            dir = neighborDir;
                        }
                    });

                chunk.cachedFlowfields[target].directions[cellId] = uint8_t(dir);
            });

            if (block.thread_rank() == 0) {
                chunk.cachedFlowfields[target].state = VALID;
            }
        }
    });

    grid.sync();

    // Extract the paths
    processRange(pathfindingList.count, [&](int idx) {
        auto &info = pathfindingList.data[idx];
        if (chunk.cachedFlowfields[info.target].state == VALID) {
            entities.get(info.entityIdx).path = extractPath(chunk, info);
        }
    });
}

Path PathfindingManager::extractPath(Chunk &chunk, const PathfindingInfo &info) const {
    int current = info.origin;
    bool reached = false;
    Path path;

    while (!reached && path.length() < Path::MAX_LENGTH) {
        Direction dir = Direction(chunk.cachedFlowfields[info.target].directions[current]);
        path.append(dir);
        current = chunk.neighborCell(current, dir);
        if (current == -1) {
            printf("pathfinding error\n");
            return Path();
        }
    }
    return path;
}
