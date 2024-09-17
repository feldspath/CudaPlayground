#include "common/helper_math.h"
#include "common/utils.cuh"

#include "pathfinding.cuh"

static int2 bordersStartingPoints[] = {int2(CHUNK_X - 1, 0), int2(0, 0), int2(0, CHUNK_Y - 1),
                                       int2(0, 0)};
static int2 borderDirs[] = {int2(0, 1), int2(0, 1), int2(1, 0), int2(1, 0)};

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
            auto target = entity.destination;
            auto origin = map.cellAtPosition(entity.position);
            if (!origin.valid()) {
                return;
            }
            if (!map.sharedNetworks(origin, target).data[0].valid()) {
                printf("Error: entity %d cannot reach its destination. Placing it back at home.\n",
                       entityIndex);
                entity.position = map.getCellPosition(entity.house);
                return;
            }
            uint32_t id = atomicAdd(&lostCount, 1);
            if (id > MAX_PATHS_PER_FRAME) {
                return;
            }

            PathfindingInfo info;
            info.origin = origin;
            info.entityIdx = entityIndex;
            info.destination = target;
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

void PathfindingManager::entitiesPathfinding(Map &map, Entities &entities, Allocator allocator) {
    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    // Extract the paths
    processRange(pathfindingList.count, [&](int idx) {
        auto &info = pathfindingList.data[idx];
        if (info.destination.chunkId == info.origin.chunkId) {
            auto &chunk = map.getChunk(info.destination.chunkId);
            if (chunk.cachedFlowfields[info.destination.cellId].state == VALID) {
                entities.get(info.entityIdx).path =
                    extractPath(chunk, info.origin.cellId, info.destination.cellId);
            }
        } else {
            int2 chunkDestCoords = map.chunkCoord(info.destination.chunkId);
            int2 chunkOrCoords = map.chunkCoord(info.origin.chunkId);
            int2 diff = chunkDestCoords - chunkOrCoords;

            if (abs(diff.x) + abs(diff.y) == 1) {
                Direction dir = enumFromCoord(diff);
                int side = int(dir);

                int minDist = uint32_t(Infinity);
                MapId bestCell = MapId::invalidId();

                for (int i = 0; i < CHUNK_X; ++i) {
                    int2 localCellCoord = bordersStartingPoints[side] + borderDirs[side] * i;
                    MapId cell(info.origin.chunkId, localCellCoord.y * CHUNK_X + localCellCoord.x);

                    MapId otherSide = map.cellFromCoords(map.cellCoords(cell) + diff);
                    if (map.getChunk(otherSide.chunkId)
                            .sharedNetworks(otherSide.cellId, info.destination.cellId)
                            .data[0] == -1) {
                        continue;
                    }
                    if (map.get(cell).tileId == ROAD && map.get(otherSide).tileId == ROAD) {
                        // We found one path, compute its length
                        int dist =
                            pathLength(map.getChunk(cell.chunkId), info.origin.cellId, cell.cellId);
                        if (dist < minDist) {
                            minDist = dist;
                            bestCell = cell;
                        }
                    }
                }

                if (!bestCell.valid()) {
                    printf("Error: entity %d cannot reach its destination.\n", info.entityIdx);
                    return;
                }

                Path path;
                auto &chunk = map.getChunk(info.origin.chunkId);
                if (chunk.cachedFlowfields[bestCell.cellId].state != VALID) {
                    return;
                }
                path = extractPath(chunk, info.origin.cellId, bestCell.cellId);
                if (path.length() < Path::MAX_LENGTH) {
                    path.append(dir);
                }
                entities.get(info.entityIdx).path = path;
                return;
            }
        }
    });
}

void PathfindingManager::update(Map &map, Allocator allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // list all the flowfields that have to be computed this frame
    uint32_t &flowfieldsToComputeCount = *allocator.alloc<uint32_t *>(sizeof(uint32_t));
    if (grid.thread_rank() == 0) {
        flowfieldsToComputeCount = 0;
    }
    grid.sync();

    auto flowfieldsToCompute = allocator.alloc<MapId *>(sizeof(MapId) * maxFlowfieldsPerFrame());

    auto atomicAddFlowfield = [&](MapId destination) {
        auto &chunk = map.getChunk(destination.chunkId);

        FlowfieldState oldState = FlowfieldState(atomicCAS(
            (int *)(&chunk.cachedFlowfields[destination.cellId].state), int(INVALID), int(MARKED)));

        if (oldState == INVALID) {
            int flowfieldIdx = atomicAdd(&flowfieldsToComputeCount, 1);
            if (flowfieldIdx >= maxFlowfieldsPerFrame()) {
                atomicCAS((int *)(&chunk.cachedFlowfields[destination.cellId].state), int(MARKED),
                          int(INVALID));
                return;
            }
            flowfieldsToCompute[flowfieldIdx] = MapId(destination.chunkId, destination.cellId);
        }
    };

    // First, the saved integrations fields
    processRange(gridDim.x, [&](int idx) {
        if (savedFields[idx].ongoingComputation) {
            auto target = savedFields[idx].target;
            map.getChunk(target.chunkId).cachedFlowfields[target.cellId].state = MARKED;
            int flowfieldIdx = atomicAdd(&flowfieldsToComputeCount, 1);
            flowfieldsToCompute[flowfieldIdx] = target;
        }
    });

    grid.sync();

    // Then the workplaces
    map.workplaces.processEachCell([&](MapId shop) {
        if (map.getChunk(shop.chunkId).cachedFlowfields[shop.cellId].state != VALID) {
            atomicAddFlowfield(shop);
        }
    });

    // And the houses
    map.houses.processEachCell([&](MapId house) {
        if (map.getChunk(house.chunkId).cachedFlowfields[house.cellId].state == VALID) {
            return;
        }
        atomicAddFlowfield(house);
    });

    // Finally, all the chunks borders
    for_blockwise(map.getCount(), [&](int chunkId) {
        processRangeBlock(CHUNK_X * 2 + CHUNK_Y * 2, [&](int borderId) {
            // Compute local coords
            // Assume that CHUNK_X == CHUNK_Y
            int side = borderId / CHUNK_X;
            int2 dirCoord = coordFromEnum(Direction(side));

            int2 localCellCoord =
                bordersStartingPoints[side] + borderDirs[side] * (borderId % CHUNK_X);
            MapId cell(chunkId, localCellCoord.y * CHUNK_X + localCellCoord.x);

            if (map.get(cell).tileId == ROAD &&
                map.get(map.cellFromCoords(map.cellCoords(cell) + dirCoord)).tileId == ROAD) {
                atomicAddFlowfield(cell);
            }
        });
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

    // Each block handles a flowfield
    int32_t previousChunk = -1;
    for_blockwise(min(flowfieldsToComputeCount, maxFlowfieldsPerFrame()), [&](int bufferIdx) {
        auto target = flowfieldsToCompute[bufferIdx];
        auto &chunk = map.getChunk(target.chunkId);

        // Move the tile ids to shared memory
        if (previousChunk != target.chunkId) {
            processRangeBlock(CHUNK_SIZE, [&](int cellId) {
                tilesBuffer[cellId] = uint8_t(chunk.get(cellId).tileId);
            });
            previousChunk = target.chunkId;
        }
        block.sync();

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
                if (idx == target.cellId) {
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
                // Check if cell is reachable
                if (chunk.sharedNetworks(currentCellId, target.cellId).data[0] == -1) {
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
                        (neighborId != target.cellId && TileId(tileIds[neighborId]) != ROAD)) {
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
                                             target.cellId)) {
                            return;
                        }

                        uint32_t distance = fieldBuffer[neighborId];
                        if (distance < minDistance) {
                            minDistance = distance;
                            dir = neighborDir;
                        }
                    });
                uint8_t approxDist = uint8_t(13 * (log2(float(minDistance)) - 3.32192809489f));
                chunk.cachedFlowfields[target.cellId].directions[cellId] = uint8_t(dir);
                chunk.cachedFlowfields[target.cellId].distances[cellId] = approxDist;
            });

            if (block.thread_rank() == 0) {
                chunk.cachedFlowfields[target.cellId].state = VALID;
            }
        }
    });

    grid.sync();
}

Path PathfindingManager::extractPath(Chunk &chunk, uint32_t origin, uint32_t target) const {
    int current = origin;
    bool reached = false;
    Path path;

    while (current != target && path.length() < Path::MAX_LENGTH) {
        Direction dir = Direction(chunk.cachedFlowfields[target].directions[current]);
        path.append(dir);
        current = chunk.neighborCell(current, dir);
        if (current == -1) {
            printf("pathfinding error\n");
            return Path();
        }
    }
    return path;
}

int PathfindingManager::pathLength(Chunk &chunk, uint32_t origin, uint32_t target) const {
    if (chunk.sharedNetworks(origin, target).data[0] == -1) {
        return int(Infinity);
    }

    uint8_t approx = chunk.cachedFlowfields[target].distances[origin];
    // uint_8 approxDist = uint_8(13 * (log2(float(minDistance)) - 3.32192809489f));
    return pow(2.0, double(approx) / 13 + 3.32192809489f);

    // int current = origin;
    // bool reached = false;
    // int length = 0;

    // while (current != target) {
    //     Direction dir = Direction(chunk.cachedFlowfields[target].directions[current]);
    //     length += int(dir) < 4 ? 10 : 14;
    //     current = chunk.neighborCell(current, dir);
    //     if (current == -1) {
    //         printf("path length pathfinding error\n");
    //         return -1;
    //     }
    // }
    // return length;
}