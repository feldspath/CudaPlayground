#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "entities.h"
#include "helper_math.h"
#include "map.h"
#include "matrix_math.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
GameState *gameState;
Allocator *allocator;
uint64_t nanotime_start;

struct UpdateInfo {
    bool update;
    int tileToUpdate;
    TileId newTileId;
};

void updateCell(Map *map, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    TileId new_tile = updateInfo.newTileId;
    int id = updateInfo.tileToUpdate;

    grid.sync();
    if (grid.thread_rank() == 0) {
        map->setTileId(id, new_tile);
    }
    grid.sync();

    switch (new_tile) {
    case ROAD: {
        int *cumulNeighborNetworksSizes = allocator->alloc<int *>(sizeof(int) * 5);
        int *neighborNetworksReprs = allocator->alloc<int *>(sizeof(int) * 4);

        if (grid.thread_rank() == 0) {
            // check nearby tiles.
            auto neighbors = map->neighborCells(id);
            int neighborNetworksSizes[4];

            for (int i = 0; i < 4; i++) {
                int nId = neighbors.data[i];
                // if one tile is not grass, update the connected components
                if (nId != -1 && map->getTileId(nId) == ROAD) {
                    int repr = map->roadNetworkRepr(nId);
                    // Skip the tile if it was already updated this frame
                    if (map->roadNetworkRepr(repr) == repr) {
                        neighborNetworksSizes[i] = map->roadNetworkId(repr);
                        neighborNetworksReprs[i] = repr;
                        map->roadNetworkRepr(repr) = id;
                        continue;
                    }
                }
                neighborNetworksSizes[i] = 0;
                neighborNetworksReprs[i] = -1;
            }

            cumulNeighborNetworksSizes[0] = 0;
            for (int i = 0; i < 4; i++) {
                cumulNeighborNetworksSizes[i + 1] =
                    cumulNeighborNetworksSizes[i] + neighborNetworksSizes[i];
            }

            // Init the new road tile
            map->roadNetworkRepr(id) = id;
            map->roadNetworkId(id) = cumulNeighborNetworksSizes[4] + 1;
        }

        grid.sync();

        // Flatten network
        map->processEachCell(ROAD, [&](int cellId) {
            int neighborId = -1;
            for (int i = 0; i < 4; ++i) {
                if (map->roadNetworkRepr(cellId) == neighborNetworksReprs[i] ||
                    cellId == neighborNetworksReprs[i]) {
                    neighborId = i;
                    break;
                }
            }
            if (neighborId == -1) {
                return;
            }

            map->roadNetworkRepr(cellId) = id;
            map->roadNetworkId(cellId) += cumulNeighborNetworksSizes[neighborId];
        });
        break;
    }
    case FACTORY:
        if (grid.thread_rank() == 0) {
            // Set capacity
            *map->factoryTileData(id) = FACTORY_CAPACITY;
        }
        break;

    case HOUSE:
        if (grid.thread_rank() == 0) {
            // Set house to unassigned
            *map->houseTileData(id) = -1;
        }
        break;
    default:
        break;
    }
}

void assignHouseToFactory(Map *map, Entities *entities, int32_t houseId, int32_t factoryId) {
    int32_t newEntity = entities->newEntity(map->getCellPosition(houseId), houseId, factoryId);
    int32_t *houseData = map->houseTileData(houseId);
    *houseData = newEntity;

    *map->factoryTileData(factoryId) -= 1;
}

void assignOneHouse(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    if (grid.thread_rank() == 0) {
        assigned = 0;
    }
    grid.sync();

    // TODO: first scan the tiles that require assignment using the whole grid
    __shared__ uint64_t targetFactory;
    for (int offset = 0; offset < map->count; offset += grid.num_blocks()) {
        int currentTile = offset + grid.block_rank();
        if (currentTile > map->count) {
            break;
        }

        // Skip if tile is not a house or already assigned
        if (map->getTileId(currentTile) != HOUSE || *map->houseTileData(currentTile) != -1) {
            continue;
        }

        // Get neighbor networks
        auto houseNets = map->neighborNetworks(currentTile);
        int2 tileCoords = map->cellCoords(currentTile);

        if (block.thread_rank() == 0) {
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        // Check all tiles for factories
        for (int offset = 0; offset < map->count; offset += block.num_threads()) {
            int factoryId = block.thread_rank() + offset;
            if (factoryId >= map->count) {
                break;
            }

            // Look for factories ...
            if (map->getTileId(factoryId) != FACTORY) {
                continue;
            }
            // ... with some capacity
            if (*map->factoryTileData(factoryId) == 0) {
                continue;
            }

            // Get the networks the factory is connected to
            auto factoryNets = map->neighborNetworks(factoryId);
            for (int i = 0; i < 16; i++) {
                int f = factoryNets.data[i % 4];
                int h = houseNets.data[i / 4];
                if (f != -1 && f == h) {
                    // This factory shares the same network
                    int2 factoryCoords = map->cellCoords(factoryId);
                    int2 diff = factoryCoords - tileCoords;
                    uint32_t distance = abs(diff.x) + abs(diff.y);
                    uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(factoryId);
                    // keep the closest factory
                    atomicMin(&targetFactory, target);
                    break;
                }
            }
        }

        block.sync();

        if (block.thread_rank() == 0) {
            int32_t *houseData = map->houseTileData(currentTile);
            if (targetFactory != uint64_t(Infinity) << 32ull && !atomicAdd(&assigned, 1)) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                assignHouseToFactory(map, entities, currentTile, factoryId);
            } else {
                *houseData = -1;
            }
        }

        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }

void updateEntities(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Each thread handles an entity
    for (int offset = 0; offset < entities->getCount(); offset += grid.num_threads()) {
        int entityIndex = offset + grid.thread_rank();
        if (entityIndex >= entities->getCount()) {
            break;
        }

        switch (entities->entityState(entityIndex)) {
        case GoHome: {
            if (entities->isPathValid(entityIndex)) {
                Direction dir = entities->nextPathDirection(entityIndex);
                Entity &entity = *entities->entityPtr(entityIndex);
                if (entities->moveEntityDir(entityIndex, dir, gameState->dt, map)) {
                    entities->advancePath(entityIndex);
                    entities->stateStart_ms(entityIndex) = gameState->currentTime_ms;
                    if (entities->getPathLength(entityIndex) == 0 &&
                        map->cellAtPosition(entity.position) == entity.houseId) {
                        entity.state = Rest;
                        entity.stateStart_ms = gameState->currentTime_ms;
                        entity.position = map->getCellPosition(entity.houseId);
                    }
                }
            }
            break;
        }
        case GoToWork: {
            if (entities->isPathValid(entityIndex)) {
                Direction dir = entities->nextPathDirection(entityIndex);
                Entity &entity = *entities->entityPtr(entityIndex);
                if (entities->moveEntityDir(entityIndex, dir, gameState->dt, map)) {
                    entities->advancePath(entityIndex);
                    entities->stateStart_ms(entityIndex) = gameState->currentTime_ms;
                    if (entities->getPathLength(entityIndex) == 0 &&
                        map->cellAtPosition(entity.position) == entity.factoryId) {
                        entity.state = Work;
                        entity.stateStart_ms = gameState->currentTime_ms;
                        entity.position = map->getCellPosition(entity.factoryId);
                    }
                }
            }
            break;
        }
        case Work:
            if (gameState->currentTime_ms - entities->stateStart_ms(entityIndex) >= WORK_TIME_MS) {
                entities->entityState(entityIndex) = GoHome;
            }
            break;
        case Rest:
            if (gameState->currentTime_ms - entities->stateStart_ms(entityIndex) >= REST_TIME_MS) {
                entities->entityState(entityIndex) = GoToWork;
            }
            break;
        default:
            break;
        }
    }
}

void updateGameState() {
    gameState->dt = ((float)(nanotime_start - gameState->previousFrameTime_ns)) / 1e9;
    gameState->previousFrameTime_ns = nanotime_start;
    gameState->currentTime_ms = currentTime_ms();
}

struct PathCell {
    uint32_t distance;
    int32_t cellId;
};

struct PathfindingInfo {
    // cell id of the networks repr
    NeighborNetworks networkIds;

    uint32_t entityIdx;
    uint32_t targetId;
    uint32_t originId;

    // buffer of size sum of all networks to explore
    PathCell *buffer;
    uint32_t bufferSize;
};

int pathfindingBufferIndex(Map *map, PathfindingInfo &info, uint32_t cellId) {
    int offset = 0;
    int32_t network = map->roadNetworkRepr(cellId);

    for (int i = 0; i < 4; i++) {
        int ni = ((int *)(&info.networkIds))[i];
        if (network == ni) {
            break;
        }
        offset += map->roadNetworkId(ni);
    }

    return offset + map->roadNetworkId(cellId) % map->roadNetworkId(network);
}

void performPathFinding(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    PathfindingInfo *lostEntities =
        allocator->alloc<PathfindingInfo *>(entities->getCount() * sizeof(PathfindingInfo));
    uint32_t &lostCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        lostCount = 0;
    }
    grid.sync();

    // Locate all entities that required pathfinding
    for (int offset = 0; offset < entities->getCount(); offset += grid.num_threads()) {
        int entityIndex = grid.thread_rank() + offset;
        if (entityIndex >= entities->getCount()) {
            break;
        }

        Entity &entity = *entities->entityPtr(entityIndex);
        if ((entity.state == GoToWork || entity.state == GoHome) &&
            !entities->isPathValid(entityIndex)) {
            uint32_t bufferId = atomicAdd(&lostCount, 1);
            PathfindingInfo info;
            uint32_t targetId = entity.state == GoHome ? entity.houseId : entity.factoryId;
            int originId = map->cellAtPosition(entities->entityPosition(entityIndex));
            info.entityIdx = entityIndex;
            info.networkIds = map->sharedNetworks(originId, targetId);
            info.targetId = targetId;
            info.originId = originId;
            lostEntities[bufferId] = info;
        }
    }

    grid.sync();

    // Allocate a buffer in global memory for each lost entity
    for (int i = 0; i < lostCount; ++i) {
        int bufferSize = 0;
        for (int j = 0; j < 4; ++j) {
            int n = ((int *)(&(lostEntities[i].networkIds)))[j];
            if (n == -1) {
                break;
            }
            bufferSize += map->roadNetworkId(n);
        }

        PathCell *ptr = allocator->alloc<PathCell *>(bufferSize * sizeof(PathCell));
        if (grid.thread_rank() == 0) {
            lostEntities[i].buffer = ptr;
            lostEntities[i].bufferSize = bufferSize;
        }
    }

    grid.sync();

    // Each block handles a lost entity
    for (int offset = 0; offset < lostCount; offset += grid.num_blocks()) {
        int bufferIdx = offset + grid.block_rank();
        if (bufferIdx >= lostCount) {
            break;
        }

        PathfindingInfo info = lostEntities[bufferIdx];

        int targetBufferId = pathfindingBufferIndex(map, info, info.targetId);
        int originBufferId = pathfindingBufferIndex(map, info, info.originId);

        // Init buffer
        for (int roadOffset = 0; roadOffset < info.bufferSize; roadOffset += block.num_threads()) {
            int idx = roadOffset + block.thread_rank();
            if (idx >= info.bufferSize) {
                break;
            }

            PathCell init;
            init.distance = uint32_t(Infinity);
            init.cellId = -1;
            info.buffer[idx] = init;
        }

        if (block.thread_rank() == 0) {
            // Init neighbor tiles of target
            auto neighbors = map->neighborCells(info.targetId);
            for (int i = 0; i < 4; i++) {
                int n = neighbors.data[i];
                if (n != -1 && map->getTileId(n) == ROAD &&
                    info.networkIds.contains(map->roadNetworkRepr(n))) {
                    int bufferId = pathfindingBufferIndex(map, info, n);
                    PathCell pathCell;
                    pathCell.cellId = n;
                    pathCell.distance = 0;
                    info.buffer[bufferId] = pathCell;
                }
            }
        }

        block.sync();

        auto isOriginReached = [](int origin, Map *map, PathfindingInfo &info) {
            if (map->getTileId(origin) == ROAD) {
                int bufferId = pathfindingBufferIndex(map, info, origin);
                return info.buffer[bufferId].distance != uint32_t(Infinity);
            }

            auto originNeighbors = map->neighborCells(origin);
            for (int i = 0; i < 4; ++i) {
                int nId = originNeighbors.data[i];
                if (map->getTileId(nId) == ROAD) {
                    int bufferId = pathfindingBufferIndex(map, info, nId);
                    if (info.buffer[bufferId].distance != uint32_t(Infinity)) {
                        return true;
                    }
                }
            }
            return false;
        };

        // Build flowfield
        while (!isOriginReached(info.originId, map, info)) {
            for (int roadOffset = 0; roadOffset < info.bufferSize;
                 roadOffset += block.num_threads()) {
                int currentRoadBufferId = roadOffset + block.thread_rank();
                if (currentRoadBufferId > info.bufferSize) {
                    break;
                }

                PathCell cell = info.buffer[currentRoadBufferId];
                if (cell.cellId == -1) {
                    continue;
                }

                // Retrieve neighbor tiles
                auto neighbors = map->neighborCells(cell.cellId);
                for (int i = 0; i < 4; ++i) {
                    int neighborId = neighbors.data[i];
                    if (neighborId != -1 && map->getTileId(neighborId) == ROAD) {
                        int neighborBufferId = pathfindingBufferIndex(map, info, neighborId);
                        // Atomically update neighbor tiles id if not set
                        int oldNeighborId =
                            atomicCAS(&(info.buffer[neighborBufferId].cellId), -1, neighborId);

                        if (oldNeighborId != -1) {
                            // Set distance value
                            info.buffer[currentRoadBufferId].distance =
                                min(info.buffer[currentRoadBufferId].distance,
                                    info.buffer[neighborBufferId].distance + 1);
                        }
                    }
                }
            }
            // not sure if this is needed or not
            block.sync();
        }

        // Extract path
        if (block.thread_rank() == 0) {
            int current = info.originId;
            bool reached = false;
            entities->resetPath(info.entityIdx);
            while (!reached && entities->getPathLength(info.entityIdx) < 29) {
                // Retrieve path
                auto neighbors = map->neighborCells(current);
                uint32_t min = uint32_t(Infinity);
                Direction dir;
                int nextCell;
                for (int i = 0; i < 4; ++i) {
                    int nId = neighbors.data[i];
                    if (nId == info.targetId) {
                        reached = true;
                        dir = Direction(i);
                        break;
                    } else if (map->getTileId(nId) == ROAD) {
                        int bufferId = pathfindingBufferIndex(map, info, nId);
                        if (info.buffer[bufferId].distance < min) {
                            min = info.buffer[bufferId].distance;
                            dir = Direction(i);
                            nextCell = nId;
                        }
                    }
                }
                entities->pushBackPath(info.entityIdx, dir);
                current = nextCell;
            }
        }
    }
}

void updateGrid(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime_start));

    if (uniforms.cursorPos.x >= 0 && uniforms.cursorPos.x < uniforms.width &&
        uniforms.cursorPos.y >= 0 && uniforms.cursorPos.y < uniforms.height) {
        UpdateInfo updateInfo;

        bool mousePressed = uniforms.mouseButtons & 1;
        updateInfo.update = false;

        if (mousePressed) {
            float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
            float3 pos_W =
                unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
            int id = map->cellAtPosition(float2{pos_W.x, pos_W.y});

            if (id != -1 && map->getTileId(id) == GRASS) {
                updateInfo.update = true;
                updateInfo.tileToUpdate = id;
                updateInfo.newTileId = (TileId)uniforms.modeId;
            }
        }

        if (updateInfo.update) {
            updateCell(map, updateInfo);
        }
    }

    assignOneHouse(map, entities);

    performPathFinding(map, entities);
    updateEntities(map, entities);

    // grid.sync();
    if (grid.thread_rank() == 0) {
        updateGameState();
    }
}

extern "C" __global__ void update(const Uniforms _uniforms, GameState *_gameState,
                                  unsigned int *buffer, uint32_t numRows, uint32_t numCols,
                                  char *cells, void *entitiesBuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    gameState = _gameState;

    grid.sync();

    {
        Map *map = allocator->alloc<Map *>(sizeof(Map));
        *map = Map(numRows, numCols, cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(entitiesBuffer);

        updateGrid(map, entities);
    }
}
