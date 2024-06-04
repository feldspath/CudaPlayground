#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cells.h"
#include "entities.h"
#include "helper_math.h"
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

int cellNetworkId(int cellId, Grid2D *grid2D) {
    if (cellId != -1 && grid2D->getTileId(cellId) == ROAD) {
        return grid2D->roadNetworkRepr(cellId);
    } else {
        return -1;
    }
}

int4 neighborCells(int cellId, Grid2D *grid2D) {
    int2 coords = grid2D->cellCoords(cellId);
    int right = grid2D->idFromCoords(coords.x + 1, coords.y);
    int left = grid2D->idFromCoords(coords.x - 1, coords.y);
    int up = grid2D->idFromCoords(coords.x, coords.y + 1);
    int down = grid2D->idFromCoords(coords.x, coords.y - 1);

    return int4{right, left, up, down};
}

int4 neighborNetworks(int cellId, Grid2D *grid2D) {
    int4 neighbors = neighborCells(cellId, grid2D);
    int4 comps = {cellNetworkId(neighbors.x, grid2D), cellNetworkId(neighbors.y, grid2D),
                  cellNetworkId(neighbors.z, grid2D), cellNetworkId(neighbors.w, grid2D)};

    return comps;
}

bool networkInNetworks(int network, int4 networks) {
    for (int i = 0; i < 4; ++i) {
        int n = ((int *)(&network))[i];
        if (n == network) {
            return true;
        }
    }
    return false;
}

int4 commonNetworks(int4 nets1, int4 nets2, Grid2D *grid2D) {
    int4 result = {-1, -1, -1, -1};
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        int ni = ((int *)(&nets1))[i];
        if (ni != -1 && networkInNetworks(ni, nets2)) {
            ((int *)(&result))[count] = ni;
            count++;
        }
    }
    return result;
}

int4 commonNetworks(int cellId1, int cellId2, Grid2D *grid2D) {
    int4 nets1 = neighborNetworks(cellId1, grid2D);
    int4 nets2 = neighborNetworks(cellId2, grid2D);
    return commonNetworks(nets1, nets2, grid2D);
}

void updateCell(Grid2D *grid2D, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    TileId new_tile = updateInfo.newTileId;
    int id = updateInfo.tileToUpdate;

    grid.sync();
    if (grid.thread_rank() == 0) {
        grid2D->setTileId(id, new_tile);
    }
    grid.sync();

    switch (new_tile) {
    case ROAD: {
        // TODO: update using the entire grid
        if (grid.block_rank() == 0) {

            __shared__ int cumulNeighborNetworksSizes[5];
            __shared__ int neighborNetworksReprs[4];

            if (block.thread_rank() == 0) {

                // check nearby tiles.
                int4 neighbors = neighborCells(id, grid2D);
                int neighborNetworksSizes[4];

                for (int i = 0; i < 4; i++) {
                    int nId = ((int *)(&neighbors))[i];
                    // if one tile is not grass, update the connected components
                    if (nId != -1 && grid2D->getTileId(nId) == ROAD) {
                        int repr = grid2D->roadNetworkRepr(nId);
                        // Skip the tile if it was already updated this frame
                        if (grid2D->roadNetworkRepr(repr) == repr) {
                            neighborNetworksSizes[i] = grid2D->roadNetworkId(repr);
                            neighborNetworksReprs[i] = repr;
                            grid2D->roadNetworkRepr(repr) = id;
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
                grid2D->roadNetworkRepr(id) = id;
                grid2D->roadNetworkId(id) = cumulNeighborNetworksSizes[4] + 1;
            }

            block.sync();

            // Flatten network
            for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
                int cellId = block.thread_rank() + offset;
                if (cellId >= grid2D->count || grid2D->getTileId(cellId) != ROAD) {
                    continue;
                }

                int neighborId = -1;
                for (int i = 0; i < 4; ++i) {
                    if (grid2D->roadNetworkRepr(cellId) == neighborNetworksReprs[i] ||
                        cellId == neighborNetworksReprs[i]) {
                        neighborId = i;
                        break;
                    }
                }
                if (neighborId == -1) {
                    continue;
                }

                grid2D->roadNetworkRepr(cellId) = id;
                grid2D->roadNetworkId(cellId) += cumulNeighborNetworksSizes[neighborId];
            }
        }
        break;
    }
    case FACTORY:
        if (grid.thread_rank() == 0) {
            // Set capacity
            *grid2D->factoryTileData(id) = FACTORY_CAPACITY;
        }
        break;

    case HOUSE:
        if (grid.thread_rank() == 0) {
            // Set house to unassigned
            *grid2D->houseTileData(id) = -1;
        }
        break;
    default:
        break;
    }
}

void assignHouseToFactory(Grid2D *grid2D, Entities *entities, int32_t houseId, int32_t factoryId) {
    int32_t newEntity = entities->newEntity(grid2D->getCellPosition(houseId), houseId, factoryId);
    int32_t *houseData = grid2D->houseTileData(houseId);
    *houseData = newEntity;

    *grid2D->factoryTileData(factoryId) -= 1;
}

void assignOneHouse(Grid2D *grid2D, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    if (grid.thread_rank() == 0) {
        assigned = 0;
    }
    grid.sync();

    // TODO: first scan the tiles that require assignment using the whole grid

    __shared__ uint64_t targetFactory;
    for (int offset = 0; offset < grid2D->count; offset += grid.num_blocks()) {
        int currentTile = offset + grid.block_rank();
        if (currentTile > grid2D->count) {
            break;
        }

        // Skip if tile is not a house or already assigned
        if (grid2D->getTileId(currentTile) != HOUSE || *grid2D->houseTileData(currentTile) != -1) {
            continue;
        }

        // Get neighbor networks
        int4 houseNets = neighborNetworks(currentTile, grid2D);
        int2 tileCoords = grid2D->cellCoords(currentTile);

        if (block.thread_rank() == 0) {
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        // Check all tiles for factories
        for (int offset = 0; offset < grid2D->count; offset += block.num_threads()) {
            int factoryId = block.thread_rank() + offset;
            if (factoryId >= grid2D->count) {
                break;
            }

            // Look for factories ...
            if (grid2D->getTileId(factoryId) != FACTORY) {
                continue;
            }
            // ... with some capacity
            if (*grid2D->factoryTileData(factoryId) == 0) {
                continue;
            }

            // Get the networks the factory is connected to
            int4 factoryNets = neighborNetworks(factoryId, grid2D);
            for (int i = 0; i < 16; i++) {
                int f = ((int *)(&factoryNets))[i % 4];
                int h = ((int *)(&houseNets))[i / 4];
                if (f != -1 && f == h) {
                    // This factory shares the same network
                    int2 factoryCoords = grid2D->cellCoords(factoryId);
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
            int32_t *houseData = grid2D->houseTileData(currentTile);
            if (targetFactory != uint64_t(Infinity) << 32ull && !atomicAdd(&assigned, 1)) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                assignHouseToFactory(grid2D, entities, currentTile, factoryId);
            } else {
                *houseData = -1;
            }
        }

        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }
float frameTime() { return ((float)(nanotime_start - gameState->previousFrameTime_ns)) / 1e9; }

void updateEntities(Grid2D *grid2D, Entities *entities) {
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
                if (entities->moveEntityDir(entityIndex, dir, frameTime(), grid2D)) {
                    entities->advancePath(entityIndex);
                    if (entities->getPathLength(entityIndex) == 0 &&
                        grid2D->cellAtPosition(entity.position) == entity.houseId) {
                        entity.state = Rest;
                        entity.stateStart_ms = currentTime_ms();
                        entity.position = grid2D->getCellPosition(entity.houseId);
                    }
                }
            }
            break;
        }
        case GoToWork: {
            if (entities->isPathValid(entityIndex)) {
                Direction dir = entities->nextPathDirection(entityIndex);
                Entity &entity = *entities->entityPtr(entityIndex);
                if (entities->moveEntityDir(entityIndex, dir, frameTime(), grid2D)) {
                    entities->advancePath(entityIndex);
                    if (entities->getPathLength(entityIndex) == 0 &&
                        grid2D->cellAtPosition(entity.position) == entity.factoryId) {
                        entity.state = Work;
                        entity.stateStart_ms = currentTime_ms();
                        entity.position = grid2D->getCellPosition(entity.factoryId);
                    }
                }
            }
            break;
        }
        case Work:
            if (currentTime_ms() - entities->stateStart_ms(entityIndex) >= WORK_TIME_MS) {
                entities->entityState(entityIndex) = GoHome;
            }
            break;
        case Rest:
            if (currentTime_ms() - entities->stateStart_ms(entityIndex) >= REST_TIME_MS) {
                entities->entityState(entityIndex) = GoToWork;
            }
            break;
        default:
            break;
        }
    }
}

void updateGameState() { gameState->previousFrameTime_ns = nanotime_start; }

struct PathCell {
    uint32_t distance;
    int32_t cellId;
};

struct PathfindingInfo {
    // cell id of the networks repr
    int4 networkIds;

    uint32_t entityIdx;
    uint32_t targetId;

    // buffer of size sum of all networks to explore
    PathCell *buffer;
    uint32_t bufferSize;
};

int pathfindingBufferIndex(Grid2D *grid2D, PathfindingInfo &info, uint32_t cellId) {
    int offset = 0;
    int32_t network = grid2D->roadNetworkRepr(cellId);

    for (int i = 0; i < 4; i++) {
        int ni = ((int *)(&info.networkIds))[i];
        if (network == ni) {
            break;
        }
        offset += grid2D->roadNetworkId(ni);
    }

    return offset + grid2D->roadNetworkId(cellId) % grid2D->roadNetworkId(network);
}

void performPathFinding(Grid2D *grid2D, Entities *entities) {
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
            info.entityIdx = entityIndex;
            info.networkIds = commonNetworks(entityIndex, targetId, grid2D);
            info.targetId = targetId;
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
            bufferSize += grid2D->roadNetworkId(n);
        }
        PathCell *ptr = allocator->alloc<PathCell *>(bufferSize * sizeof(PathCell));
        if (grid.thread_rank() == 0) {
            lostEntities[i].buffer = ptr;
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

        Entity &entity = *entities->entityPtr(info.entityIdx);
        int32_t origin = grid2D->cellAtPosition(entity.position);

        int targetBufferId = pathfindingBufferIndex(grid2D, info, info.targetId);
        int originBufferId = pathfindingBufferIndex(grid2D, info, origin);

        // Reset buffer
        for (int roadOffset = 0; roadOffset < info.bufferSize; roadOffset += block.num_threads()) {
            int idx = roadOffset + block.thread_rank();
            if (idx >= info.bufferSize) {
                break;
            }

            PathCell init;
            init.distance = 0;
            init.cellId = -1;
            info.buffer[idx] = init;
        }

        if (block.thread_rank() == 0) {
            // Init neighbor tiles of target
            int4 neighbors = neighborCells(info.targetId, grid2D);
            for (int i = 0; i < 4; i++) {
                int n = ((int *)(&neighbors))[i];
                if (n != -1 && grid2D->getTileId(n) == ROAD &&
                    networkInNetworks(grid2D->roadNetworkRepr(n), info.networkIds)) {
                    int bufferId = pathfindingBufferIndex(grid2D, info, n);
                    PathCell pathCell;
                    pathCell.cellId = n;
                    pathCell.distance = 0;
                    info.buffer[bufferId] = pathCell;
                }
            }
        }

        block.sync();

        auto isOriginReached = [](int origin, Grid2D *grid2D, PathfindingInfo &info) {
            if (grid2D->getTileId(origin) == ROAD) {
                int bufferId = pathfindingBufferIndex(grid2D, info, origin);
                return info.buffer[bufferId].distance != uint32_t(Infinity);
            }

            int4 originNeighbors = neighborCells(origin, grid2D);
            for (int i = 0; i < 4; ++i) {
                int nId = ((int *)(&originNeighbors))[i];
                if (grid2D->getTileId(nId) == ROAD) {
                    int bufferId = pathfindingBufferIndex(grid2D, info, nId);
                    if (info.buffer[bufferId].distance != uint32_t(Infinity)) {
                        return true;
                    }
                }
            }
            return false;
        };

        // Build flowfield
        while (!isOriginReached(origin, grid2D, info)) {
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
                int4 neighbors = neighborCells(cell.cellId, grid2D);
                for (int i = 0; i < 4; ++i) {
                    int neighborId = ((int *)(&neighbors))[i];
                    if (neighborId != -1 && grid2D->getTileId(neighborId) == ROAD) {
                        int neighborBufferId = pathfindingBufferIndex(grid2D, info, neighborId);
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
            block.sync();
        }

        if (block.thread_rank() == 0) {
            // Retrieve path
            int4 originNeighbors = neighborCells(origin, grid2D);

            int min = uint32_t(Infinity);
            int minId = -1;
            Direction dir;
            for (int i = 0; i < 4; ++i) {
                int nId = ((int *)(&originNeighbors))[i];
                if (grid2D->getTileId(nId) == ROAD) {
                    int bufferId = pathfindingBufferIndex(grid2D, info, nId);
                    if (info.buffer[bufferId].distance < min) {
                        min = info.buffer[bufferId].distance;
                        minId = nId;
                        dir = Direction(i);
                    }
                }
            }

            entities->setPathDir(info.entityIdx, dir, 0);
            entities->setPathLength(info.entityIdx, 1);
        }
    }
}

void updateGrid(Grid2D *grid2D, Entities *entities) {
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
            int id = grid2D->cellAtPosition(float2{pos_W.x, pos_W.y});

            if (id != -1 && grid2D->getTileId(id) == GRASS) {
                updateInfo.update = true;
                updateInfo.tileToUpdate = id;
                updateInfo.newTileId = (TileId)uniforms.modeId;
            }
        }

        if (updateInfo.update) {
            updateCell(grid2D, updateInfo);
        }
    }

    assignOneHouse(grid2D, entities);

    performPathFinding(grid2D, entities);
    updateEntities(grid2D, entities);

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
        Grid2D *grid2D = allocator->alloc<Grid2D *>(sizeof(Grid2D));
        *grid2D = Grid2D(numRows, numCols, cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(entitiesBuffer);

        updateGrid(grid2D, entities);
    }
}
