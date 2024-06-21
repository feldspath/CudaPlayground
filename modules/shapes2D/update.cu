#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "entities.h"
#include "helper_math.h"
#include "map.h"
#include "matrix_math.h"
#include "movement.cuh"
#include "pathfinding.h"

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
        if (uniforms.creativeMode) {
            map->setTileId(id, new_tile);
        } else if (tileCost(new_tile) <= gameState->playerMoney) {
            gameState->playerMoney -= tileCost(new_tile);
            map->setTileId(id, new_tile);
        }
    }
    grid.sync();

    if (map->getTileId(id) != new_tile) {
        // tile was not updated
        return;
    }

    switch (new_tile) {
    case ROAD: {
        int *cumulNeighborNetworksSizes = allocator->alloc<int *>(sizeof(int) * 5);
        int *neighborNetworks = allocator->alloc<int *>(sizeof(int) * 4);

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
                        neighborNetworks[i] = repr;
                        map->roadNetworkRepr(repr) = id;
                        continue;
                    }
                }
                neighborNetworksSizes[i] = 0;
                neighborNetworks[i] = -1;
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
                int network = neighborNetworks[i];
                if (map->roadNetworkRepr(cellId) == network || cellId == network) {
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
    case SHOP:
        if (grid.thread_rank() == 0) {
            // Set capacity
            *map->shopTileData(id) = SHOP_CAPACITY;
        }
        break;
    default:
        break;
    }
}

void assignHouseToWorkplace(Map *map, Entities *entities, int32_t houseId, int32_t workplaceId) {
    int32_t newEntity = entities->newEntity(map->getCellPosition(houseId), houseId, workplaceId);
    int32_t *houseData = map->houseTileData(houseId);
    *houseData = newEntity;

    if (map->getTileId(workplaceId) == FACTORY) {
        *map->factoryTileData(workplaceId) -= 1;
    } else if (map->getTileId(workplaceId) == SHOP) {
        *map->shopTileData(workplaceId) -= 1;
    }
}

void assignOneHouse(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    uint32_t &unassignedHouseCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &availableWorkplaceCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &globalHouseIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &globalWorkplaceIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        unassignedHouseCount = 0;
        availableWorkplaceCount = 0;
        globalHouseIdx = 0;
        globalWorkplaceIdx = 0;
        assigned = 0;
    }

    grid.sync();

    map->processEachCell(HOUSE | FACTORY | SHOP, [&](int cellId) {
        if (map->getTileId(cellId) == HOUSE && *map->houseTileData(cellId) == -1) {
            atomicAdd(&unassignedHouseCount, 1);
        } else if ((map->getTileId(cellId) == FACTORY && *map->factoryTileData(cellId) > 0) ||
                   (map->getTileId(cellId) == SHOP && *map->shopTileData(cellId) > 0)) {
            atomicAdd(&availableWorkplaceCount, 1);
        }
    });

    grid.sync();

    if (unassignedHouseCount == 0 || availableWorkplaceCount == 0) {
        return;
    }

    uint32_t *availableWorkplaces =
        allocator->alloc<uint32_t *>(sizeof(uint32_t) * availableWorkplaceCount);
    uint32_t *unassignedHouses =
        allocator->alloc<uint32_t *>(sizeof(uint32_t) * unassignedHouseCount);

    map->processEachCell(HOUSE | FACTORY | SHOP, [&](int cellId) {
        if (map->getTileId(cellId) == HOUSE && *map->houseTileData(cellId) == -1) {
            int idx = atomicAdd(&globalHouseIdx, 1);
            unassignedHouses[idx] = cellId;
        } else if ((map->getTileId(cellId) == FACTORY && *map->factoryTileData(cellId) > 0) ||
                   (map->getTileId(cellId) == SHOP && *map->shopTileData(cellId) > 0)) {
            int idx = atomicAdd(&globalWorkplaceIdx, 1);
            availableWorkplaces[idx] = cellId;
        }
    });

    grid.sync();

    __shared__ uint64_t targetWorkplace;
    for (int gridOffset = 0; gridOffset < unassignedHouseCount; gridOffset += grid.num_blocks()) {
        int hIdx = gridOffset + grid.block_rank();
        if (hIdx >= unassignedHouseCount) {
            break;
        }

        int houseId = unassignedHouses[hIdx];

        // Get neighbor networks
        auto houseNets = map->neighborNetworks(houseId);
        int2 tileCoords = map->cellCoords(houseId);

        if (block.thread_rank() == 0) {
            targetWorkplace = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        // Check all tiles for factories
        for (int blockOffset = 0; blockOffset < availableWorkplaceCount;
             blockOffset += block.num_threads()) {
            int fIdx = block.thread_rank() + blockOffset;
            if (fIdx >= availableWorkplaceCount) {
                break;
            }
            int workplaceId = availableWorkplaces[fIdx];

            // Get the networks the factory is connected to
            auto factoryNets = map->neighborNetworks(workplaceId);
            if (map->sharedNetworks(factoryNets, houseNets).data[0] != -1) {
                // This factory shares the same network
                int2 workplaceCoords = map->cellCoords(workplaceId);
                int2 diff = workplaceCoords - tileCoords;
                uint32_t distance = abs(diff.x) + abs(diff.y);
                uint64_t target = (uint64_t(distance) << 32ull) | uint64_t(workplaceId);
                // keep the closest factory
                atomicMin(&targetWorkplace, target);
                break;
            }
        }

        block.sync();

        if (block.thread_rank() == 0) {
            int32_t *houseData = map->houseTileData(houseId);
            if (targetWorkplace != uint64_t(Infinity) << 32ull && !atomicAdd(&assigned, 1)) {
                int32_t workplaceId = targetWorkplace & 0xffffffffull;
                assignHouseToWorkplace(map, entities, houseId, workplaceId);
            } else {
                *houseData = -1;
            }
        }

        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }

void updateEntitiesState(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Each thread handles an entity
    for (int offset = 0; offset < entities->getCount(); offset += grid.num_threads()) {
        int entityIndex = offset + grid.thread_rank();
        if (entityIndex >= entities->getCount()) {
            break;
        }
        Entity &entity = entities->get(entityIndex);

        switch (entity.state) {
        case GoHome: {
            if (map->cellAtPosition(entity.position) == entity.houseId) {
                entity.path.reset();
                entity.state = Rest;
                entity.stateStart_ms = gameState->currentTime_ms;
                entity.position = map->getCellPosition(entity.houseId);
            }
            break;
        }
        case GoToWork: {
            if (map->cellAtPosition(entity.position) == entity.workplaceId) {
                entity.path.reset();
                entity.state = Work;
                entity.stateStart_ms = gameState->currentTime_ms;
                entity.position = map->getCellPosition(entity.workplaceId);
            }
            break;
        }
        case Work:
            if (gameState->currentTime_ms - entity.stateStart_ms >= WORK_TIME_MS) {
                entity.state = GoHome;
                atomicAdd(&gameState->playerMoney, 10);
            }
            break;
        case Rest:
            if (gameState->currentTime_ms - entity.stateStart_ms >= REST_TIME_MS) {
                entity.state = GoToWork;
            }
            break;
        default:
            break;
        }
    }
}

void updateGameState(Entities *entities) {
    gameState->dt = ((float)(nanotime_start - gameState->previousFrameTime_ns)) / 1e9;
    gameState->previousFrameTime_ns = nanotime_start;
    gameState->currentTime_ms = currentTime_ms();
    gameState->population = *entities->count;
}

template <typename Function> void printDuration(char *name, Function &&f) {
    if (!uniforms.printTimings) {
        f();
        return;
    }

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    uint64_t t_start = nanotime();

    f();

    grid.sync();

    uint64_t t_end = nanotime();

    if (grid.thread_rank() == 0) {
        double nanos = double(t_end) - double(t_start);
        float millis = nanos / 1e6;
        printf("%s: %f ms\n", name, millis);
    }
}

void updateGrid(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    nanotime_start = nanotime();

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

    printDuration("assignOneHouse", [&]() { assignOneHouse(map, entities); });
    printDuration("performPathFinding", [&]() { performPathFinding(map, entities, allocator); });
    printDuration("fillCells", [&]() { fillCells(map, entities); });
    printDuration("moveEntities", [&]() { moveEntities(map, entities, allocator, gameState->dt); });
    printDuration("updateEntitiesState", [&]() { updateEntitiesState(map, entities); });

    // grid.sync();
    if (grid.thread_rank() == 0) {
        updateGameState(entities);
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
