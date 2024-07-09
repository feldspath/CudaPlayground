#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/Entities/entities.cuh"
#include "World/Entities/movement.cuh"
#include "World/Path/pathfinding.cuh"
#include "World/map.cuh"
#include "World/time.h"
#include "builtin_types.h"
#include "common/helper_math.h"
#include "common/matrix_math.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
Allocator *allocator;
uint64_t nanotime_start;

curandStateXORWOW_t thread_random_state;

struct UpdateInfo {
    bool update;
    int cellToUpdate;
    TileId newTileId;
};

void updateCell(Map *map, Entities *entities, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int cellId = updateInfo.cellToUpdate;

    TileId newTile = updateInfo.newTileId;
    TileId oldTile = map->getTileId(cellId);

    grid.sync();
    if (grid.thread_rank() == 0) {
        if (uniforms.creativeMode) {
            map->setTileId(cellId, newTile);
        } else if (tileCost(newTile) <= GameState::instance->playerMoney) {
            GameState::instance->playerMoney -= tileCost(newTile);
            map->setTileId(cellId, newTile);
        }
    }
    grid.sync();

    if (map->getTileId(cellId) == oldTile) {
        // tile was not updated
        return;
    }

    switch (newTile) {
    case ROAD: {
        int *cumulNeighborNetworksSizes =
            allocator->alloc<int *>(sizeof(int) * (Neighbors::size() + 1));
        int *neighborNetworks = allocator->alloc<int *>(sizeof(int) * Neighbors::size());

        if (grid.thread_rank() == 0) {
            // check nearby tiles.
            auto neighbors = map->neighborCells(cellId);
            int neighborNetworksSizes[Neighbors::size()];

            for (int i = 0; i < Neighbors::size(); i++) {
                int nId = neighbors.data[i];
                // if one tile is not grass, update the connected components
                if (nId != -1 && map->getTileId(nId) == ROAD) {
                    int repr = map->roadNetworkRepr(nId);
                    // Skip the tile if it was already updated this frame
                    if (map->roadNetworkRepr(repr) == repr) {
                        neighborNetworksSizes[i] = map->roadNetworkId(repr);
                        neighborNetworks[i] = repr;
                        map->roadNetworkRepr(repr) = cellId;
                        continue;
                    }
                }
                neighborNetworksSizes[i] = 0;
                neighborNetworks[i] = -1;
            }

            cumulNeighborNetworksSizes[0] = 0;
            for (int i = 0; i < Neighbors::size(); i++) {
                cumulNeighborNetworksSizes[i + 1] =
                    cumulNeighborNetworksSizes[i] + neighborNetworksSizes[i];
            }

            // Init the new road tile
            map->roadNetworkRepr(cellId) = cellId;
            map->roadNetworkId(cellId) = cumulNeighborNetworksSizes[Neighbors::size()] + 1;
        }

        grid.sync();

        // Flatten network
        map->processEachCell(ROAD, [&](int otherCellId) {
            int neighborId = -1;
            for (int i = 0; i < Neighbors::size(); ++i) {
                int network = neighborNetworks[i];
                if (map->roadNetworkRepr(otherCellId) == network || otherCellId == network) {
                    neighborId = i;
                    break;
                }
            }
            if (neighborId == -1) {
                return;
            }

            map->roadNetworkRepr(otherCellId) = cellId;
            map->roadNetworkId(otherCellId) += cumulNeighborNetworksSizes[neighborId];
        });
        break;
    }
    case FACTORY: {

        if (grid.thread_rank() == 0) {
            // Set capacity
            *map->factoryTileData(cellId) = FACTORY_CAPACITY;
        }

        processRange(map->count, [&](int cellId) {
            auto diff = map->cellCoords(cellId) - map->cellCoords(cellId);
            int dist = length(make_float2(diff));
            if (dist < 20) {
                map->cellsData[cellId].landValue =
                    max(map->cellsData[cellId].landValue - 20 + int(dist), 0);
            }
        });

        break;
    }
    case HOUSE:
        if (grid.thread_rank() == 0) {
            // Set house to unassigned
            *map->houseTileData(cellId) = -1;
        }
        break;
    case SHOP:
        if (grid.thread_rank() == 0) {
            // Set capacities
            map->workplaceCapacity(cellId) = SHOP_WORK_CAPACITY;
            map->shopCurrentWorkerCount(cellId) = 0;
        }
        break;
    case GRASS: {
        switch (oldTile) {
        case HOUSE: {
            // if it was a house, destroy the entity living there
            if (grid.thread_rank() == 0) {
                int entityId = *map->houseTileData(cellId);
                if (entityId == -1) {
                    break;
                }
                int workplaceId = entities->get(entityId).workplaceId;
                map->workplaceCapacity(workplaceId)++;
                entities->remove(entityId);
            }
            break;
        }
        case FACTORY:
        case SHOP: {
            // if the removed tile was a factory or shop, destroy the associated entities
            entities->processAll([&](int entityId) {
                auto &entity = entities->get(entityId);
                if (entity.workplaceId == cellId) {
                    map->workplaceCapacity(entity.workplaceId)++;
                    entities->remove(entityId);
                }
            });
        }
        default:
            break;
        }

        // if it was a road, refresh the networks data:
        //
        break;
    }
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
    for_blockwise(unassignedHouseCount, [&](int hIdx) {
        if (assigned) {
            return;
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
        processRangeBlock(availableWorkplaceCount, [&](int fIdx) {
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
            }
        });

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
    });
}

void assignOneCustomerToShop(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    uint32_t &availableShopsCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &globalShopIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        assigned = 0;
        availableShopsCount = 0;
        globalShopIdx = 0;
    }

    grid.sync();

    map->processEachCell(SHOP, [&](int cellId) {
        // Look for open shop
        if (map->shopCurrentWorkerCount(cellId) > 0) {
            atomicAdd(&availableShopsCount, 1);
        }
    });

    grid.sync();

    if (availableShopsCount == 0) {
        return;
    }

    uint32_t *availableShops = allocator->alloc<uint32_t *>(sizeof(uint32_t) * availableShopsCount);
    uint32_t *reachableShops = allocator->alloc<uint32_t *>(sizeof(uint32_t) * availableShopsCount);

    map->processEachCell(SHOP, [&](int cellId) {
        // Look for open shop
        if (map->shopCurrentWorkerCount(cellId) > 0) {
            int idx = atomicAdd(&globalShopIdx, 1);
            availableShops[idx] = cellId;
        }
    });

    grid.sync();

    for (int entityIdx = 0; entityIdx < entities->getCount(); entityIdx++) {
        Entity &entity = entities->get(entityIdx);
        if (!entity.active || entity.state != GoShopping || entity.destination != -1) {
            continue;
        }

        if (grid.thread_rank() == 0) {
            globalShopIdx = 0;
        }
        grid.sync();

        int entityCell = map->cellAtPosition(entity.position);

        // Get neighbor networks
        auto entityNets = map->neighborNetworks(entityCell);
        int2 tileCoords = map->cellCoords(entityCell);

        processRange(availableShopsCount, [&](int sIdx) {
            int shopId = availableShops[sIdx];

            // Get the networks the shop is connected to
            auto shopNets = map->neighborNetworks(shopId);
            if (map->sharedNetworks(shopNets, entityNets).data[0] != -1) {
                // This factory shares the same network
                int bufferIdx = atomicAdd(&globalShopIdx, 1);
                reachableShops[bufferIdx] = shopId;
            }
        });

        grid.sync();

        if (globalShopIdx == 0) {
            // No reachable shop
            continue;
        }

        if (grid.thread_rank() == 0) {
            uint32_t randomShopIdx = curand(&thread_random_state) % globalShopIdx;
            entity.destination = reachableShops[randomShopIdx];
        }
        break;
    }
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }

void updateEntitiesState(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto gameTime = GameState::instance->gameTime;

    // Each thread handles an entity
    entities->processAll([&](int entityIndex) {
        Entity &entity = entities->get(entityIndex);

        bool destinationReached = false;
        if (entity.destination != -1 &&
            map->cellAtPosition(entity.position) == entity.destination) {
            destinationReached = true;
            entity.position = map->getCellPosition(entity.destination);
            entity.happiness = max(
                int(entity.happiness) - (gameTime.minutesElapsedSince(entity.stateStart)) / 10, 0);
        }

        TimeInterval workHours;
        if (map->getTileId(entity.workplaceId) == SHOP) {
            workHours = TimeInterval::shopHours;
        } else if (map->getTileId(entity.workplaceId) == FACTORY) {
            workHours = TimeInterval::factoryHours;
        }

        switch (entity.state) {
        case GoHome: {
            if (destinationReached) {
                entity.changeState(Rest);
            } else if (entity.destination == -1) {
                entity.destination = entity.houseId;
            }
            break;
        }
        case GoToWork: {
            if (!workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoShopping);
                break;
            }

            if (destinationReached) {
                entity.changeState(Work);
                if (map->getTileId(entity.workplaceId) == SHOP) {
                    atomicAdd(&map->shopCurrentWorkerCount(entity.workplaceId), 1);
                }

            } else if (entity.destination == -1) {
                entity.destination = entity.workplaceId;
            }
            break;
        }
        case GoShopping: {
            if (!TimeInterval::shopHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoHome);
                break;
            }
            // entity destination is not handled here
            if (destinationReached) {
                entity.changeState(Shop);
            } else if (!TimeInterval::shopHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoHome);
            }
            break;
        }
        case Work:
            if (!workHours.contains(gameTime.timeOfDay())) {
                int moneyEarned = gameTime.minutesElapsedSince(entity.stateStart) / 7;
                int taxes = moneyEarned / 10;
                int rent = map->rentCost(entity.houseId);
                atomicAdd(&GameState::instance->playerMoney, rent + taxes);
                entity.money += max(moneyEarned - rent, 0);
                entity.changeState(GoShopping);

                if (map->getTileId(entity.workplaceId) == SHOP) {
                    atomicAdd(&map->shopCurrentWorkerCount(entity.workplaceId), -1);
                }
            }
            break;
        case Rest:
            if (workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoToWork);
            }
            break;
        case Shop:
            // waiting to be handled by a shop worker
            if (!TimeInterval::shopHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoHome);
            }
            break;
        default:
            break;
        }
    });
}

void entitiesInteractions(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    // Update interactions
    entities->processAll([&](int entityIndex) {
        Entity &entity = entities->get(entityIndex);

        switch (entity.state) {
        case Work: {
            if (map->getTileId(entity.workplaceId) == SHOP) {
                if (entity.interaction == -1) {
                    for (int i = 0; i < ENTITIES_PER_CELL; ++i) {
                        int otherIndex = map->cellsData[entity.workplaceId].entities[i];
                        if (otherIndex == -1) {
                            break;
                        }
                        auto &other = entities->get(otherIndex);
                        if (other.state == Shop &&
                            atomicCAS(&other.interaction, -1, entityIndex) == -1) {
                            entity.interaction = otherIndex;
                            entity.resetStateStart();
                        }
                    }
                } else if ((GameState::instance->gameTime.minutesElapsedSince(entity.stateStart)) >
                           SHOP_TIME_MIN) {
                    auto &other = entities->get(entity.interaction);
                    entity.interaction = -1;
                    other.interaction = -1;
                    other.changeState(GoHome);
                    other.happiness = min(other.happiness + other.money / 5, 255);
                    int tax = other.money / 10;
                    atomicAdd(&GameState::instance->playerMoney, tax);
                    other.money = 0;
                }
            }
        }
        default:
            break;
        }
    });
}

void updateGameState(Entities *entities) {
    auto grid = cg::this_grid();

    if (grid.thread_rank() == 0) {
        float dt = ((float)(nanotime_start - GameState::instance->previousFrameTime_ns)) / 1e9;
        GameState::instance->previousFrameTime_ns = nanotime_start;
        GameState::instance->currentTime_ms = currentTime_ms();
        GameState::instance->population = entities->getCount();
        GameState::instance->previousMouseButtons = uniforms.mouseButtons;

        if (GameState::instance->firstFrame) {
            GameState::instance->firstFrame = false;
            GameState::instance->gameTime.dt = 0.0f;
        } else {
            GameState::instance->gameTime.incrementRealTime(dt * uniforms.timeMultiplier);
        }
    }
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
        printf("%s: %8.3f ms\n", name, millis);
    }
}

void handleInputs(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (uniforms.cursorPos.x >= 0 && uniforms.cursorPos.x < uniforms.width &&
        uniforms.cursorPos.y >= 0 && uniforms.cursorPos.y < uniforms.height) {
        UpdateInfo updateInfo;

        bool mouseClicked =
            (uniforms.mouseButtons & 1) & ((~GameState::instance->previousMouseButtons) & 1);
        bool mousePressed = uniforms.mouseButtons & 1;
        updateInfo.update = false;

        float2 px = float2{uniforms.cursorPos.x, uniforms.height - uniforms.cursorPos.y};
        float3 pos_W =
            unproject(px, uniforms.invview * uniforms.invproj, uniforms.width, uniforms.height);
        int id = map->cellAtPosition(float2{pos_W.x, pos_W.y});

        if (mouseClicked && (TileId)uniforms.modeId != GRASS) {
            if (grid.thread_rank() == 0) {
                if (map->getTileId(id) & (HOUSE | FACTORY | SHOP)) {
                    if (GameState::instance->buildingDisplay == id) {
                        GameState::instance->buildingDisplay = -1;
                    } else {
                        GameState::instance->buildingDisplay = id;
                    }
                }
            }
        } else if (mousePressed) {
            if (id != -1 && (map->getTileId(id) == GRASS || (TileId)uniforms.modeId == GRASS)) {
                updateInfo.update = true;
                updateInfo.cellToUpdate = id;
                updateInfo.newTileId = (TileId)uniforms.modeId;
            }
        }
        if (updateInfo.update) {
            updateCell(map, entities, updateInfo);
        }
    }
}

void updateGrid(Map *map, Entities *entities) {
    nanotime_start = nanotime();

    if (uniforms.printTimings && cg::this_grid().thread_rank() == 0) {
        printf("================================\n");
    }

    printDuration("handleInputs                ", [&]() { handleInputs(map, entities); });
    printDuration("fillCells                   ", [&]() { fillCells(map, entities); });
    printDuration("assignOneHouse              ", [&]() { assignOneHouse(map, entities); });
    printDuration("assignOneCustomerToShop     ",
                  [&]() { assignOneCustomerToShop(map, entities); });
    printDuration("performPathFinding          ",
                  [&]() { performPathFinding(map, entities, allocator); });
    printDuration("moveEntities                ", [&]() {
        moveEntities(map, entities, allocator, GameState::instance->gameTime.getDt());
    });
    printDuration("updateEntitiesState         ", [&]() { updateEntitiesState(map, entities); });
    printDuration("entitiesInteractions        ", [&]() { entitiesInteractions(map, entities); });
    printDuration("updateGameState             ", [&]() { updateGameState(entities); });
}

extern "C" __global__ void update(const Uniforms _uniforms, GameState *_gameState,
                                  unsigned int *buffer, uint32_t numRows, uint32_t numCols,
                                  char *cells, void *entitiesBuffer) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    uniforms = _uniforms;

    Allocator _allocator(buffer, 0);
    allocator = &_allocator;

    GameState::instance = _gameState;

    curand_init(grid.thread_rank() + GameState::instance->currentTime_ms, 0, 0,
                &thread_random_state);

    grid.sync();

    {
        Map *map = allocator->alloc<Map *>(sizeof(Map));
        *map = Map(numRows, numCols, cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(entitiesBuffer);

        updateGrid(map, entities);
    }
}
