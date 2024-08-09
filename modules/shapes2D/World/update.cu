#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "common/helper_math.h"
#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/Entities/entities.cuh"
#include "World/Entities/movement.cuh"
#include "World/Path/pathfinding.cuh"
#include "World/cell_buffer.cuh"
#include "World/map.cuh"
#include "World/time.h"
#include "builtin_types.h"
#include "common/matrix_math.h"

namespace cg = cooperative_groups;

GameData gameData;
Allocator *allocator;
PathfindingManager *pathfindingManager;
uint64_t nanotime_start;

CellBuffer shops;
CellBuffer houses;
CellBuffer factories;
CellBuffer workplaces;

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
        if (gameData.uniforms.creativeMode) {
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
            map->getTyped<FactoryCell>(cellId) = FactoryCell();
        }

        map->processEachCell(UNKNOWN, [&](int cellId) {
            auto diff = map->cellCoords(cellId) - map->cellCoords(cellId);
            int dist = length(make_float2(diff));
            if (dist < 20) {
                map->get(cellId).landValue = max(map->get(cellId).landValue - 20 + int(dist), 0);
            }
        });

        break;
    }
    case HOUSE:
        if (grid.thread_rank() == 0) {
            map->getTyped<HouseCell>(cellId) = HouseCell();
        }
        break;
    case SHOP:
        if (grid.thread_rank() == 0) {
            map->getTyped<ShopCell>(cellId) = ShopCell();
        }
        break;
    case GRASS: {
        switch (oldTile) {
        case HOUSE: {
            // if it was a house, destroy the entities living there
            entities->processAll([&](int entityId) {
                if (entities->get(entityId).houseId == cellId) {
                    int workplaceId = entities->get(entityId).workplaceId;
                    atomicAdd(&map->workplaceCapacity(workplaceId), 1);
                    entities->remove(entityId);
                }
            });
            break;
        }
        case FACTORY: {
            // if the removed tile was a factory or shop, destroy the associated entities
            entities->processAll([&](int entityId) {
                auto &entity = entities->get(entityId);
                if (entity.workplaceId == cellId) {
                    map->getTyped<HouseCell>(entity.houseId) = HouseCell();
                    entities->remove(entityId);
                }
            });
            break;
        }
        case SHOP: {
            // if the removed tile was a factory or shop, destroy the associated entities
            entities->processAll([&](int entityId) {
                auto &entity = entities->get(entityId);
                if (entity.workplaceId == cellId) {
                    map->getTyped<HouseCell>(entity.houseId) = HouseCell();
                    entities->remove(entityId);
                }
            });
            break;
        }
        case ROAD: {
            // if it was a road, refresh the networks data
            int32_t network = map->roadNetworkRepr(cellId);

            bool *validCells = allocator->alloc<bool *>(sizeof(bool) * map->getCount());

            map->processEachCell(ROAD, [&](int cellId) { validCells[cellId] = true; });

            // reset relevant network
            map->processEachCell(ROAD, [&](int cellId) {
                if (map->roadNetworkRepr(cellId) == network) {
                    map->roadNetworkRepr(cellId) = cellId;
                    validCells[cellId] = false;
                }
            });
            grid.sync();

            entities->processAll([&](int entityId) {
                auto &entity = entities->get(entityId);
                if (!validCells[map->cellAtPosition(entity.position)]) {
                    entity.path.reset();
                }
            });

            // recompute connected components
            // https://largo.lip6.fr/~lacas/Publications/IPTA17.pdf
            bool &changed = *allocator->alloc<bool *>(sizeof(bool));
            for (int i = 0; i < 10; ++i) {
                if (grid.thread_rank() == 0) {
                    changed = false;
                }
                grid.sync();
                map->processEachCell(ROAD, [&](int cellId) {
                    if (validCells[cellId]) {
                        return;
                    }
                    int m = map->neighborNetworks(cellId).min();
                    int old = atomicMin(&map->roadNetworkRepr(map->roadNetworkRepr(cellId)), m);
                    if (m < old) {
                        changed = true;
                    }
                });
                grid.sync();
                if (!changed) {
                    // if (grid.thread_rank() == 0) {
                    //     printf("ended in %d iterations\n", i);
                    // }
                    break;
                }
                map->processEachCell(ROAD, [&](int cellId) {
                    if (validCells[cellId]) {
                        return;
                    }
                    int network = map->roadNetworkRepr(cellId);
                    while (network != map->roadNetworkRepr(network)) {
                        network = map->roadNetworkRepr(network);
                    }
                    map->roadNetworkRepr(cellId) = network;
                });
            }

            // recompute network ids
            // count the different unique networks
            int &uniqueNetworksCount = *allocator->alloc<int32_t *>(sizeof(int32_t));
            if (grid.thread_rank() == 0) {
                uniqueNetworksCount = 0;
            }
            grid.sync();
            map->processEachCell(ROAD, [&](int cellId) {
                if (validCells[cellId]) {
                    return;
                }
                int net = map->roadNetworkRepr(cellId);
                if (cellId == net) {
                    map->roadNetworkId(cellId) = atomicAdd(&uniqueNetworksCount, 1);
                }
            });
            grid.sync();

            // variable for each unique network
            int *networkIds = allocator->alloc<int *>(sizeof(int) * uniqueNetworksCount);
            if (grid.thread_rank() == 0) {
                for (int i = 0; i < uniqueNetworksCount; ++i) {
                    networkIds[i] = 1;
                }
            }
            grid.sync();

            // assign the ids
            map->processEachCell(ROAD, [&](int cellId) {
                if (validCells[cellId]) {
                    return;
                }
                int net = map->roadNetworkRepr(cellId);
                if (cellId != net) {
                    map->roadNetworkId(cellId) = atomicAdd(&networkIds[map->roadNetworkId(net)], 1);
                }
            });
            grid.sync();

            map->processEachCell(ROAD, [&](int cellId) {
                if (validCells[cellId]) {
                    return;
                }
                int net = map->roadNetworkRepr(cellId);
                if (cellId == net) {
                    map->roadNetworkId(cellId) = networkIds[map->roadNetworkId(net)];
                }
            });
            break;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

void assignHouseToWorkplace(Map *map, Entities *entities, int32_t houseId, int32_t workplaceId) {
    int32_t newEntity = entities->newEntity(map->getCellPosition(houseId), houseId, workplaceId);
    map->getTyped<HouseCell>(houseId).residentCount += 1;
    map->getTyped<WorkplaceCell>(workplaceId).workplaceCapacity -= 1;
}

void assignOneHouse(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();
    int32_t &assigned = *allocator->alloc<int32_t *>(sizeof(int32_t));
    if (grid.thread_rank() == 0) {
        assigned = 0;
    }
    grid.sync();

    CellBuffer unassignedHouses = houses.subBuffer(allocator, [&](int cellId) {
        HouseCell &house = map->getTyped<HouseCell>(cellId);
        return house.residentCount < house.maxResidents();
    });
    grid.sync();
    if (unassignedHouses.getCount() == 0) {
        return;
    }

    CellBuffer availableWorkplaces = workplaces.subBuffer(allocator, [&](int cellId) {
        return map->getTyped<WorkplaceCell>(cellId).workplaceCapacity > 0;
    });
    grid.sync();
    if (availableWorkplaces.getCount() == 0) {
        return;
    }

    unassignedHouses.processEachCell_blockwise([&](int houseCellId) {
        if (assigned) {
            return;
        }
        int32_t workplaceId =
            workplaces.findClosestOnNetworkBlockwise(*map, houseCellId, [&](int cellId) {
                return map->getTyped<WorkplaceCell>(cellId).workplaceCapacity > 0;
            });

        if (block.thread_rank() == 0 && workplaceId != -1 && !atomicAdd(&assigned, 1)) {
            assignHouseToWorkplace(map, entities, houseCellId, workplaceId);
        }
    });
}

void assignOneCustomerToShop(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.block_rank() == 0) {
        __shared__ bool assigned;
        if (block.thread_rank() == 0) {
            assigned = false;
        }
        block.sync();
        for (int entityIdx = 0; entityIdx < entities->getCount(); entityIdx++) {
            if (assigned) {
                return;
            }
            Entity &entity = entities->get(entityIdx);
            if (!entity.active || entity.state != GoShopping || entity.destination != -1) {
                continue;
            }

            int32_t shopId = shops.findClosestOnNetworkBlockwise(
                *map, map->cellAtPosition(entity.position),
                [&](int cellId) { return map->getTyped<ShopCell>(cellId).woodCount > 0; });

            if (block.thread_rank() == 0) {
                if (shopId != -1) {
                    entity.destination = shopId;
                    assigned = true;
                } else {
                    entity.changeState(GoHome);
                }
            }
            block.sync();
            if (assigned) {
                return;
            }
        }
    }
}

void assignShopWorkerToFactory(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.block_rank() == 0) {
        __shared__ bool assigned;
        if (block.thread_rank() == 0) {
            assigned = false;
        }
        block.sync();
        for (int entityIdx = 0; entityIdx < entities->getCount(); entityIdx++) {
            if (assigned) {
                return;
            }
            Entity &entity = entities->get(entityIdx);
            if (!entity.active || entity.state != WorkAtShop || entity.destination != -1) {
                continue;
            }

            int32_t factoryId = factories.findClosestOnNetworkBlockwise(
                *map, map->cellAtPosition(entity.position),
                [&](int cellId) { return map->getTyped<FactoryCell>(cellId).stockCount > 0; });

            if (block.thread_rank() == 0) {
                if (factoryId != -1) {
                    entity.destination = factoryId;
                    assigned = true;
                }
            }
            block.sync();
            if (assigned) {
                return;
            }
        }
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
        entity.checkWaitStatus();

        if (!entity.active) {
            return;
        }

        bool destinationReached = false;
        if (entity.destination != -1 &&
            map->cellAtPosition(entity.position) == entity.destination) {
            destinationReached = true;
            entity.position = map->getCellPosition(entity.destination);
            entity.velocity = float2{0.0f, 0.0f};
        }

        TimeInterval workHours = TimeInterval::workHours;

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
            if (entity.destination == -1) {
                entity.destination = entity.workplaceId;
                break;
            }

            if (destinationReached) {
                atomicAdd(&map->getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount, 1);
                switch (map->getTileId(entity.workplaceId)) {
                case SHOP: {
                    entity.changeState(WorkAtShop);
                    break;
                }
                case FACTORY: {
                    entity.changeState(WorkAtFactory);
                    break;
                }
                default:
                }
            }
            break;
        }
        case GoShopping: {
            if (!TimeInterval::upgradeHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoHome);
                break;
            }
            // entity destination is not handled here
            if (destinationReached) {
                entity.changeState(Shop);
            }
            break;
        }
        case WorkAtFactory: {
            if (!workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoShopping);
                atomicAdd(&map->getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount, -1);
                break;
            }
            atomicAdd(&map->getTyped<FactoryCell>(entity.workplaceId).stockCount, 1);
            entity.wait(30);
            break;
        }
        case WorkAtShop: {
            if (!workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoShopping);
                atomicAdd(&map->getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount, -1);
                break;
            }
            if (destinationReached && entity.destination == entity.workplaceId) {
                atomicAdd(&map->getTyped<ShopCell>(entity.workplaceId).woodCount, entity.inventory);
                entity.inventory = 0;
                entity.destination = -1;
                entity.wait(10);
            } else if (destinationReached && map->getTileId(entity.destination) == FACTORY) {
                entity.inventory += min(map->getTyped<FactoryCell>(entity.destination).stockCount,
                                        SHOP_WORKER_INVENTORY_SIZE);
                entity.destination = entity.workplaceId;
                entity.wait(10);
            }
            break;
        }
        case Rest:
            if (workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoToWork);
            }
            break;
        case Shop: {
            if (!TimeInterval::upgradeHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoHome);
            }

            int shopId = map->cellAtPosition(entity.position);
            int32_t &shopWood = map->getTyped<ShopCell>(shopId).woodCount;

            int stock = 5;
            int old = atomicSub(&shopWood, stock);

            if (old - stock < 0) {
                int eq = min(-(old - stock), stock);
                atomicAdd(&shopWood, eq);
                stock -= eq;
            }
            if (stock > 0) {
                entity.inventory += stock;
                entity.changeState(UpgradeHouse);
                entity.destination = entity.houseId;
                entity.wait(10);
            }
            break;
        }
        case UpgradeHouse: {
            if (destinationReached) {
                atomicAdd(&map->getTyped<HouseCell>(entity.houseId).woodCount, entity.inventory);
                entity.inventory = 0;
                entity.changeState(GoShopping);
            }
            break;
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
        GameState::instance->previousMouseButtons = gameData.uniforms.mouseButtons;
        GameState::instance->previousGameTime = GameState::instance->gameTime;

        if (GameState::instance->firstFrame) {
            GameState::instance->firstFrame = false;
            GameState::instance->gameTime.dt = 0.0f;
        } else {
            GameState::instance->gameTime.incrementRealTime(dt * gameData.uniforms.timeMultiplier);
        }
    }
}

template <typename Function> void printDuration(char *name, Function &&f) {
    if (!gameData.uniforms.printTimings) {
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

    if (gameData.uniforms.cursorPos.x >= 0 &&
        gameData.uniforms.cursorPos.x < gameData.uniforms.width &&
        gameData.uniforms.cursorPos.y >= 0 &&
        gameData.uniforms.cursorPos.y < gameData.uniforms.height) {
        UpdateInfo updateInfo;

        bool mouseClicked = (gameData.uniforms.mouseButtons & 1) &
                            ((~GameState::instance->previousMouseButtons) & 1);
        bool mousePressed = gameData.uniforms.mouseButtons & 1;
        updateInfo.update = false;

        float2 px = float2{gameData.uniforms.cursorPos.x,
                           gameData.uniforms.height - gameData.uniforms.cursorPos.y};
        float3 pos_W = unproject(px, gameData.uniforms.invview * gameData.uniforms.invproj,
                                 gameData.uniforms.width, gameData.uniforms.height);
        int id = map->cellAtPosition(float2{pos_W.x, pos_W.y});

        if (mouseClicked && (TileId)gameData.uniforms.modeId != GRASS) {
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
            if (id != -1 &&
                (map->getTileId(id) == GRASS || (TileId)gameData.uniforms.modeId == GRASS)) {
                updateInfo.update = true;
                updateInfo.cellToUpdate = id;
                updateInfo.newTileId = (TileId)gameData.uniforms.modeId;
            }
        }
        if (updateInfo.update) {
            updateCell(map, entities, updateInfo);
        }
    }
}

void fillCellBuffers(Map *map) {
    auto grid = cg::this_grid();
    workplaces.fill(*map, allocator, [&](int cellId) { return map->isWorkplace(cellId); });
    factories.fill(*map, allocator, [&](int cellId) { return map->getTileId(cellId) == FACTORY; });
    shops.fill(*map, allocator, [&](int cellId) { return map->getTileId(cellId) == SHOP; });
    houses.fill(*map, allocator, [&](int cellId) { return map->getTileId(cellId) == HOUSE; });
}

void handleEvents(Map *map, Entities *entities) {
    auto tod = GameState::instance->gameTime.timeOfDay();
    auto prevTod = GameState::instance->previousGameTime.timeOfDay();

    TOD sellResourcesTime{23, 0};
    if (prevTod <= sellResourcesTime && sellResourcesTime <= tod) {
        // sell all resources at 23:00
        shops.processEachCell([&](int cellId) {
            ShopCell &shop = map->getTyped<ShopCell>(cellId);
            atomicAdd(&GameState::instance->playerMoney, shop.woodCount * WOOD_SELL_PRICE);
            shop.woodCount = 0;
        });
    }

    houses.processEachCell([&](int cellId) {
        HouseCell &house = map->getTyped<HouseCell>(cellId);
        if (house.woodCount >= house.upgradeCost()) {
            house.levelUp();
        }
    });
}

void updateGrid(Map *map, Entities *entities) {
    auto grid = cg::this_grid();

    nanotime_start = nanotime();

    if (gameData.uniforms.printTimings && cg::this_grid().thread_rank() == 0) {
        printf("================================\n");
    }

    printDuration("handleInputs                ", [&]() { handleInputs(map, entities); });
    grid.sync();
    printDuration("fillCellBuffers             ", [&]() { fillCellBuffers(map); });
    grid.sync();
    printDuration("fillCells                   ", [&]() { fillCells(map, entities); });
    grid.sync();
    printDuration("handleEvents                ", [&]() { handleEvents(map, entities); });
    printDuration("assignOneHouse              ", [&]() { assignOneHouse(map, entities); });
    printDuration("assignOneCustomerToShop     ",
                  [&]() { assignOneCustomerToShop(map, entities); });
    printDuration("assignShopWorkerToFactory   ",
                  [&]() { assignShopWorkerToFactory(map, entities); });
    printDuration("pathfinding                 ",
                  [&]() { pathfindingManager->update(*map, *entities, *allocator); });
    grid.sync();
    printDuration("moveEntities                ", [&]() {
        moveEntities(map, entities, allocator, GameState::instance->gameTime.getDt());
    });
    grid.sync();
    printDuration("updateEntitiesState         ", [&]() { updateEntitiesState(map, entities); });
    grid.sync();
    // printDuration("entitiesInteractions        ", [&]() { entitiesInteractions(map, entities);
    // });
    printDuration("updateGameState             ", [&]() { updateGameState(entities); });
}

extern "C" __global__ void update(GameData _gameData) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    gameData = _gameData;
    GameState::instance = gameData.state;

    Allocator _allocator(gameData.buffer, 0);
    allocator = &_allocator;

    PathfindingManager _pathfindingManager(gameData.pathfindingBuffer);
    pathfindingManager = &_pathfindingManager;

    curand_init(grid.thread_rank() + GameState::instance->currentTime_ms, 0, 0,
                &thread_random_state);

    grid.sync();

    {
        Map *map = allocator->alloc<Map *>(sizeof(Map));
        *map = Map(gameData.numRows, gameData.numCols, gameData.cells);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(gameData.entitiesBuffer);

        updateGrid(map, entities);
    }
}
