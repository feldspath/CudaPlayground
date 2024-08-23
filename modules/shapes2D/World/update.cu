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

curandStateXORWOW_t thread_random_state;

struct UpdateInfo {
    bool update;
    int cellToUpdate;
    TileId newTileId;
};

void updateCell(Chunk &chunk, Entities &entities, UpdateInfo updateInfo) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int cellId = updateInfo.cellToUpdate;

    TileId newTile = updateInfo.newTileId;
    TileId oldTile = chunk.get(cellId).tileId;

    grid.sync();
    if (grid.thread_rank() == 0) {
        if (gameData.uniforms.creativeMode) {
            chunk.get(cellId).tileId = newTile;
        } else if (tileCost(newTile) <= GameState::instance->playerMoney) {
            GameState::instance->playerMoney -= tileCost(newTile);
            chunk.get(cellId).tileId = newTile;
        }
    }
    grid.sync();

    if (chunk.get(cellId).tileId == oldTile) {
        // tile was not updated
        return;
    }

    switch (newTile) {
    case ROAD: {
        pathfindingManager->invalidateCache(chunk);
        entities.processAllActive([&](int entityId) { entities.get(entityId).path.reset(); });

        int *neighborNetworks = allocator->alloc<int *>(sizeof(int) * Neighbors::size());
        if (grid.thread_rank() == 0) {
            // check nearby tiles.
            auto neighbors = chunk.neighborCells(cellId);

            for (int i = 0; i < Neighbors::size(); i++) {
                int nId = neighbors.data[i];
                // if one tile is a road, update the connected components
                if (nId != -1 && chunk.get(nId).tileId == ROAD) {
                    int repr = chunk.roadNetworkRepr(nId);
                    // Skip the tile if it was already updated this frame
                    if (chunk.roadNetworkRepr(repr) == repr) {
                        neighborNetworks[i] = repr;
                        chunk.roadNetworkRepr(repr) = cellId;
                        continue;
                    }
                }
                neighborNetworks[i] = -1;
            }

            // Init the new road tile
            chunk.roadNetworkRepr(cellId) = cellId;
        }

        grid.sync();

        // Flatten network
        chunk.processEachCell(ROAD, [&](int otherCellId) {
            int neighborId = -1;
            for (int i = 0; i < Neighbors::size(); ++i) {
                int network = neighborNetworks[i];
                if (chunk.roadNetworkRepr(otherCellId) == network || otherCellId == network) {
                    neighborId = i;
                    break;
                }
            }
            if (neighborId == -1) {
                return;
            }

            chunk.roadNetworkRepr(otherCellId) = cellId;
        });
        break;
    }
    case FACTORY: {
        if (grid.thread_rank() == 0) {
            chunk.getTyped<FactoryCell>(cellId) = FactoryCell();
        }
        break;
    }
    case HOUSE:
        if (grid.thread_rank() == 0) {
            chunk.getTyped<HouseCell>(cellId) = HouseCell();
        }
        break;
    case SHOP:
        if (grid.thread_rank() == 0) {
            chunk.getTyped<ShopCell>(cellId) = ShopCell();
        }
        break;
    case GRASS: {
        switch (oldTile) {
        case HOUSE: {
            // if it was a house, destroy the entities living there
            entities.processAll([&](int entityId) {
                if (entities.get(entityId).houseId == cellId) {
                    int workplaceId = entities.get(entityId).workplaceId;
                    atomicAdd(&chunk.getTyped<WorkplaceCell>(workplaceId).workplaceCapacity, 1);
                    entities.remove(entityId);
                }
            });
            break;
        }
        case FACTORY: {
            // if the removed tile was a factory or shop, destroy the associated entities
            entities.processAll([&](int entityId) {
                auto &entity = entities.get(entityId);
                if (entity.workplaceId == cellId) {
                    atomicSub(&chunk.getTyped<HouseCell>(entity.houseId).residentCount, 1);
                    entities.remove(entityId);
                }
            });
            break;
        }
        case SHOP: {
            // if the removed tile was a factory or shop, destroy the associated entities
            entities.processAll([&](int entityId) {
                auto &entity = entities.get(entityId);
                if (entity.workplaceId == cellId) {
                    chunk.getTyped<HouseCell>(entity.houseId) = HouseCell();
                    entities.remove(entityId);
                }
            });
            break;
        }
        case ROAD: {
            // reset the entities path on this network
            int invalidNetwork = chunk.roadNetworkRepr(cellId);
            entities.processAll([&](int entityId) {
                auto &entity = entities.get(entityId);
                if (chunk.roadNetworkRepr(chunk.cellAtPosition(entity.position)) ==
                    invalidNetwork) {
                    entity.path.reset();
                }
            });

            // recompute relevant network
            chunk.updateNetworkComponents(invalidNetwork, *allocator);

            // invalidate pathfinding cache in this chunk
            pathfindingManager->invalidateCache(chunk);

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

void assignHouseToWorkplace(Chunk &chunk, Entities &entities, int32_t houseId,
                            int32_t workplaceId) {
    int32_t newEntity = entities.newEntity(chunk.getCellPosition(houseId), houseId, workplaceId);
    chunk.assignEntityToWorkplace(houseId, workplaceId);
}

void assignOneHouse(Map &map, Entities &entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();
    int32_t &assigned = *allocator->alloc<int32_t *>(sizeof(int32_t));
    if (grid.thread_rank() == 0) {
        assigned = 0;
    }
    grid.sync();

    CellBuffer unassignedHouses = map.houses.subBuffer(allocator, [&](MapId house) {
        HouseCell &houseCell = map.getTyped<HouseCell>(house);
        return houseCell.residentCount < houseCell.maxResidents();
    });
    grid.sync();
    if (unassignedHouses.getCount() == 0) {
        return;
    }

    CellBuffer availableWorkplaces = map.workplaces.subBuffer(allocator, [&](MapId workplace) {
        return map.getTyped<WorkplaceCell>(workplace).workplaceCapacity > 0;
    });
    grid.sync();
    if (availableWorkplaces.getCount() == 0) {
        return;
    }

    unassignedHouses.processEachCell_blockwise([&](MapId house) {
        if (assigned) {
            return;
        }
        int32_t workplaceId = map.findClosestOnNetworkBlockwise(
            availableWorkplaces, house, [&](MapId workplace) { return true; });

        if (block.thread_rank() == 0 && workplaceId != -1 && !atomicAdd(&assigned, 1)) {
            assignHouseToWorkplace(map.getChunk(house.chunkId), entities, house.cellId,
                                   workplaceId);
        }
    });
}

void assignOneCustomerToShop(Map &map, Entities &entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    __shared__ bool assigned;
    if (block.thread_rank() == 0) {
        assigned = false;
    }
    for_blockwise(entities.getCount(), [&](int entityIdx) {
        block.sync();
        if (assigned) {
            return;
        }
        Entity &entity = entities.get(entityIdx);
        if (!entity.active || entity.state != GoShopping || entity.destination != -1) {
            return;
        }

        int32_t shopId = map.findClosestOnNetworkBlockwise(
            map.shops, map.cellAtPosition(entity.position),
            [&](MapId shop) { return map.getTyped<ShopCell>(shop).woodCount > 0; });

        if (block.thread_rank() == 0) {
            if (shopId != -1) {
                entity.destination = shopId;
                assigned = true;
            } else {
                entity.changeState(GoHome);
            }
        }
    });
}

void assignShopWorkerToFactory(Map &map, Entities &entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    __shared__ bool assigned;
    if (block.thread_rank() == 0) {
        assigned = false;
    }
    block.sync();
    for_blockwise(entities.getCount(), [&](int entityIdx) {
        if (assigned) {
            return;
        }
        Entity &entity = entities.get(entityIdx);
        if (!entity.active || entity.state != WorkAtShop || entity.destination != -1) {
            return;
        }

        int32_t factoryId = map.findClosestOnNetworkBlockwise(
            map.factories, map.cellAtPosition(entity.position),
            [&](MapId factory) { return map.getTyped<FactoryCell>(factory).stockCount > 0; });

        if (block.thread_rank() == 0) {
            if (factoryId != -1) {
                entity.destination = factoryId;
                assigned = true;
            }
        }
        block.sync();
    });
}

uint32_t currentTime_ms() { return uint32_t((nanotime_start / (uint64_t)1e6) & 0xffffffff); }

void updateEntitiesState(Map &map, Entities &entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto &chunk = map.getChunk(0);

    auto gameTime = GameState::instance->gameTime;

    // Each thread handles an entity
    entities.processAll([&](int entityIndex) {
        Entity &entity = entities.get(entityIndex);
        entity.checkWaitStatus();

        if (!entity.active) {
            return;
        }

        bool destinationReached = false;
        if (entity.destination != -1 &&
            chunk.cellAtPosition(entity.position) == entity.destination) {
            destinationReached = true;
            entity.position = chunk.getCellPosition(entity.destination);
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
                atomicAdd(&chunk.getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount, 1);
                switch (chunk.get(entity.workplaceId).tileId) {
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
                atomicAdd(&chunk.getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount,
                          -1);
                break;
            }
            atomicAdd(&chunk.getTyped<FactoryCell>(entity.workplaceId).stockCount, 1);
            entity.wait(30);
            break;
        }
        case WorkAtShop: {
            if (!workHours.contains(gameTime.timeOfDay())) {
                entity.changeState(GoShopping);
                atomicAdd(&chunk.getTyped<WorkplaceCell>(entity.workplaceId).currentWorkerCount,
                          -1);
                break;
            }
            if (destinationReached && entity.destination == entity.workplaceId) {
                atomicAdd(&chunk.getTyped<ShopCell>(entity.workplaceId).woodCount,
                          entity.inventory);
                entity.inventory = 0;
                entity.destination = -1;
                entity.wait(10);
            } else if (destinationReached && chunk.get(entity.destination).tileId == FACTORY) {
                entity.inventory += min(chunk.getTyped<FactoryCell>(entity.destination).stockCount,
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

            int shopId = chunk.cellAtPosition(entity.position);
            int32_t &shopWood = chunk.getTyped<ShopCell>(shopId).woodCount;

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
                atomicAdd(&chunk.getTyped<HouseCell>(entity.houseId).woodCount, entity.inventory);
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

void updateGameState(Entities &entities) {
    auto grid = cg::this_grid();
    if (grid.thread_rank() == 0) {
        float dt = ((float)(nanotime_start - GameState::instance->previousFrameTime_ns)) / 1e9;
        GameState::instance->previousFrameTime_ns = nanotime_start;
        GameState::instance->currentTime_ms = currentTime_ms();
        GameState::instance->population = entities.getCount();
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

void handleInputs(Map &map, Entities &entities) {
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
        float2 worldPos = float2{pos_W.x, pos_W.y};
        int chunkId = map.chunkIdFromWorldPos(worldPos);
        if (chunkId == -1) {
            return;
        }
        auto &chunk = map.getChunk(chunkId);
        int id = chunk.cellAtPosition(worldPos);

        if (mouseClicked && (TileId)gameData.uniforms.modeId != GRASS) {
            if (grid.thread_rank() == 0) {
                if (chunk.get(id).tileId & (HOUSE | FACTORY | SHOP)) {
                    if (GameState::instance->buildingDisplay == id) {
                        GameState::instance->buildingDisplay = -1;
                    } else {
                        GameState::instance->buildingDisplay = id;
                    }
                }
            }
        } else if (mousePressed) {
            if (id != -1 &&
                (chunk.get(id).tileId == GRASS || (TileId)gameData.uniforms.modeId == GRASS)) {
                updateInfo.update = true;
                updateInfo.cellToUpdate = id;
                updateInfo.newTileId = (TileId)gameData.uniforms.modeId;
            }
        }
        if (updateInfo.update) {
            updateCell(chunk, entities, updateInfo);
        }
    }
}

void fillCellBuffers(Map &map) {
    auto grid = cg::this_grid();
    map.workplaces = map.selectCells(
        allocator, [&](MapId cell) { return map.getChunk(cell.chunkId).isWorkplace(cell.cellId); });
    map.factories =
        map.selectCells(allocator, [&](MapId cell) { return map.get(cell).tileId == FACTORY; });
    map.shops =
        map.selectCells(allocator, [&](MapId cell) { return map.get(cell).tileId == SHOP; });
    map.houses =
        map.selectCells(allocator, [&](MapId cell) { return map.get(cell).tileId == HOUSE; });
}

void handleEvents(Map &map, Entities &entities) {
    auto tod = GameState::instance->gameTime.timeOfDay();
    auto prevTod = GameState::instance->previousGameTime.timeOfDay();

    TOD sellResourcesTime{23, 0};
    // if (prevTod <= sellResourcesTime && sellResourcesTime <= tod) {
    //     // sell all resources at 23:00
    //     shops.processEachCell([&](int cellId) {
    //         ShopCell &shop = map->getTyped<ShopCell>(cellId);
    //         atomicAdd(&GameState::instance->playerMoney, shop.woodCount * WOOD_SELL_PRICE);
    //         shop.woodCount = 0;
    //     });
    // }

    map.houses.processEachCell([&](MapId house) {
        HouseCell &houseCell = map.getTyped<HouseCell>(house);
        if (houseCell.woodCount >= houseCell.upgradeCost()) {
            houseCell.levelUp();
        }
    });
}

void moveEntitiesBetter(Map &map, Entities &entities, Allocator *allocator, float gameDt) {
    auto grid = cg::this_grid();
    float dt = min(0.01f, gameDt);
    float cumul = 0.0f;
    int iterations = 0;
    do {
        moveEntities(map, entities, allocator, dt);
        grid.sync();
        cumul += dt;
        dt = min(0.01f, gameDt - cumul);
        iterations++;
    } while (cumul < gameDt);
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

void updateGrid(Map &map, Entities &entities) {
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
    grid.sync();
    printDuration("assignOneCustomerToShop     ",
                  [&]() { assignOneCustomerToShop(map, entities); });
    printDuration("assignShopWorkerToFactory   ",
                  [&]() { assignShopWorkerToFactory(map, entities); });
    printDuration("pathfinding                 ",
                  [&]() { pathfindingManager->update(map.getChunk(0), entities, *allocator); });
    grid.sync();
    printDuration("moveEntitiesBetter           ", [&]() {
        moveEntitiesBetter(map, entities, allocator, GameState::instance->gameTime.getDt());
    });
    grid.sync();
    printDuration("updateEntitiesState         ", [&]() { updateEntitiesState(map, entities); });
    grid.sync();
    //// printDuration("entitiesInteractions        ", [&]() { entitiesInteractions(map, entities);
    //// });
    printDuration("updateGameState             ", [&]() { updateGameState(entities); });
}

extern "C" __global__ void update(GameData _gameData) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    gameData = _gameData;
    GameState::instance = gameData.state;

    Allocator _allocator(gameData.buffer, 0);
    allocator = &_allocator;

    PathfindingManager _pathfindingManager(gameData.savedFieldsBuffer);
    pathfindingManager = &_pathfindingManager;

    curand_init(grid.thread_rank() + GameState::instance->currentTime_ms, 0, 0,
                &thread_random_state);

    grid.sync();

    {
        Map *map = allocator->alloc<Map *>(sizeof(Map));
        *map = Map(gameData.numRows, gameData.numCols, gameData.chunks);

        Entities *entities = allocator->alloc<Entities *>(sizeof(Entities));
        *entities = Entities(gameData.entitiesBuffer);

        updateGrid(*map, *entities);
    }
}
