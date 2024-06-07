#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "./../common/utils.cuh"
#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "entities.h"
#include "helper_math.h"
#include "map.h"
#include "matrix_math.h"
#include "pathfinding.h"

namespace cg = cooperative_groups;

Uniforms uniforms;
GameState *gameState;
Allocator *allocator;
uint64_t nanotime_start;

curandState crState;

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

    grid.sync();

    int32_t &assigned = *allocator->alloc<int32_t *>(4);
    uint32_t &unassignedHouseCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &availableFactoriesCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &globalHouseIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &globalFactoryIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        unassignedHouseCount = 0;
        availableFactoriesCount = 0;
        globalHouseIdx = 0;
        globalFactoryIdx = 0;
        assigned = 0;
    }

    grid.sync();

    map->processEachCell(HOUSE | FACTORY, [&](int cellId) {
        if (map->getTileId(cellId) == HOUSE && *map->houseTileData(cellId) == -1) {
            atomicAdd(&unassignedHouseCount, 1);
        } else if (map->getTileId(cellId) == FACTORY && *map->factoryTileData(cellId) > 0) {
            atomicAdd(&availableFactoriesCount, 1);
        }
    });

    grid.sync();

    if (unassignedHouseCount == 0 || availableFactoriesCount == 0) {
        return;
    }

    uint32_t *availableFactories =
        allocator->alloc<uint32_t *>(sizeof(uint32_t) * availableFactoriesCount);
    uint32_t *unassignedHouses =
        allocator->alloc<uint32_t *>(sizeof(uint32_t) * unassignedHouseCount);

    map->processEachCell(HOUSE | FACTORY, [&](int cellId) {
        if (map->getTileId(cellId) == HOUSE && *map->houseTileData(cellId) == -1) {
            int idx = atomicAdd(&globalHouseIdx, 1);
            unassignedHouses[idx] = cellId;
        } else if (map->getTileId(cellId) == FACTORY && *map->factoryTileData(cellId) > 0) {
            int idx = atomicAdd(&globalFactoryIdx, 1);
            availableFactories[idx] = cellId;
        }
    });

    grid.sync();

    __shared__ uint64_t targetFactory;
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
            targetFactory = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        // Check all tiles for factories
        for (int blockOffset = 0; blockOffset < availableFactoriesCount;
             blockOffset += block.num_threads()) {
            int fIdx = block.thread_rank() + blockOffset;
            if (fIdx >= availableFactoriesCount) {
                break;
            }
            int factoryId = availableFactories[fIdx];

            // Get the networks the factory is connected to
            auto factoryNets = map->neighborNetworks(factoryId);
            if (map->sharedNetworks(factoryNets, houseNets).data[0] != -1) {
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

        block.sync();

        if (block.thread_rank() == 0) {
            int32_t *houseData = map->houseTileData(houseId);
            if (targetFactory != uint64_t(Infinity) << 32ull && !atomicAdd(&assigned, 1)) {
                int32_t factoryId = targetFactory & 0xffffffffull;
                assignHouseToFactory(map, entities, houseId, factoryId);
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
            if (map->cellAtPosition(entity.position) == entity.factoryId) {
                entity.path.reset();
                entity.state = Work;
                entity.stateStart_ms = gameState->currentTime_ms;
                entity.position = map->getCellPosition(entity.factoryId);
            }
            break;
        }
        case Work:
            if (gameState->currentTime_ms - entity.stateStart_ms >= WORK_TIME_MS) {
                entity.state = GoHome;
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

void updateGameState() {
    gameState->dt = ((float)(nanotime_start - gameState->previousFrameTime_ns)) / 1e9;
    gameState->previousFrameTime_ns = nanotime_start;
    gameState->currentTime_ms = currentTime_ms();
}

float generateRandomNumber() {
    float myrandf = curand_uniform(&crState);
    return myrandf;
}

void moveEntities(Map *map, Entities *entities) {
    // Count entities to move
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    uint32_t &entitiesToMoveCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
    uint32_t &bufferIdx = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        entitiesToMoveCount = 0;
        bufferIdx = 0;
    }

    grid.sync();

    processRange(entities->getCount(), [&](int entityIdx) {
        if (entities->get(entityIdx).path.isValid()) {
            atomicAdd(&entitiesToMoveCount, 1);
        }
    });

    grid.sync();

    if (entitiesToMoveCount == 0) {
        return;
    }

    uint32_t *entitiesToMove = allocator->alloc<uint32_t *>(sizeof(uint32_t) * entitiesToMoveCount);

    processRange(entities->getCount(), [&](int entityIdx) {
        if (entities->get(entityIdx).path.isValid()) {
            int idx = atomicAdd(&bufferIdx, 1);
            entitiesToMove[idx] = entityIdx;
        }
    });

    grid.sync();

    float pressure_normalization = 15.0f / (3.141592 * powf(KERNEL_RADIUS, 6.0f));

    // Update velocities
    processRange(entitiesToMoveCount, [&](int idx) {
        auto &entity = entities->get(entitiesToMove[idx]);

        float2 forces = {0.0f, 0.0f};

        // Compute repulsive force of other entities
        for (int i = 0; i < entitiesToMoveCount; ++i) {
            if (i == idx) {
                continue;
            }
            Entity &other = entities->get(entitiesToMove[i]);
            float2 diffVector = entity.position - other.position;
            float norm = length(diffVector);
            if (norm < 1e-6) {
                continue;
            }
            if (norm < KERNEL_RADIUS) {
                forces += diffVector / norm * powf((KERNEL_RADIUS - norm), 3.0) *
                          REPULSIVE_STRENGTH * pressure_normalization;
            }
        }

        // Stirring force
        forces += directionFromEnum(entity.path.nextDir()) * STIR_STRENGTH;

        // float angle = generateRandomNumber() * 2.0 * 3.141592;
        // forces += length(forces) * 0.5 * float2{cos(angle), sin(angle)};

        entity.velocity += forces * gameState->dt;

        // Clamp velocity
        float velocityNorm = length(entity.velocity);
        if (velocityNorm > ENTITY_SPEED) {
            entity.velocity = entity.velocity / velocityNorm * ENTITY_SPEED;
        }
    });

    grid.sync();

    // Update positions
    processRange(entitiesToMoveCount, [&](int idx) {
        uint32_t entityId = entitiesToMove[idx];
        auto &entity = entities->get(entityId);

        float2 entityDirVector = directionFromEnum(entity.path.nextDir());
        uint32_t previousCellId = map->cellAtPosition(entity.position);
        uint32_t previousMaxCellId =
            map->cellAtPosition(entity.position - entityDirVector * ENTITY_RADIUS * 1.2);
        entity.position += entity.velocity * gameState->dt;

        // check each side of the entity for collision
        auto neighborCells = map->neighborCells(previousCellId);
        neighborCells.forEachDir([&](Direction direction, uint32_t cellId) {
            // no collision with roads, house and assigned factory
            if (map->getTileId(cellId) == ROAD || cellId == entity.factoryId ||
                cellId == entity.houseId) {
                return;
            }

            float2 vectorDir = directionFromEnum(direction);
            if (map->cellAtPosition(entity.position + vectorDir * ENTITY_RADIUS) == cellId) {
                // Collision with wall, project back to boundary
                float2 posToPreviousCellCenter = entity.position + vectorDir * ENTITY_RADIUS -
                                                 map->getCellPosition(previousCellId);
                entity.position -=
                    (dot(posToPreviousCellCenter, vectorDir) - CELL_RADIUS) * vectorDir;
            }
        });

        uint32_t newCellId =
            map->cellAtPosition(entity.position - entityDirVector * ENTITY_RADIUS * 1.2);

        if (newCellId != previousMaxCellId) {
            int dir = -1;
            neighborCells.forEachDir([&](Direction direction, uint32_t cellId) {
                if (cellId = newCellId) {
                    dir = int(direction);
                }
            });

            if (dir != -1 && Direction(dir) == entity.path.nextDir()) {
                // Entity is on intended road
                entity.path.pop();
            } else {
                // if the entity went too far, or is not an intended road, invalidate path
                entity.path.reset();
            }
        }
    });
}

void updateGrid(Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(idx, idx, 0, &crState);

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

    performPathFinding(map, entities, allocator);

    moveEntities(map, entities);

    updateEntitiesState(map, entities);

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
