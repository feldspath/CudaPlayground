#pragma once

#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "builtin_types.h"
#include "cell_buffer.cuh"
#include "config.h"
#include "direction.h"

static unsigned int tileCost(TileId tile) {
    switch (tile) {
    case UNKNOWN:
        return 0;
    case GRASS:
        return 0;
    case ROAD:
        return ROAD_COST;
    case HOUSE:
        return HOUSE_COST;
    case FACTORY:
        return FACTORY_COST;
    default:
        return 0;
    }
}

class Map {
private:
    Chunk *chunks;
    int rows;
    int cols;
    int count;

public:
    CellBuffer<MapId> shops;
    CellBuffer<MapId> houses;
    CellBuffer<MapId> factories;
    CellBuffer<MapId> workplaces;

    inline int getCount() const { return count; }

    Map(uint32_t numRows, uint32_t numCols, Chunk *chunks)
        : chunks(chunks), rows(numRows), cols(numCols), count(numCols * numRows) {}

    Chunk &getChunk(int chunkId) { return chunks[chunkId]; }
    const Chunk &getChunk(int chunkId) const { return chunks[chunkId]; }

    template <typename T> T &getTyped(MapId cell) {
        return getChunk(cell.chunkId).getTyped<T>(cell.cellId);
    }
    template <typename T> const T &getTyped(MapId cell) const {
        return getChunk(cell.chunkId).getTyped<T>(cell.cellId);
    }
    BaseCell &get(MapId cell) { return getChunk(cell.chunkId).get(cell.cellId); }
    const BaseCell &get(MapId cell) const { return getChunk(cell.chunkId).get(cell.cellId); }

    int chunkIdFromWorldPos(float2 pos) const {
        int col = floor(pos.x / (CHUNK_X * 2.0f * CELL_RADIUS));
        int row = floor(pos.y / (CHUNK_Y * 2.0f * CELL_RADIUS));
        return chunkIdFromCoord(col, row);
    }
    int2 chunkCoord(int chunkId) const { return {chunkId % cols, chunkId / cols}; }
    int chunkIdFromCoord(int chunkCol, int chunkRow) const {
        if (chunkRow < 0 || chunkRow >= rows || chunkCol < 0 || chunkCol >= cols) {
            return -1;
        }
        return chunkRow * cols + chunkCol;
    }

    int neighborChunkId(int chunkId, Direction dir) const {
        int2 dirCoord = coordFromEnum(dir);
        int2 otherChunkCoord = chunkCoord(chunkId) + dirCoord;
        return chunkIdFromCoord(otherChunkCoord.x, otherChunkCoord.y);
    }

    MapId cellAtPosition(float2 pos) const {
        int chunkId = chunkIdFromWorldPos(pos);
        if (chunkId == -1) {
            return MapId::invalidId();
        }

        int cellId = getChunk(chunkId).cellAtPosition(pos);
        return {chunkId, cellId};
    }
    float2 getCellPosition(MapId &cell) const {
        return getChunk(cell.chunkId).getCellPosition(cell.cellId);
    }

    int2 cellCoords(MapId cell) const { return getChunk(cell.chunkId).cellCoords(cell.cellId); }
    MapId cellFromCoords(int2 coords) const {
        int col = coords.x / CHUNK_X;
        int row = coords.y / CHUNK_Y;
        int chunkId = chunkIdFromCoord(col, row);
        if (chunkId == -1) {
            return MapId::invalidId();
        }
        int cellId = getChunk(chunkId).idFromCoords(coords);
        return {chunkId, cellId};
    }

    template <typename Function> void processEachCell(Function &&f) {
        // Each block handles a chunk
        for_blockwise(count, [&](int chunkId) {
            // Each thread in the block handles a cell
            getChunk(chunkId).processEachCellBlock([&](int cellId) { f({chunkId, cellId}); });
        });
    }

    template <typename Function> void processEachCell(TileId filter, Function &&f) {
        // Each block handles a chunk
        for_blockwise(count, [&](int chunkId) {
            // Each thread in the block handles a cell
            getChunk(chunkId).processEachCellBlock(filter,
                                                   [&](int cellId) { f(MapId(chunkId, cellId)); });
        });
    }

    // Buffers
    // TODO: cells in order of chunkId
    template <typename Function>
    CellBuffer<MapId> selectCells(Allocator *allocator, Function &&filter) {
        auto grid = cg::this_grid();

        // count occurences
        uint32_t &cellCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));
        if (grid.thread_rank() == 0) {
            cellCount = 0;
        }
        grid.sync();

        processEachCell([&](MapId cell) {
            if (!filter(cell)) {
                return;
            }
            atomicAdd(&cellCount, 1);
        });

        grid.sync();

        if (cellCount == 0) {
            return CellBuffer<MapId>();
        }

        // allocate buffer
        MapId *cellIds = allocator->alloc<MapId *>(sizeof(MapId) * cellCount);
        grid.sync();

        // fill buffer
        if (grid.thread_rank() == 0) {
            cellCount = 0;
        }
        grid.sync();

        processEachCell([&](MapId cell) {
            if (!filter(cell)) {
                return;
            }
            int idx = atomicAdd(&cellCount, 1);
            cellIds[idx] = cell;
        });

        grid.sync();

        return CellBuffer(cellIds, cellCount);
    }

    template <typename Function>
    MapId findClosestOnNetworkBlockwise(CellBuffer<MapId> &buffer, MapId cell, Function &&filter) {
        auto block = cg::this_thread_block();

        __shared__ uint64_t targetCell;
        if (block.thread_rank() == 0) {
            targetCell = uint64_t(Infinity) << 32ull;
        }

        block.sync();

        auto cellNets = neighborNetworks(cell);
        int2 originCoords = cellCoords(cell);

        buffer.processEachCellBlock([&](MapId other) {
            if (!filter(other)) {
                return;
            }
            auto targetNets = neighborNetworks(other);
            if (sharedNetworks(cellNets, targetNets).data[0] != -1) {
                int2 targetCoords = cellCoords(other);
                int2 diff = targetCoords - originCoords;
                uint32_t distance = abs(diff.x) + abs(diff.y);
                uint64_t target = (uint64_t(distance) << 32ull) |
                                  uint64_t(other.chunkId * CHUNK_SIZE + other.cellId);
                // keep the closest
                atomicMin(&targetCell, target);
            }
        });

        block.sync();

        if (targetCell != uint64_t(Infinity) << 32ull) {
            int32_t t = int32_t(targetCell & 0xffffffffull);
            return MapId(t / CHUNK_SIZE, t % CHUNK_SIZE);
        } else {
            return MapId::invalidId();
        }
    }

    // Util functions
    MapNeighbors neighborCells(MapId cell) const {
        int2 coords = cellCoords(cell);
        MapNeighbors result;
        result.setDir([&](Direction dir) {
            int2 dirCoord = coordFromEnum(dir);
            return cellFromCoords({coords.x + dirCoord.x, coords.y + dirCoord.y});
        });
        return result;
    }

    // Network functions
    MapNeighbors neighborNetworks(MapId cell) {
        auto neighbors = neighborCells(cell);
        return neighbors.apply([&](MapId neighborCell) {
            if (get(neighborCell).tileId == ROAD) {
                return getTyped<RoadCell>(neighborCell).networkRepr;
            } else {
                return MapId::invalidId();
            }
        });
    }

    MapNeighbors sharedNetworks(MapNeighbors nets1, MapNeighbors nets2) {
        MapNeighbors result;
        int count = 0;
        nets1.forEach([&](MapId networkRepr) {
            if (nets2.contains(networkRepr)) {
                result.data[count] = networkRepr;
                count++;
            }
        });
        return result;
    }

    MapNeighbors sharedNetworks(MapId cell1, MapId cell2) {
        MapNeighbors nets1 = neighborNetworks(cell1);
        MapNeighbors nets2 = neighborNetworks(cell2);
        return sharedNetworks(nets1, nets2);
    }

    int flattenMapId(MapId mapId) { return mapId.chunkId * CHUNK_SIZE + mapId.cellId; }
    MapId unflattenMapId(int flattenedId) {
        return MapId(flattenedId / CHUNK_SIZE, flattenedId % CHUNK_SIZE);
    }

    void updateNetworkComponents(MapId invalidNetwork, int32_t invalidChunk, Allocator allocator) {
        auto grid = cg::this_grid();
        auto block = cg::this_thread_block();

        bool *validCells = allocator.alloc<bool *>(sizeof(bool) * CHUNK_SIZE * getCount());

        // reset invalid network
        processEachCell(ROAD, [&](MapId cell) {
            auto &c = getTyped<RoadCell>(cell);
            int id = flattenMapId(cell);
            if (c.networkRepr != invalidNetwork) {
                validCells[id] = true;
                return;
            }
            validCells[id] = false;
            if (cell.chunkId == invalidChunk) {
                // Chunk is invalid, reset both networks
                c.networkRepr = cell;
                c.chunkNetworkRepr = cell.cellId;
            } else {
                // Chunk network is fine
                c.networkRepr = MapId(cell.chunkId, c.chunkNetworkRepr);
            }
        });

        grid.sync();

        // recompute connected components
        // https://largo.lip6.fr/~lacas/Publications/IPTA17.pdf

        // Changed chunk first
        auto &chunk = getChunk(invalidChunk);
        bool &changed = *allocator.alloc<bool *>(sizeof(bool));
        for (int i = 0; i < 10; ++i) {
            if (grid.thread_rank() == 0) {
                changed = false;
            }
            grid.sync();
            chunk.processEachCell(ROAD, [&](int cellId) {
                if (validCells[flattenMapId(MapId(invalidChunk, cellId))]) {
                    return;
                }

                int32_t m = -1;
                auto neighbors = chunk.neighborNetworks(cellId);
                for (int i = 0; i < 4; ++i) {
                    if (neighbors.data[i] == -1) {
                        continue;
                    }
                    if (m == -1) {
                        m = neighbors.data[i];
                    } else {
                        m = min(neighbors.data[i], m);
                    }
                }
                if (m == -1) {
                    return;
                }

                int32_t repr = chunk.getTyped<RoadCell>(cellId).chunkNetworkRepr;
                int32_t old = atomicMin(&chunk.getTyped<RoadCell>(repr).chunkNetworkRepr, m);
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
            chunk.processEachCell(ROAD, [&](int cellId) {
                if (validCells[flattenMapId(MapId(invalidChunk, cellId))]) {
                    return;
                }
                auto &cellTile = chunk.getTyped<RoadCell>(cellId);
                auto &network = cellTile.chunkNetworkRepr;
                while (network != chunk.getTyped<RoadCell>(network).chunkNetworkRepr) {
                    network = chunk.getTyped<RoadCell>(network).chunkNetworkRepr;
                }
                cellTile.chunkNetworkRepr = network;
            });
        }
        grid.sync();

        chunk.processEachCell(ROAD, [&](int cellId) {
            if (validCells[flattenMapId(MapId(invalidChunk, cellId))]) {
                return;
            }

            auto &cell = chunk.getTyped<RoadCell>(cellId);
            cell.networkRepr = MapId(invalidChunk, cell.chunkNetworkRepr);
        });

        grid.sync();

        // Then the global network
        for (int i = 0; i < 10; ++i) {
            if (grid.thread_rank() == 0) {
                changed = false;
            }
            grid.sync();
            processEachCell(ROAD, [&](MapId cell) {
                if (validCells[flattenMapId(cell)]) {
                    return;
                }

                int64_t m = -1;
                auto neighbors = neighborNetworks(cell);
                for (int i = 0; i < 4; ++i) {
                    if (!neighbors.data[i].valid()) {
                        continue;
                    }
                    if (m == -1) {
                        m = neighbors.data[i].as_int64();
                    } else {
                        m = min(neighbors.data[i].as_int64(), m);
                    }
                }
                if (m == -1) {
                    return;
                }

                auto &repr = getTyped<RoadCell>(cell).networkRepr;
                int64_t old = atomicMin(&getTyped<RoadCell>(repr).networkRepr.as_int64(), m);
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
            processEachCell(ROAD, [&](MapId cell) {
                if (validCells[flattenMapId(cell)]) {
                    return;
                }
                auto &cellTile = getTyped<RoadCell>(cell);
                auto &network = cellTile.networkRepr;
                while (network != getTyped<RoadCell>(network).networkRepr) {
                    network = getTyped<RoadCell>(network).networkRepr;
                }
                cellTile.networkRepr = network;
            });
        }

        grid.sync();
    }

    void assignEntityToWorkplace(MapId house, MapId workplace) {
        getTyped<HouseCell>(house).residentCount = getTyped<HouseCell>(house).residentCount + 1;
        getTyped<WorkplaceCell>(workplace).workplaceCapacity -= 1;
    }
};
