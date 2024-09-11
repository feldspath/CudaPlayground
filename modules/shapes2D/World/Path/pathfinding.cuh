#pragma once
#include "World/Entities/entities.cuh"
#include "World/map.cuh"

struct PathfindingInfo {
    MapId origin;
    MapId destination;
    uint32_t entityIdx;
};

struct PathfindingList {
    PathfindingInfo *data;
    uint32_t count;
};

class PathfindingManager {
private:
    IntegrationField *savedFields;
    uint8_t *tileIds;

public:
    PathfindingManager(void *savedFieldsBuffer)
        : savedFields((IntegrationField *)savedFieldsBuffer) {}

    // Perform pathfinding
    // Passing a copy of the allocator so that its state is reset after computation
    void update(Map &map, Allocator allocator);
    void entitiesPathfinding(Map &map, Entities &entities, Allocator allocator);

    void invalidateCache(Chunk &chunk) {
        chunk.invalidateCachedFlowfields();
        processRange(gridDim.x, [&](int idx) { savedFields[idx].ongoingComputation = false; });
    }

    static int maxFlowfieldsPerFrame() { return gridDim.x; };

private:
    PathfindingList locateLostEntities(Map &map, Entities &entities, Allocator &allocator) const;
    inline bool isNeighborValid(Chunk &chunk, uint32_t cellId, uint32_t neighborId, int2 dirCoords,
                                uint32_t targetId) const {
        if (neighborId != targetId && TileId(tileIds[neighborId]) != ROAD) {
            return false;
        }

        if (dirCoords.x == 0 || dirCoords.y == 0) {
            return true;
        }

        int2 currentCellCoord = chunk.cellCoords(cellId);
        int id1 = chunk.idFromCoords(currentCellCoord + int2{dirCoords.x, 0});
        int id2 = chunk.idFromCoords(currentCellCoord + int2{0, dirCoords.y});
        return (id1 != -1 && TileId(tileIds[id1]) == ROAD) ||
               (id2 != -1 && TileId(tileIds[id2]) == ROAD);
    }

    Path extractPath(Chunk &chunk, uint32_t origin, uint32_t target) const;
    int pathLength(Chunk &chunk, uint32_t origin, uint32_t target) const;
};