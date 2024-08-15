#pragma once
#include "World/Entities/entities.cuh"
#include "World/map.cuh"

struct PathfindingInfo {
    uint32_t entityIdx;
    uint32_t origin;
    uint32_t target;
};

struct PathfindingList {
    PathfindingInfo *data;
    uint32_t count;
};

class PathfindingManager {
private:
    Flowfield *cachedFlowfields;
    TileId *tileIds;

public:
    PathfindingManager(void *buffer) : cachedFlowfields((Flowfield *)(buffer)) {}

    // Perform pathfinding
    void update(Map &map, Entities &entities, Allocator &allocator);

    void invalidateCache() {
        processRange(MAPX * MAPY, [&](int cellId) { cachedFlowfields[cellId].state = INVALID; });
    }

private:
    PathfindingList locateLostEntities(Map &map, Entities &entities, Allocator &allocator) const;
    inline bool isNeighborValid(Map &map, uint32_t cellId, uint32_t neighborId, int2 dirCoords,
                                uint32_t targetId) const {
        if (neighborId != targetId && tileIds[neighborId] != ROAD) {
            return false;
        }

        if (dirCoords.x == 0 || dirCoords.y == 0) {
            return true;
        }

        int2 currentCellCoord = map.cellCoords(cellId);
        int id1 = map.idFromCoords(currentCellCoord + int2{dirCoords.x, 0});
        int id2 = map.idFromCoords(currentCellCoord + int2{0, dirCoords.y});
        return (id1 != -1 && tileIds[id1] == ROAD) || (id2 != -1 && tileIds[id2] == ROAD);
    }

    Path extractPath(Map &map, const PathfindingInfo &info) const;
};