#pragma once
#include "World/Entities/entities.cuh"
#include "World/map.cuh"

struct IntegrationField {
    uint32_t *distances;
    uint32_t targetId;

    IntegrationField() {}

    // Buffer has to be at least of size IntegrationField::size()
    IntegrationField(uint32_t target, uint32_t *buffer) {
        this->targetId = target;
        this->distances = buffer;
    }

    static constexpr uint32_t size() { return MAPX * MAPY; }

    void resetCell(uint32_t cellId);

    inline uint32_t &getCell(uint32_t cellId) { return distances[cellId]; }
    inline const uint32_t &getCell(uint32_t cellId) const { return distances[cellId]; }
};

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
    bool isNeighborValid(Map &map, uint32_t cellId, uint32_t neighborId, Direction neighborDir,
                         uint32_t targetId) const;
    Path extractPath(Map &map, const PathfindingInfo &info) const;
};