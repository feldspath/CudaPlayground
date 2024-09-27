#pragma once
#include "World/Entities/entities.cuh"
#include "World/graph.h"
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

class NetworkGraph {
private:
    int *numNetworks;
    NetworkNode *networks;

public:
    NetworkGraph(void *networkBuffer) {
        networks = (NetworkNode *)((int *)networkBuffer + 2);
        numNetworks = (int *)networkBuffer;
    }

    int getNetworkCount() const { return *numNetworks; }

    int addNetwork(MapId network) {
        networks[*numNetworks] = NetworkNode(network);
        int newId = *numNetworks;
        *numNetworks = (*numNetworks) + 1;
        return newId;
    }

    void changeNetworkRepr(int networkId, MapId newNetworkRepr) {
        networks[networkId].networkRepr = newNetworkRepr;
    }

    void addNeighbor(int networkId, int neighborId) {
        auto &node = networks[networkId];
        node.neighborIds[node.numNeighbors] = neighborId;
        node.numNeighbors++;
    }

    void eraseNeighbor(int networkId, int neighborId) {
        auto &node = networks[networkId];
        int idx = -1;
        for (int i = 0; i < node.numNeighbors; i++) {
            if (node.neighborIds[i] == neighborId) {
                idx = i;
            }
        }
        if (idx == -1) {
            printf("error: node %d is not a neighbor of %d\n", neighborId, networkId);
            return;
        }

        node.numNeighbors--;
        if (idx == node.numNeighbors) {
            return;
        }
        node.neighborIds[idx] = node.numNeighbors;
    }

    void replaceNeighbor(int networkId, int oldNeighborId, int newNeighborId) {
        auto &node = networks[networkId];
        int idx = -1;
        for (int i = 0; i < node.numNeighbors; i++) {
            if (node.neighborIds[i] == oldNeighborId) {
                idx = i;
            }
        }
        if (idx == -1) {
            printf("error: node %d is not a neighbor of %d\n", oldNeighborId, networkId);
            return;
        }

        node.neighborIds[idx] = newNeighborId;
    }

    /// Add neighborId to the list of neighbors of networkId, return true if the neighbor was added,
    /// false if it was already in the list.
    bool addNeighborIfMissing(int networkId, int neighborId) {
        if (!networksConnected(networkId, neighborId)) {
            addNeighbor(networkId, neighborId);
            return true;
        }
        return false;
    }

    const NetworkNode &getNode(int networkId) const { return networks[networkId]; }

    void eraseNetwork(int networkId, Map &map) {
        *numNetworks = (*numNetworks) - 1;
        int oldNetworkId = *numNetworks;
        if (networkId != oldNetworkId) {
            // Copying the last network to the erased network position
            // TODO: check performance
            networks[networkId] = networks[oldNetworkId];
            auto &networkToUpdateNode = networks[networkId];

            // Update the network id
            map.getTyped<RoadCell>(networkToUpdateNode.networkRepr).networkId = networkId;

            // Update all the references to the old networkId
            // TODO: the neighbors could just be MapIds
            for (int i = 0; i < networkToUpdateNode.numNeighbors; ++i) {
                auto &node = networks[networkToUpdateNode.neighborIds[i]];
                for (int j = 0; j < node.numNeighbors; j++) {
                    if (node.neighborIds[j] == oldNetworkId) {
                        node.neighborIds[j] = networkId;
                        break;
                    }
                }
            }
        }
    }

    bool networksConnected(int networkId1, int networkId2) const {
        auto &node1 = networks[networkId1];
        for (int i = 0; i < node1.numNeighbors; i++) {
            if (node1.neighborIds[i] == networkId2) {
                return true;
            }
        }
        return false;
    }

    void mergeNetworks(int dstNetwork, int srcNetwork, Map &map) {
        if (dstNetwork == srcNetwork) {
            printf("Error: dstNetwork and srcNetwork are identical");
            return;
        }

        // Merge neighbors
        auto &srcNode = networks[srcNetwork];
        for (int i = 0; i < srcNode.numNeighbors; i++) {
            bool added = addNeighborIfMissing(dstNetwork, srcNode.neighborIds[i]);
            if (added) {
                replaceNeighbor(srcNode.neighborIds[i], srcNetwork, dstNetwork);
            } else {
                eraseNeighbor(srcNode.neighborIds[i], srcNetwork);
            }
        }

        // Erase src
        eraseNetwork(srcNetwork, map);
    }

    void updateNetwork(Neighbors networksToMerge, MapId cellToUpdate, Map &map) {
        int mergeNet = networksToMerge.data[0];

        // Merge all networks in mergeNet
        networksToMerge.forEach([&](int networkId) {
            if (networkId < mergeNet) {
                int oldNet = mergeNet;
                mergeNet = networkId;
                mergeNetworks(mergeNet, oldNet, map);
            } else if (networkId > mergeNet) {
                mergeNetworks(mergeNet, networkId, map);
            }
        });

        // If there is no network around the new tile, create a new one
        if (mergeNet == -1) {
            mergeNet = addNetwork(cellToUpdate);
        }

        // cellToUpdate is the new networkRepr
        auto &cell = map.getTyped<RoadCell>(cellToUpdate);
        cell.networkId = mergeNet;
        changeNetworkRepr(mergeNet, cellToUpdate);
    }

    void recompute(Map &map, Allocator allocator) {
        auto grid = cg::this_grid();
        auto block = cg::this_thread_block();

        int &networkCount = *allocator.alloc<int *>(sizeof(int));

        if (grid.thread_rank() == 0) {
            networkCount = 0;
        }
        grid.sync();

        // networkId assigment
        map.processEachCell(ROAD, [&](MapId cell) {
            auto &road = map.getTyped<RoadCell>(cell);
            if (road.chunkNetworkRepr == cell.cellId) {
                road.networkId = atomicAdd(&networkCount, 1);
                networks[road.networkId].networkRepr = cell;
            }
        });

        grid.sync();

        // Neighbor matrix
        bool *neighborMatrix = allocator.alloc<bool *>(sizeof(bool) * networkCount * networkCount);

        // init matrix
        processRange(networkCount * networkCount, [&](int idx) { neighborMatrix[idx] = false; });
        grid.sync();

        // fill matrix
        map.processEachCell(ROAD, [&](MapId cell) {
            map.neighborCells(cell).forEach([&](MapId neighbor) {
                if (cell.chunkId != neighbor.chunkId && map.get(neighbor).tileId == ROAD) {
                    int network1 = map.roadNetworkId(cell);
                    int network2 = map.roadNetworkId(neighbor);
                    neighborMatrix[network1 + network2 * networkCount] = true;
                }
            });
        });

        grid.sync();

        // Reset graph
        if (grid.thread_rank() == 0) {
            reset(networkCount);
        }
        grid.sync();

        // Fill neighbors
        processRange(networkCount, [&](int networkId) {
            for (int i = 0; i < networkCount; i++) {
                if (neighborMatrix[networkId + networkCount * i]) {
                    addNeighbor(networkId, i);
                }
            }
        });

        grid.sync();
    }

    void reset(int newSize) {
        *numNetworks = newSize;
        for (int i = 0; i < newSize; i++) {
            networks[i].numNeighbors = 0;
        }
    }

    void print() {
        for (int i = 0; i < *numNetworks; i++) {
            auto &node = getNode(i);
            printf("node %d:\n", i);
            for (int j = 0; j < node.numNeighbors; j++) {
                printf("\tneighbor: %d\n", node.neighborIds[j]);
            }
        }
        printf("\n");
    }
};

class PathfindingManager {
private:
    IntegrationField *savedFields;
    uint8_t *tileIds;

public:
    NetworkGraph networkGraph;

    PathfindingManager(void *savedFieldsBuffer, void *networkBuffer)
        : savedFields((IntegrationField *)savedFieldsBuffer), networkGraph(networkBuffer) {}

    // Perform pathfinding
    // Passing a copy of the allocator so that its state is reset after computation
    void update(Map &map, Allocator allocator);
    void entitiesPathfinding(Map &map, Entities &entities, Allocator allocator);

    void invalidateCache(Chunk &chunk) {
        chunk.invalidateCachedFlowfields();
        invalidateSavedFields();
    }
    void invalidateSavedFields() {
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