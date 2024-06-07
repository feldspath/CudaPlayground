#include "../common/utils.cuh"
#include "pathfinding.h"

struct PathfindingList {
    Pathfinding *data;
    uint32_t count;
};

// Locate all entities that required pathfinding
static PathfindingList locateLostEntities(Map *map, Entities *entities, Allocator *allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    grid.sync();

    // One flow field per lost entity
    PathfindingList pathfindingList;
    pathfindingList.data =
        allocator->alloc<Pathfinding *>(entities->getCount() * sizeof(Pathfinding));
    uint32_t &lostCount = *allocator->alloc<uint32_t *>(sizeof(uint32_t));

    if (grid.thread_rank() == 0) {
        lostCount = 0;
    }
    grid.sync();

    processRange(entities->getCount(), [&](int entityIndex) {
        Entity &entity = entities->get(entityIndex);
        if (entity.isLost()) {
            uint32_t id = atomicAdd(&lostCount, 1);
            uint32_t targetId = entity.state == GoHome ? entity.houseId : entity.factoryId;
            int originId = map->cellAtPosition(entity.position);
            Pathfinding p;
            p.flowField = FlowField(map, targetId, originId);
            p.entityIdx = entityIndex;
            p.origin = originId;
            p.target = targetId;
            pathfindingList.data[id] = p;
        }
    });

    grid.sync();

    pathfindingList.count = lostCount;

    // Allocate the field for each pathfinding
    for (int i = 0; i < pathfindingList.count; ++i) {
        FieldCell *field = allocator->alloc<FieldCell *>(pathfindingList.data[i].flowField.size());
        if (grid.thread_rank() == 0) {
            pathfindingList.data[i].flowField.setBuffer(field);
        }
    }

    grid.sync();

    return pathfindingList;
}

void performPathFinding(Map *map, Entities *entities, Allocator *allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    PathfindingList pathfindingList = locateLostEntities(map, entities, allocator);

    // Each block handles a lost entity
    for (int offset = 0; offset < pathfindingList.count; offset += grid.num_blocks()) {
        int bufferIdx = offset + grid.block_rank();
        if (bufferIdx >= pathfindingList.count) {
            break;
        }

        Pathfinding info = pathfindingList.data[bufferIdx];
        FlowField &flowField = info.flowField;

        // Init buffer
        for (int roadOffset = 0; roadOffset < info.flowField.length();
             roadOffset += block.num_threads()) {
            int idx = roadOffset + block.thread_rank();
            if (idx < info.flowField.length()) {
                flowField.resetCell(idx);
            }
        }

        if (block.thread_rank() == 0) {
            // Init neighbor tiles of target
            map->neighborCells(info.target).forEach([&flowField](int cellId) {
                int fieldId = flowField.fieldId(cellId);
                if (fieldId != -1) {
                    auto &cell = flowField.getFieldCell(fieldId);
                    cell.cellId = cellId;
                    cell.distance = 0;
                }
            });
        }

        block.sync();

        auto isOriginReached = [](int origin, Map *map, Pathfinding &info) {
            return map->neighborCells(origin).oneTrue([&](int cellId) {
                int fieldId = info.flowField.fieldId(cellId);
                return fieldId != -1 &&
                       info.flowField.getFieldCell(fieldId).distance < uint32_t(Infinity);
            });
        };

        // Build flowfield
        while (!isOriginReached(info.origin, map, info)) {
            // The field is split accross the threads of the block
            for (int roadOffset = 0; roadOffset < info.flowField.length();
                 roadOffset += block.num_threads()) {
                int currentFieldId = roadOffset + block.thread_rank();
                if (currentFieldId > info.flowField.length()) {
                    break;
                }
                FieldCell &fieldCell = info.flowField.getFieldCell(currentFieldId);
                if (fieldCell.cellId == -1) {
                    // cell has not been explored yet
                    continue;
                }

                map->neighborCells(fieldCell.cellId).forEach([&](int neighborId) {
                    int neighborFieldId = flowField.fieldId(neighborId);
                    if (neighborFieldId == -1) {
                        return;
                    }

                    auto &neighborFieldCell = flowField.getFieldCell(neighborFieldId);

                    // Atomically update neighbor tiles id if not set
                    int oldNeighborId = atomicCAS(&neighborFieldCell.cellId, -1, neighborId);

                    if (oldNeighborId != -1) {
                        // Set distance value
                        fieldCell.distance =
                            min(fieldCell.distance, neighborFieldCell.distance + 1);
                    }
                });
            }
            // not sure if this is needed or not
            block.sync();
        }

        // Extract path
        if (block.thread_rank() == 0) {
            entities->get(info.entityIdx).path = info.flowField.extractPath(info.origin);
        }
    }
}

Path FlowField::extractPath(uint32_t originId) const {
    int current = originId;
    bool reached = false;
    Path path;

    while (!reached && path.length() < 29) {
        // Retrieve path
        uint32_t min = uint32_t(Infinity);
        Direction dir;
        int nextCell;

        // We assume that there is always a possible path to the target
        map->neighborCells(current).forEachDir([&](Direction neighbordDir, int neighborId) {
            if (reached) {
                return;
            }
            int fId = fieldId(neighborId);
            if (neighborId == target) {
                reached = true;
                dir = neighbordDir;
            } else if (fId != -1) {
                uint32_t distance = getFieldCell(fId).distance;
                if (distance < min) {
                    min = distance;
                    dir = neighbordDir;
                    nextCell = neighborId;
                }
            }
        });

        path.append(dir);
        current = nextCell;
    }
    return path;
}

int32_t FlowField::fieldId(uint32_t cellId) const {
    if (!isCellInField(cellId)) {
        return -1;
    }
    int offset = 0;
    int32_t cellNetwork = map->roadNetworkRepr(cellId);
    for (int i = 0; i < 4; i++) {
        int ni = networkIds.data[i];
        if (cellNetwork == ni) {
            break;
        }
        offset += map->roadNetworkId(ni);
    }
    return offset + map->roadNetworkId(cellId) % map->roadNetworkId(cellNetwork);
}

void FlowField::resetCell(uint32_t fieldId) {
    auto &fieldCell = getFieldCell(fieldId);
    fieldCell.cellId = -1;
    fieldCell.distance = uint32_t(Infinity);
}
