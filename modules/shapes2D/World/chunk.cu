#include "chunk.h"

// Currently this function uses the whole grid, but maybe it's better to only use a group for that.
void Chunk::updateNetworkComponents(int invalidNetwork, Allocator &allocator) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    bool *validCells = allocator.alloc<bool *>(sizeof(bool) * CHUNK_SIZE);

    processEachCell(ROAD, [&](int cellId) { validCells[cellId] = true; });

    // reset relevant network
    processEachCell(ROAD, [&](int cellId) {
        if (roadNetworkRepr(cellId) == invalidNetwork) {
            roadNetworkRepr(cellId) = cellId;
            validCells[cellId] = false;
        }
    });
    grid.sync();

    // recompute connected components
    // https://largo.lip6.fr/~lacas/Publications/IPTA17.pdf
    bool &changed = *allocator.alloc<bool *>(sizeof(bool));
    for (int i = 0; i < 10; ++i) {
        if (grid.thread_rank() == 0) {
            changed = false;
        }
        grid.sync();
        processEachCell(ROAD, [&](int cellId) {
            if (validCells[cellId]) {
                return;
            }
            int m = neighborNetworks(cellId).min();
            int old = atomicMin(&roadNetworkRepr(roadNetworkRepr(cellId)), m);
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
        processEachCell(ROAD, [&](int cellId) {
            if (validCells[cellId]) {
                return;
            }
            int network = roadNetworkRepr(cellId);
            while (network != roadNetworkRepr(network)) {
                network = roadNetworkRepr(network);
            }
            roadNetworkRepr(cellId) = network;
        });
    }

    grid.sync();
}

void Chunk::assignEntityToWorkplace(int houseId, int workplaceCellId) {
    getTyped<HouseCell>(houseId).residentCount = getTyped<HouseCell>(houseId).residentCount + 1;
    getTyped<WorkplaceCell>(workplaceCellId).workplaceCapacity -= 1;
}