#include "chunk.h"

void Chunk::assignEntityToWorkplace(int houseId, int workplaceCellId) {
    getTyped<HouseCell>(houseId).residentCount = getTyped<HouseCell>(houseId).residentCount + 1;
    getTyped<WorkplaceCell>(workplaceCellId).workplaceCapacity -= 1;
}