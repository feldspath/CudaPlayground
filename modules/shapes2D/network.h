#pragma once

struct Network {
    uint32_t *parents;

    uint32_t cellRepr(int cellId) {
        // The network is flattened, so we need to go at most 2 levels up after the component
        // update.
        return parents[parents[cellId]];
        // int current = cellId;
        // while (parents[current] != current) {
        //     current = parents[current];
        // }
        // return current;
    }

    void update(int cellId, uint32_t newComponent) { parents[cellRepr(cellId)] = newComponent; }
};
