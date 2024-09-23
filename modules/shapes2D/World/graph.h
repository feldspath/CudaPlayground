#pragma once
#include "cell.h"

constexpr int MAX_NEIGHBORS = 64;
constexpr int MAX_NETWORK_NODES = 200;

struct NetworkNode {
    MapId networkRepr;
    int numNeighbors;
    int neighborIds[MAX_NEIGHBORS];

    NetworkNode() : networkRepr(-1), numNeighbors(0) {}
    NetworkNode(MapId network) : networkRepr(network), numNeighbors(0) {}
};
