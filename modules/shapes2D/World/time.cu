#include "common/utils.cuh"

#include "time.h"

const TimeInterval TimeInterval::factoryHours = {{6, 0}, {15, 0}};
const TimeInterval TimeInterval::shopHours = {{10, 0}, {19, 0}};