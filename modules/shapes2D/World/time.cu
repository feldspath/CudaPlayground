#include "common/utils.cuh"

#include "time.h"

const TimeInterval TimeInterval::workHours = {{6, 0}, {15, 0}};
const TimeInterval TimeInterval::upgradeHours = {{15, 0}, {21, 0}};