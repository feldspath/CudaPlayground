#include "common/helper_math.h"
#include "common/utils.cuh"

#include "direction.h"

Direction enumFromCoord(int2 coord) {
    static Direction mapping[] = {DIAG_DL, RIGHT, LEFT, UP, DIAG_UR, DIAG_UL, DOWN, DIAG_DR};
    int2 a = abs(coord);
    // magic hash function
    int2 h = a * (2 - (a + coord) / 2);
    return mapping[h.x + h.y * 3];
}