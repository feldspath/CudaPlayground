#pragma once

#include "builtin_types.h"

constexpr float PI = 3.1415;

float4 operator*(const mat4 &a, const float4 &b) {
    return make_float4(dot(a.rows[0], b), dot(a.rows[1], b), dot(a.rows[2], b), dot(a.rows[3], b));
}

mat4 operator*(const mat4 &a, const mat4 &b) {

    mat4 result;

    result.rows[0].x = dot(a.rows[0], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[0].y = dot(a.rows[0], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[0].z = dot(a.rows[0], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[0].w = dot(a.rows[0], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[1].x = dot(a.rows[1], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[1].y = dot(a.rows[1], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[1].z = dot(a.rows[1], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[1].w = dot(a.rows[1], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[2].x = dot(a.rows[2], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[2].y = dot(a.rows[2], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[2].z = dot(a.rows[2], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[2].w = dot(a.rows[2], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    result.rows[3].x = dot(a.rows[3], {b.rows[0].x, b.rows[1].x, b.rows[2].x, b.rows[3].x});
    result.rows[3].y = dot(a.rows[3], {b.rows[0].y, b.rows[1].y, b.rows[2].y, b.rows[3].y});
    result.rows[3].z = dot(a.rows[3], {b.rows[0].z, b.rows[1].z, b.rows[2].z, b.rows[3].z});
    result.rows[3].w = dot(a.rows[3], {b.rows[0].w, b.rows[1].w, b.rows[2].w, b.rows[3].w});

    return result;
}

float3 unproject(float2 pixelCoords, mat4 unprojection, int screenWidth, int screenHeight) {
    float4 ndc = {pixelCoords.x / screenWidth * 2.0f - 1.0f,
                  pixelCoords.y / screenHeight * 2.0f - 1.0f, 1.0, 1.0};
    float4 worldPos = unprojection * ndc;
    float3 pos = {worldPos.x / worldPos.w, worldPos.y / worldPos.w, worldPos.z / worldPos.w};
    return pos;
}

float2 projectPosToScreenPos(float3 worldPos, mat4 projection, int screenWidth, int screenHeight) {
    float4 worldCoord = make_float4(worldPos, 1.0f);
    float4 screenCoord = projection * worldCoord;

    float2 ndc = {screenCoord.x / screenCoord.w, screenCoord.y / screenCoord.w};

    return {(ndc.x + 1.0f) * 0.5f * screenWidth, (ndc.y + 1.0f) * 0.5f * screenHeight};
}

float2 projectVectorToScreenPos(float3 vector, mat4 projection, int screenWidth, int screenHeight) {
    float4 worldCoord = make_float4(vector, 0.0f);
    float4 screenCoord = projection * worldCoord;

    return float2{screenCoord.x * screenWidth * 0.5f, screenCoord.y * screenHeight * 0.5f};
}