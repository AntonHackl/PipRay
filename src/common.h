#pragma once
#include <optix.h>
#include <cuda_runtime.h>

// Simple data structures used by both host and device

struct RayGenData {
    float3 origin;
    float3 direction;
};

struct HitGroupData {
    float3 color;
};

struct RayResult {
    int hit;           // 1 if ray hit something, 0 if miss
    float t;           // distance to hit point
    float3 hit_point;  // world coordinates of hit point
    float2 barycentrics; // barycentric coordinates for triangle hit
};

struct LaunchParams {
    RayGenData ray_gen;
    HitGroupData hit_group;
    OptixTraversableHandle handle;
    RayResult* result; // Pointer to result buffer on device
}; 