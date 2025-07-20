#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include "common.h"

// Launch parameters (constant memory)
extern "C" __constant__ LaunchParams params;

// Ray generation program: trace single ray stored in params
extern "C" __global__ void __raygen__rg()
{
    const float3 orig = params.ray_gen.origin;
    const float3 dir  = params.ray_gen.direction;

    // Initialize payload values (hit=0, t=0.0f)
    unsigned int p0 = 0;  // hit flag (0 = miss, 1 = hit)
    unsigned int p1 = __float_as_uint(0.0f); // distance t

    optixTrace(
        params.handle,
        orig,
        dir,
        0.0f,                 // tmin
        1e16f,                // tmax
        0.0f,                 // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        /*SBT offset*/ 0,
        /*SBT stride*/ 1,
        /*missSBTIndex*/ 0,
        /*payload*/ p0, p1);

    // Store results in global memory
    if (params.result) {
        params.result->hit = p0;
        params.result->t = __uint_as_float(p1);
        
        // Calculate hit point if we hit something
        if (p0) {
            float t = __uint_as_float(p1);
            params.result->hit_point = make_float3(
                orig.x + t * dir.x,
                orig.y + t * dir.y,
                orig.z + t * dir.z
            );
        } else {
            params.result->hit_point = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

// Miss program: ray missed everything
extern "C" __global__ void __miss__ms()
{
    // Set payload to indicate miss
    optixSetPayload_0(0); // hit = 0 (miss)
    optixSetPayload_1(__float_as_uint(0.0f)); // t = 0.0f
}

// Closest hit program: ray hit the triangle
extern "C" __global__ void __closesthit__ch()
{
    const float2 bc = optixGetTriangleBarycentrics();
    const float t = optixGetRayTmax();
    
    // Set payload to indicate hit
    optixSetPayload_0(1); // hit = 1 (hit)
    optixSetPayload_1(__float_as_uint(t)); // distance to hit point
    
    // Store barycentric coordinates in the result if available
    if (params.result) {
        params.result->barycentrics = bc;
    }
} 