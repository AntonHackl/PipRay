#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include "common.h"

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int ray_id = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (ray_id >= params.num_rays) {
        return;
    }
    
    const float3 orig = params.ray_origins[ray_id];
    const float3 dir = params.ray_directions[ray_id];

    unsigned int p0 = 0;  // hit flag (0 = miss, 1 = hit)
    unsigned int p1 = __float_as_uint(0.0f);

    optixTrace(
        params.handle,
        orig,
        dir,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        /*SBT offset*/ 0,
        /*SBT stride*/ 1,
        /*missSBTIndex*/ 0,
        /*payload*/ p0, p1);

    if (params.result) {
        params.result[ray_id].hit = p0;
        params.result[ray_id].t = __uint_as_float(p1);
        
        if (p0) {
            float t = __uint_as_float(p1);
            params.result[ray_id].hit_point = make_float3(
                orig.x + t * dir.x,
                orig.y + t * dir.y,
                orig.z + t * dir.z
            );
        } else {
            params.result[ray_id].hit_point = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0); // hit = 0 (miss)
    optixSetPayload_1(__float_as_uint(0.0f)); // t = 0.0f
}

extern "C" __global__ void __closesthit__ch()
{
    const float2 bc = optixGetTriangleBarycentrics();
    const float t = optixGetRayTmax();
    
    optixSetPayload_0(1); // hit = 1 (hit)
    optixSetPayload_1(__float_as_uint(t)); // distance to hit point
    
    if (params.result) {
        params.result->barycentrics = bc;
    }
} 