// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "common.h"
#include "triangulation.h"

constexpr const char* ptxPath = "C:/Users/anton/Documents/Uni/PipRay/build/raytracing.ptx";

static std::vector<char> readPTX(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

#define OPTIX_CHECK(call) do { OptixResult res = call; if(res!=OPTIX_SUCCESS){ std::cerr << "OptiX error " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(EXIT_FAILURE);} } while(0)
#define CUDA_CHECK(call) do { cudaError_t err = call; if(err!=cudaSuccess){ std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(EXIT_FAILURE);} } while(0)

int main(int argc, char* argv[])
{
    std::string datasetPath = "";
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                datasetPath = argv[++i];
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--dataset <path_to_wkt_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --dataset <path>   Path to WKT dataset file to triangulate" << std::endl;
                std::cout << "  --help, -h         Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    std::vector<std::vector<CDT::V2d<float>>> polygons;
    std::vector<CDT::TriangleVec> triangulated_polygons;
    
    if (!datasetPath.empty()) {
        std::cout << "=== Dataset Triangulation ===" << std::endl;
        std::cout << "Loading polygons from: " << datasetPath << std::endl;
        
        polygons = readPolygonVerticesFromFile(datasetPath);
        if (polygons.empty()) {
            std::cerr << "Error: No valid polygons found in dataset file." << std::endl;
            return 1;
        }
        
        std::cout << "Found " << polygons.size() << " polygons in dataset" << std::endl;
        
        for (const auto& poly : polygons) {
            auto triangulated = triangulatePolygon(poly);
            triangulated_polygons.push_back(triangulated);
        }
        
        size_t totalTriangles = countTriangles(triangulated_polygons);
        
        std::cout << "Triangulation completed successfully!" << std::endl;
        std::cout << "Total number of triangles: " << totalTriangles << std::endl;
        std::cout << "=============================\n" << std::endl;
    }

    std::cout << "OptiX single ray example" << std::endl;

    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &context));

    std::vector<float3> vertices;
    std::vector<uint3> indices;
    
    if (!datasetPath.empty()) {
        std::cout << "Converting dataset triangles to OptiX format..." << std::endl;
        
        size_t vertexOffset = 0;
        for (size_t poly_idx = 0; poly_idx < triangulated_polygons.size(); ++poly_idx) {
            const auto& triangles = triangulated_polygons[poly_idx];
            const auto& polygon_vertices = polygons[poly_idx];
            
            for (const auto& vertex : polygon_vertices) {
                vertices.push_back({vertex.x, vertex.y, 0.0f});
            }
            
            for (const auto& tri : triangles) {
                indices.push_back({
                    static_cast<unsigned int>(vertexOffset + tri.vertices[0]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[1]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[2])
                });
            }
            
            vertexOffset += polygon_vertices.size();
        }
        
        std::cout << "Dataset converted to " << vertices.size() << " vertices and " << indices.size() << " triangles" << std::endl;
    } else {
        // Use hardcoded triangle if no dataset provided
        // This is just for testing
        vertices = {
            {0.0f, 0.0f, 0.0f},
            {0.5f, 1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        indices = { {0,1,2} };
    }
    
    std::cout << "Geometry loaded: " << vertices.size() << " vertices, " << indices.size() << " triangles" << std::endl;
    
    if (!datasetPath.empty()) {
        std::cout << "Using dataset triangles for raytracing acceleration structure" << std::endl;
    } else {
        std::cout << "Triangle vertices:" << std::endl;
        std::cout << "  V0: (" << vertices[0].x << ", " << vertices[0].y << ", " << vertices[0].z << ")" << std::endl;
        std::cout << "  V1: (" << vertices[1].x << ", " << vertices[1].y << ", " << vertices[1].z << ")" << std::endl;
        std::cout << "  V2: (" << vertices[2].x << ", " << vertices[2].y << ", " << vertices[2].z << ")" << std::endl;
    }

    float3* d_vertices = nullptr;
    uint3*  d_indices  = nullptr;
    size_t vbytes = vertices.size()*sizeof(float3);
    size_t ibytes = indices.size()*sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices),vbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices),ibytes));
    CUDA_CHECK(cudaMemcpy(d_vertices,vertices.data(),vbytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices,indices.data(),ibytes,cudaMemcpyHostToDevice));

    CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
    CUdeviceptr d_indices_ptr  = reinterpret_cast<CUdeviceptr>(d_indices);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    buildInput.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.indexBuffer = d_indices_ptr;
    buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size());
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags        = &triangle_input_flags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context,&accelOptions,&buildInput,1,&gasSizes));

    CUdeviceptr d_tempBuffer,d_gasOutput;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer),gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput),gasSizes.outputSizeInBytes));

    OptixTraversableHandle gasHandle = 0;
    OPTIX_CHECK(optixAccelBuild(context,0,&accelOptions,&buildInput,1,d_tempBuffer,gasSizes.tempSizeInBytes,d_gasOutput,gasSizes.outputSizeInBytes,&gasHandle,nullptr,0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));

    std::vector<char> ptxData = readPTX(ptxPath);
    std::cout << "PTX file loaded successfully, size: " << ptxData.size() << " bytes" << std::endl;

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur        = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 2;
    pipelineCompileOptions.numAttributeValues    = 2;
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixModule module = nullptr;
    char log[8192];
    size_t sizeof_log = sizeof(log);
    
    std::cout << "Creating OptiX module..." << std::endl;
    OptixResult result = optixModuleCreate(context,
                                  &moduleCompileOptions,
                                  &pipelineCompileOptions,
                                  ptxData.data(),
                                  ptxData.size(),
                                  log,
                                  &sizeof_log,
                                  &module);
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed with error code: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
    
    if (sizeof_log > 1) {
        std::cout << "OptiX module log:\n" << log << std::endl;
    }
    std::cout << "OptiX module created successfully!" << std::endl;

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";
    OptixProgramGroup raygenPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&raygenDesc,1,&pgOptions,nullptr,nullptr,&raygenPG));

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OptixProgramGroup missPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&missDesc,1,&pgOptions,nullptr,nullptr,&missPG));

    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup hitPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&hitDesc,1,&pgOptions,nullptr,nullptr,&hitPG));

    std::vector<OptixProgramGroup> pgs = { raygenPG, missPG, hitPG };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(context,&pipelineCompileOptions,&linkOptions,pgs.data(),pgs.size(),nullptr,nullptr,&pipeline));

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; RayGenData data; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord    { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; HitGroupData data; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord     { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; HitGroupData data; };

    RaygenRecord rgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG,&rgRecord));
    rgRecord.data.origin = {0.0f,0.0f,0.0f};
    rgRecord.data.direction = {0.0f,0.0f,1.0f};

    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG,&msRecord));

    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG,&hgRecord));

    CUdeviceptr d_rg,d_ms,d_hg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg),sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms),sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg),sizeof(HitRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg),&rgRecord,sizeof(RaygenRecord),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms),&msRecord,sizeof(MissRecord),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg),&hgRecord,sizeof(HitRecord),cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord            = d_rg;
    sbt.missRecordBase          = d_ms;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = 1;
    sbt.hitgroupRecordBase      = d_hg;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt.hitgroupRecordCount     = 1;

    RayResult h_result = {};
    CUdeviceptr d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(RayResult)));
    CUDA_CHECK(cudaMemcpy((void*)d_result, &h_result, sizeof(RayResult), cudaMemcpyHostToDevice));

    CUdeviceptr d_lp;
    CUDA_CHECK(cudaMalloc((void**)&d_lp,sizeof(LaunchParams)));

    struct TestRay {
        float3 origin;
        float3 direction;
        const char* description;
    };

    TestRay testRays[] = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, "Along X-axis"},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, "Along Y-axis"}, 
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "Along Z-axis"},
        {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, "Diagonal (1,1,1)"},
        {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, "From (1,1,1) to (2,2,2) direction"},
        {{2.5f, 2.5f, 1.0f}, {0.0f, 0.0f, 1.0f}, "From triangle center in Z direction"},
        {{0.5f, 0.5f, -0.1f}, {0.0f, 0.0f, 0.1f}, "Three dimensional ray for triangles in x-y plane"}
    };

    for (int i = 0; i < sizeof(testRays) / sizeof(TestRay); ++i) {
        LaunchParams lp = {};
        lp.ray_gen.origin = testRays[i].origin;
        
        float3 dir = testRays[i].direction;
        float length = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        lp.ray_gen.direction = {dir.x/length, dir.y/length, dir.z/length};
        lp.handle = gasHandle;
        lp.result = reinterpret_cast<RayResult*>(d_result);

        h_result = {};
        CUDA_CHECK(cudaMemcpy((void*)d_result, &h_result, sizeof(RayResult), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_lp,&lp,sizeof(LaunchParams),cudaMemcpyHostToDevice));

        std::cout << "\n=== Test Ray " << (i+1) << ": " << testRays[i].description << " ===" << std::endl;
        std::cout << "Ray origin: (" << lp.ray_gen.origin.x << ", " << lp.ray_gen.origin.y << ", " << lp.ray_gen.origin.z << ")" << std::endl;
        std::cout << "Ray direction: (" << lp.ray_gen.direction.x << ", " << lp.ray_gen.direction.y << ", " << lp.ray_gen.direction.z << ")" << std::endl;
        
        OPTIX_CHECK(optixLaunch(pipeline,0,d_lp,sizeof(LaunchParams),&sbt,1,1,1));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_result, (void*)d_result, sizeof(RayResult), cudaMemcpyDeviceToHost));

        if (h_result.hit) {
            std::cout << "Ray HIT the triangle!" << std::endl;
            std::cout << "  Distance: " << h_result.t << std::endl;
            std::cout << "  Hit point: (" << h_result.hit_point.x << ", " << h_result.hit_point.y << ", " << h_result.hit_point.z << ")" << std::endl;
            std::cout << "  Barycentric coordinates: (" << h_result.barycentrics.x << ", " << h_result.barycentrics.y << ")" << std::endl;
        } else {
            std::cout << "Ray MISSED the triangle" << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_result)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput)));
    CUDA_CHECK(cudaFree(d_vertices));
    CUDA_CHECK(cudaFree(d_indices));

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygenPG);
    optixProgramGroupDestroy(missPG);
    optixProgramGroupDestroy(hitPG);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);

    return 0;
} 