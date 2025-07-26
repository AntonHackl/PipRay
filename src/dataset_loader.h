#pragma once

#include <string>
#include <vector>
#include <optix.h>
#include <cuda_runtime.h>

struct GeometryData {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<int> triangleToPolygon;  // Maps triangle index to polygon index
    size_t totalTriangles;
};

GeometryData loadDatasetGeometry(const std::string& datasetPath); 