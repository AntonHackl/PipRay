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

struct PointData {
    std::vector<float3> positions;  // Ray origins (x, y, -1)
    size_t numPoints;
};

GeometryData loadDatasetGeometry(const std::string& datasetPath);
PointData loadPointDataset(const std::string& pointDatasetPath); 