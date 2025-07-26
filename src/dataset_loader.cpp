#include "dataset_loader.h"
#include "triangulation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

GeometryData loadDatasetGeometry(const std::string& datasetPath) {
    GeometryData geometry;
    
    if (!datasetPath.empty()) {
        std::cout << "=== Dataset Triangulation ===" << std::endl;
        std::cout << "Loading polygons from: " << datasetPath << std::endl;
        
        std::vector<std::vector<CDT::V2d<float>>> polygons = readPolygonVerticesFromFile(datasetPath);
        if (polygons.empty()) {
            std::cerr << "Error: No valid polygons found in dataset file." << std::endl;
            return geometry;
        }
        
        std::cout << "Found " << polygons.size() << " polygons in dataset" << std::endl;
        
        std::vector<CDT::TriangleVec> triangulated_polygons;
        for (const auto& poly : polygons) {
            auto triangulated = triangulatePolygon(poly);
            triangulated_polygons.push_back(triangulated);
        }
        
        geometry.totalTriangles = countTriangles(triangulated_polygons);
        
        std::cout << "Triangulation completed successfully!" << std::endl;
        std::cout << "Total number of triangles: " << geometry.totalTriangles << std::endl;
        std::cout << "=============================\n" << std::endl;
        
        // Convert to OptiX format
        std::cout << "Converting dataset triangles to OptiX format..." << std::endl;
        
        size_t vertexOffset = 0;
        for (size_t poly_idx = 0; poly_idx < triangulated_polygons.size(); ++poly_idx) {
            const auto& triangles = triangulated_polygons[poly_idx];
            const auto& polygon_vertices = polygons[poly_idx];
            
            for (const auto& vertex : polygon_vertices) {
                geometry.vertices.push_back({vertex.x, vertex.y, 0.0f});
            }
            
            for (const auto& tri : triangles) {
                geometry.indices.push_back({
                    static_cast<unsigned int>(vertexOffset + tri.vertices[0]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[1]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[2])
                });
                // Store the polygon association for this triangle
                geometry.triangleToPolygon.push_back(static_cast<int>(poly_idx));
            }
            
            vertexOffset += polygon_vertices.size();
        }
        
        std::cout << "Dataset converted to " << geometry.vertices.size() << " vertices and " << geometry.indices.size() << " triangles" << std::endl;
        std::cout << "Using dataset triangles for raytracing acceleration structure" << std::endl;
    } else {
        // Use hardcoded triangle if no dataset provided
        // This is just for testing
        geometry.vertices = {
            {0.0f, 0.0f, 0.0f},
            {0.5f, 1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        };
        geometry.indices = { {0,1,2} };
        geometry.triangleToPolygon = { 0 };  // Single triangle belongs to polygon 0
        geometry.totalTriangles = 1;
        
        std::cout << "Triangle vertices:" << std::endl;
        std::cout << "  V0: (" << geometry.vertices[0].x << ", " << geometry.vertices[0].y << ", " << geometry.vertices[0].z << ")" << std::endl;
        std::cout << "  V1: (" << geometry.vertices[1].x << ", " << geometry.vertices[1].y << ", " << geometry.vertices[1].z << ")" << std::endl;
        std::cout << "  V2: (" << geometry.vertices[2].x << ", " << geometry.vertices[2].y << ", " << geometry.vertices[2].z << ")" << std::endl;
    }
    
    std::cout << "Geometry loaded: " << geometry.vertices.size() << " vertices, " << geometry.indices.size() << " triangles" << std::endl;
    
    return geometry;
} 