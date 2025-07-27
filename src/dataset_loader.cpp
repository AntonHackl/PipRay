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

PointData loadPointDataset(const std::string& pointDatasetPath) {
    PointData pointData;
    
    if (pointDatasetPath.empty()) {
        std::cout << "No point dataset provided, using default test points" << std::endl;
        // Default test points if no dataset provided
        pointData.positions = {
            {0.0f, 0.0f, -1.0f},
            {0.5f, 0.5f, -1.0f},
            {1.0f, 1.0f, -1.0f}
        };
        pointData.numPoints = 3;
        return pointData;
    }
    
    std::cout << "=== Loading Point Dataset ===" << std::endl;
    std::cout << "Loading points from: " << pointDatasetPath << std::endl;
    
    std::ifstream file(pointDatasetPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open point dataset file: " << pointDatasetPath << std::endl;
        return pointData;
    }
    
    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Look for POINT entries
        if (line.find("POINT") != std::string::npos) {
            try {
                // Parse the WKT POINT format: POINT(x y)
                size_t start = line.find('(');
                size_t end = line.find(')');
                if (start != std::string::npos && end != std::string::npos) {
                    std::string coords = line.substr(start + 1, end - start - 1);
                    
                    // Split by space to get x and y coordinates
                    size_t spacePos = coords.find(' ');
                    if (spacePos != std::string::npos) {
                        float x = std::stof(coords.substr(0, spacePos));
                        float y = std::stof(coords.substr(spacePos + 1));
                        
                        // Create ray origin at (x, y, -1)
                        pointData.positions.push_back({x, y, -1.0f});
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse point on line " << lineNum << ": " << e.what() << std::endl;
                continue;
            }
        }
    }
    
    pointData.numPoints = pointData.positions.size();
    std::cout << "Loaded " << pointData.numPoints << " points from dataset" << std::endl;
    std::cout << "=============================\n" << std::endl;
    
    return pointData;
} 