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

        std::vector<PolygonWithHoles> polygons = readPolygonVerticesFromFile(datasetPath);
        if (polygons.empty()) {
            std::cerr << "Error: No valid polygons found in dataset file." << std::endl;
            return geometry;
        }
        
        std::cout << "Found " << polygons.size() << " polygons in dataset" << std::endl;
        
        // Store successfully triangulated polygons and their indices
        std::vector<CDT::TriangleVec> triangulated_polygons;
        std::vector<size_t> valid_polygon_indices;  // Track which original polygons are valid
        
        for (size_t poly_idx = 0; poly_idx < polygons.size(); ++poly_idx) {
            const auto& poly = polygons[poly_idx];
            
            try {
                auto triangulated = triangulatePolygon(poly);
                
                // Check if triangulation was successful (has triangles and they're valid)
                if (triangulated.empty()) {
                    std::cout << "Warning: Polygon " << poly_idx << " triangulation failed (empty result) - skipping" << std::endl;
                    continue;
                }
                
                // Additional validation: check if triangles have valid indices
                bool valid_triangulation = true;
                for (const auto& tri : triangulated) {
                    if (tri.vertices[0] == tri.vertices[1] || 
                        tri.vertices[1] == tri.vertices[2] || 
                        tri.vertices[2] == tri.vertices[0]) {
                        valid_triangulation = false;
                        break;
                    }
                }
                
                if (!valid_triangulation) {
                    std::cout << "Warning: Polygon " << poly_idx << " has degenerated triangles - skipping" << std::endl;
                    continue;
                }
                
                // Triangulation successful, add to valid polygons
                triangulated_polygons.push_back(triangulated);
                valid_polygon_indices.push_back(poly_idx);
                
            } catch (const std::exception& e) {
                std::cout << "Warning: Polygon " << poly_idx << " triangulation failed with exception: " << e.what() << " - skipping" << std::endl;
                continue;
            }
        }
        
        if (triangulated_polygons.empty()) {
            std::cerr << "Error: No polygons could be successfully triangulated." << std::endl;
            return geometry;
        }
        
        std::cout << "Successfully triangulated " << triangulated_polygons.size() << " out of " << polygons.size() << " polygons" << std::endl;
        geometry.totalTriangles = countTriangles(triangulated_polygons);
        
        std::cout << "Triangulation completed successfully!" << std::endl;
        std::cout << "Total number of triangles: " << geometry.totalTriangles << std::endl;
        std::cout << "=============================\n" << std::endl;
        
        // Convert to OptiX format - only for valid polygons
        std::cout << "Converting dataset triangles to OptiX format..." << std::endl;
        
        size_t vertexOffset = 0;
        for (size_t valid_idx = 0; valid_idx < triangulated_polygons.size(); ++valid_idx) {
            const auto& triangles = triangulated_polygons[valid_idx];
            const size_t original_poly_idx = valid_polygon_indices[valid_idx];
            const auto& polygon = polygons[original_poly_idx];
            
            // Add outer ring vertices
            for (const auto& vertex : polygon.outer) {
                geometry.vertices.push_back({vertex.x, vertex.y, 0.0f});
            }
            // Add hole vertices
            for (const auto& hole : polygon.holes) {
                for (const auto& vertex : hole) {
                    geometry.vertices.push_back({vertex.x, vertex.y, 0.0f});
                }
            }
            
            for (const auto& tri : triangles) {
                geometry.indices.push_back({
                    static_cast<unsigned int>(vertexOffset + tri.vertices[0]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[1]),
                    static_cast<unsigned int>(vertexOffset + tri.vertices[2])
                });
                // Store the polygon association for this triangle (use original polygon index)
                geometry.triangleToPolygon.push_back(static_cast<int>(original_poly_idx));
            }
            
            // Update vertex offset for next polygon (outer + all holes)
            vertexOffset += polygon.outer.size();
            for (const auto& hole : polygon.holes) {
                vertexOffset += hole.size();
            }
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
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (line.find("POINT") != std::string::npos) {
            try {
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