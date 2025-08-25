#include "dataset_loader.h"
#include "triangulation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <map>
#include <array>

using namespace std;

// Simple cross-platform progress bar
void printProgressBar(size_t current, size_t total, int barWidth = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) 
              << (progress * 100.0) << "% (" << current << "/" << total << ")";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}

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
        std::cout << "Triangulating polygons..." << std::endl;
        
        // Store successfully triangulated polygons and their indices
        std::vector<std::vector<Triangle>> triangulated_polygons;
        std::vector<size_t> valid_polygon_indices;  // Track which original polygons are valid
        TriangulationStats stats;  // Track triangulation method statistics
        
        for (size_t poly_idx = 0; poly_idx < polygons.size(); ++poly_idx) {
            if (poly_idx % 100 == 0 || poly_idx == polygons.size() - 1)  {
                printProgressBar(poly_idx + 1, polygons.size());
            }
            
            const auto& poly = polygons[poly_idx];
            
            try {
                auto result = triangulatePolygon(poly);
                auto triangulated = result.first;
                int method_used = result.second;
                
                // Update statistics based on method used
                if (method_used == 0) stats.cgal_success++;
                else if (method_used == 1) stats.cgal_repaired++;
                else if (method_used == 2) stats.cgal_decomposed++;
                else if (method_used == 3) stats.failed_method++;
                
                // Check if triangulation was successful (has triangles and they're valid)
                if (triangulated.empty()) {
                    continue;
                }
                
                // Additional validation: check if triangles have valid vertices
                bool valid_triangulation = true;
                for (const auto& tri : triangulated) {
                    if ((tri.vertices[0].x == tri.vertices[1].x && tri.vertices[0].y == tri.vertices[1].y) ||
                        (tri.vertices[1].x == tri.vertices[2].x && tri.vertices[1].y == tri.vertices[2].y) ||
                        (tri.vertices[2].x == tri.vertices[0].x && tri.vertices[2].y == tri.vertices[0].y)) {
                        valid_triangulation = false;
                        break;
                    }
                }
                
                if (!valid_triangulation) {
                    continue;
                }
                
                // Triangulation successful, add to valid polygons
                triangulated_polygons.push_back(triangulated);
                valid_polygon_indices.push_back(poly_idx);
                
            } catch (const std::exception& e) {
                continue;
            }
        }
        
        // Print triangulation method statistics
        std::cout << std::endl;  // New line after progress bar
        stats.print();
        
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
        
        for (size_t valid_idx = 0; valid_idx < triangulated_polygons.size(); ++valid_idx) {
            const auto& triangles = triangulated_polygons[valid_idx];
            const size_t original_poly_idx = valid_polygon_indices[valid_idx];
            
            // Create a map to track unique vertices for this polygon
            std::map<std::pair<float, float>, size_t> vertex_map;
            std::vector<Point2D> polygon_vertices;
            
            // Process all triangles for this polygon and collect unique vertices
            for (const auto& tri : triangles) {
                for (int i = 0; i < 3; ++i) {
                    std::pair<float, float> vertex_key = {tri.vertices[i].x, tri.vertices[i].y};
                    if (vertex_map.find(vertex_key) == vertex_map.end()) {
                        vertex_map[vertex_key] = polygon_vertices.size();
                        polygon_vertices.push_back(tri.vertices[i]);
                    }
                }
            }
            
            // Add vertices to global list
            size_t vertex_offset = geometry.vertices.size();
            for (const auto& vertex : polygon_vertices) {
                geometry.vertices.push_back({vertex.x, vertex.y, 0.0f});
            }
            
            // Add triangles with correct vertex indices
            for (const auto& tri : triangles) {
                std::array<unsigned int, 3> indices;
                for (int i = 0; i < 3; ++i) {
                    std::pair<float, float> vertex_key = {tri.vertices[i].x, tri.vertices[i].y};
                    indices[i] = static_cast<unsigned int>(vertex_offset + vertex_map[vertex_key]);
                }
                
                geometry.indices.push_back({indices[0], indices[1], indices[2]});
                // Store the polygon association for this triangle (use original polygon index)
                geometry.triangleToPolygon.push_back(static_cast<int>(original_poly_idx));
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

GeometryData loadGeometryFromFile(const std::string& geometryFilePath) {
    GeometryData geometry;
    
    std::cout << "=== Loading Preprocessed Geometry ===" << std::endl;
    std::cout << "Loading geometry from: " << geometryFilePath << std::endl;
    
    std::ifstream file(geometryFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open geometry file: " << geometryFilePath << std::endl;
        return geometry;
    }
    
    std::string line;
    
    // Read vertices line
    if (std::getline(file, line)) {
        if (line.substr(0, 9) == "vertices:") {
            std::string vertices_data = line.substr(9); // Remove "vertices:" prefix
            std::stringstream ss(vertices_data);
            float x, y, z;
            while (ss >> x >> y >> z) {
                geometry.vertices.push_back({x, y, z});
            }
        } else {
            std::cerr << "Error: Expected vertices line first" << std::endl;
            return GeometryData{};
        }
    }
    
    // Read indices line
    if (std::getline(file, line)) {
        if (line.substr(0, 8) == "indices:") {
            std::string indices_data = line.substr(8); // Remove "indices:" prefix
            std::stringstream ss(indices_data);
            unsigned int i1, i2, i3;
            while (ss >> i1 >> i2 >> i3) {
                geometry.indices.push_back({i1, i2, i3});
            }
        } else {
            std::cerr << "Error: Expected indices line second" << std::endl;
            return GeometryData{};
        }
    }
    
    // Read triangleToPolygon line
    if (std::getline(file, line)) {
        if (line.substr(0, 18) == "triangleToPolygon:") {
            std::string mapping_data = line.substr(18); // Remove "triangleToPolygon:" prefix
            std::stringstream ss(mapping_data);
            int polygonId;
            while (ss >> polygonId) {
                geometry.triangleToPolygon.push_back(polygonId);
            }
        } else {
            std::cerr << "Error: Expected triangleToPolygon line third" << std::endl;
            return GeometryData{};
        }
    }
    
    // Read total_triangles line
    if (std::getline(file, line)) {
        if (line.substr(0, 16) == "total_triangles:") {
            std::string total_data = line.substr(16); // Remove "total_triangles:" prefix
            std::stringstream ss(total_data);
            ss >> geometry.totalTriangles;
        } else {
            std::cerr << "Error: Expected total_triangles line fourth" << std::endl;
            return GeometryData{};
        }
    }
    
    file.close();
    
    std::cout << "Loaded preprocessed geometry:" << std::endl;
    std::cout << "  Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "  Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "  Triangle-to-polygon mappings: " << geometry.triangleToPolygon.size() << std::endl;
    std::cout << "=============================\n" << std::endl;
    
    return geometry;
} 