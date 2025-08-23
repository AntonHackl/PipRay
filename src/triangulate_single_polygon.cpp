// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "triangulation.h"
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::string datasetPath = "dtl_cnty.wkt";
    int targetPolygonIndex = 6; // Sixth polygon (0-indexed)
    
    if (argc > 1) {
        datasetPath = argv[1];
    }
    if (argc > 2) {
        targetPolygonIndex = std::atoi(argv[2]);
    }
    
    std::cout << "Reading polygons from: " << datasetPath << std::endl;
    std::cout << "Target polygon index: " << targetPolygonIndex << " (polygon #" << (targetPolygonIndex + 1) << ")" << std::endl;
    
    // Read all polygons from the file
    std::vector<PolygonWithHoles> allPolygons = readPolygonVerticesFromFile(datasetPath);
    
    if (allPolygons.empty()) {
        std::cerr << "No polygons found in the file!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << allPolygons.size() << " polygons" << std::endl;
    
    if (targetPolygonIndex >= allPolygons.size()) {
        std::cerr << "Requested polygon index " << targetPolygonIndex << " is out of range (0-" << (allPolygons.size() - 1) << ")" << std::endl;
        return 1;
    }
    
    // Get the target polygon
    const PolygonWithHoles& targetPolygon = allPolygons[targetPolygonIndex];
    
    std::cout << "Processing polygon #" << (targetPolygonIndex + 1) << ":" << std::endl;
    std::cout << "  Exterior vertices: " << targetPolygon.outer.size() << std::endl;
    std::cout << "  Interior holes: " << targetPolygon.holes.size() << std::endl;
    
    // Triangulate the polygon
    std::cout << "Triangulating polygon..." << std::endl;
    CDT::TriangleVec triangles = triangulatePolygon(targetPolygon);
    
    std::cout << "Triangulation complete:" << std::endl;
    std::cout << "  Triangles: " << triangles.size() << std::endl;
    
    // Save results to a simple text format for Python visualization
    std::ofstream outputFile("single_polygon_triangulation.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file for writing!" << std::endl;
        return 1;
    }
    
    // Set precision for floating point output
    outputFile << std::fixed << std::setprecision(6);
    
    // Write header
    outputFile << "# Single Polygon Triangulation Data" << std::endl;
    outputFile << "# Format: polygon_index polygon_number" << std::endl;
    outputFile << targetPolygonIndex << " " << (targetPolygonIndex + 1) << std::endl;
    
    // Write original polygon data
    outputFile << "# Original polygon exterior vertices" << std::endl;
    outputFile << "# Format: num_vertices" << std::endl;
    outputFile << targetPolygon.outer.size() << std::endl;
    outputFile << "# Format: x y" << std::endl;
    for (const auto& vertex : targetPolygon.outer) {
        outputFile << vertex.x << " " << vertex.y << std::endl;
    }
    
    // Write holes
    outputFile << "# Original polygon holes" << std::endl;
    outputFile << "# Format: num_holes" << std::endl;
    outputFile << targetPolygon.holes.size() << std::endl;
    
    for (const auto& hole : targetPolygon.holes) {
        outputFile << "# Hole with " << hole.size() << " vertices" << std::endl;
        outputFile << hole.size() << std::endl;
        for (const auto& vertex : hole) {
            outputFile << vertex.x << " " << vertex.y << std::endl;
        }
    }
    
    // Write triangulation data
    outputFile << "# Triangulation vertices (exterior + holes)" << std::endl;
    std::vector<CDT::V2d<float>> allVertices = targetPolygon.outer;
    for (const auto& hole : targetPolygon.holes) {
        allVertices.insert(allVertices.end(), hole.begin(), hole.end());
    }
    
    outputFile << "# Format: num_vertices" << std::endl;
    outputFile << allVertices.size() << std::endl;
    outputFile << "# Format: x y" << std::endl;
    for (const auto& vertex : allVertices) {
        outputFile << vertex.x << " " << vertex.y << std::endl;
    }
    
    // Write triangles
    outputFile << "# Triangulation triangles" << std::endl;
    outputFile << "# Format: num_triangles" << std::endl;
    outputFile << triangles.size() << std::endl;
    outputFile << "# Format: v1 v2 v3" << std::endl;
    for (const auto& triangle : triangles) {
        outputFile << triangle.vertices[0] << " " << triangle.vertices[1] << " " << triangle.vertices[2] << std::endl;
    }
    
    // Write boundary segments (constraints)
    outputFile << "# Boundary segments (constraints)" << std::endl;
    
    // Count total segments
    size_t totalSegments = targetPolygon.outer.size();
    for (const auto& hole : targetPolygon.holes) {
        totalSegments += hole.size();
    }
    
    outputFile << "# Format: num_segments" << std::endl;
    outputFile << totalSegments << std::endl;
    outputFile << "# Format: v1 v2" << std::endl;
    
    // Exterior boundary segments
    for (size_t i = 0; i < targetPolygon.outer.size(); ++i) {
        outputFile << i << " " << ((i + 1) % targetPolygon.outer.size()) << std::endl;
    }
    
    // Hole boundary segments
    size_t vertexOffset = targetPolygon.outer.size();
    for (const auto& hole : targetPolygon.holes) {
        for (size_t i = 0; i < hole.size(); ++i) {
            outputFile << (vertexOffset + i) << " " << (vertexOffset + ((i + 1) % hole.size())) << std::endl;
        }
        vertexOffset += hole.size();
    }
    
    outputFile.close();
    
    std::cout << "Results saved to: single_polygon_triangulation.txt" << std::endl;
    std::cout << "You can now run the Python visualization script to view the results." << std::endl;
    
    return 0;
} 